// src/selfplay/self_play_manager.cpp
#include "alphazero/selfplay/self_play_manager.h"
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include "alphazero/mcts/transposition_table.h"

namespace alphazero {
namespace selfplay {

SelfPlayManager::SelfPlayManager(
    nn::NeuralNetwork* neuralNetwork, 
    int numGames, 
    int numSimulations, 
    int numThreads)
    : neuralNetwork_(neuralNetwork),
      numGames_(numGames),
      numSimulations_(numSimulations),
      numThreads_(numThreads),
      dirichletAlpha_(0.03f),
      dirichletEpsilon_(0.25f),
      initialTemperature_(1.0f),
      temperatureDropMove_(30),
      finalTemperature_(0.0f),
      saveGames_(false),
      outputDir_("games"),
      progressCallback_(nullptr),
      abort_(false),
      running_(false) {
}

SelfPlayManager::~SelfPlayManager() {
    // Stop any running self-play
    setAbort(true);
    
    // Wait until not running
    while (isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::vector<GameRecord> SelfPlayManager::generateGames(
    core::GameType gameType, 
    int boardSize,
    bool useVariantRules) {
    
    // Set running flag
    running_ = true;
    abort_ = false;
    
    // Create output directory if saving games
    if (saveGames_) {
        std::filesystem::create_directories(outputDir_);
    }
    
    // Clear previous game records
    {
        std::lock_guard<std::mutex> lock(gameRecordsMutex_);
        gameRecords_.clear();
    }
    
    // Calculate threads per game
    int threadsPerGame = std::max(1, numThreads_ / std::min(numGames_, numThreads_));
    
    // Set up thread pool
    std::vector<std::future<GameRecord>> futures;
    
    // Launch games
    for (int i = 0; i < numGames_; ++i) {
        if (abort_) {
            break;
        }
        
        // Launch game thread
        futures.push_back(std::async(
            std::launch::async,
            &SelfPlayManager::playSingleGame,
            this,
            i,
            gameType,
            boardSize,
            useVariantRules
        ));
    }
    
    // Wait for all games to complete
    for (auto& future : futures) {
        if (abort_) {
            break;
        }
        
        try {
            GameRecord record = future.get();
            
            // Add to collection
            {
                std::lock_guard<std::mutex> lock(gameRecordsMutex_);
                gameRecords_.push_back(record);
            }
            
        } catch (const std::exception& e) {
            // Log error
            std::cerr << "Error in self-play game: " << e.what() << std::endl;
        }
    }
    
    // Set running flag to false
    running_ = false;
    
    // Return collected game records
    std::lock_guard<std::mutex> lock(gameRecordsMutex_);
    return gameRecords_;
}

void SelfPlayManager::setExplorationParams(
    float dirichletAlpha,
    float dirichletEpsilon,
    float initialTemperature,
    int temperatureDropMove,
    float finalTemperature) {
    
    dirichletAlpha_ = dirichletAlpha;
    dirichletEpsilon_ = dirichletEpsilon;
    initialTemperature_ = initialTemperature;
    temperatureDropMove_ = temperatureDropMove;
    finalTemperature_ = finalTemperature;
}

void SelfPlayManager::setProgressCallback(std::function<void(int, int, int, int)> callback) {
    progressCallback_ = callback;
}

void SelfPlayManager::setSaveGames(bool saveGames, const std::string& outputDir) {
    saveGames_ = saveGames;
    outputDir_ = outputDir;
}

void SelfPlayManager::setAbort(bool abort) {
    abort_ = abort;
}

bool SelfPlayManager::isRunning() const {
    return running_;
}

GameRecord SelfPlayManager::playSingleGame(
    int gameId,
    core::GameType gameType, 
    int boardSize,
    bool useVariantRules) {
    
    // Create game state
    auto state = core::createGameState(gameType, boardSize, useVariantRules);
    
    // Create game record
    GameRecord record(gameType, state->getBoardSize(), useVariantRules);
    
    // Create transposition table
    mcts::TranspositionTable tt(1048576);  // 1M entries
    
    // Create MCTS instance
    mcts::ParallelMCTS mcts(*state, neuralNetwork_, &tt, numThreads_, numSimulations_);
    
    // Set MCTS parameters
    mcts.setCPuct(1.5f);
    mcts.setFpuReduction(0.1f);
    mcts.setSelectionStrategy(mcts::MCTSNodeSelection::PUCT);
    
    // Add Dirichlet noise for exploration
    mcts.addDirichletNoise(dirichletAlpha_, dirichletEpsilon_);
    
    // Play until game is over
    int moveNum = 0;
    while (!state->isTerminal() && !abort_) {
        // Report progress
        if (progressCallback_) {
            progressCallback_(gameId, moveNum, numGames_, 0);
        }
        
        // Search for the best move
        auto start = std::chrono::high_resolution_clock::now();
        mcts.search();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate thinking time
        int64_t thinkingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Get the action probabilities
        float temperature = getTemperature(moveNum);
        std::vector<float> actionProbs = mcts.getActionProbabilities(temperature);
        
        // Select action
        int action = mcts.selectAction(true, temperature);
        
        // Get value estimate
        float value = mcts.getRootValue();
        
        // Record the move
        record.addMove(action, actionProbs, value, thinkingTimeMs);
        
        // Make the move
        state->makeMove(action);
        
        // Update MCTS tree
        mcts.updateWithMove(action);
        
        // Add Dirichlet noise after every move
        if (moveNum % 2 == 0) {  // Only add noise for first player's moves
            mcts.addDirichletNoise(dirichletAlpha_, dirichletEpsilon_);
        }
        
        // Increment move counter
        moveNum++;
    }
    
    // Set the game result
    record.setResult(state->getGameResult());
    
    // Save game record if requested
    if (saveGames_) {
        std::ostringstream filename;
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time_t);
        
        filename << outputDir_ << "/";
        filename << std::setfill('0') << std::setw(3) << gameId << "_";
        filename << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".json";
        
        record.saveToFile(filename.str());
    }
    
    return record;
}

float SelfPlayManager::getTemperature(int moveNum) const {
    if (moveNum >= temperatureDropMove_) {
        return finalTemperature_;
    }
    return initialTemperature_;
}

} // namespace selfplay
} // namespace alphazero