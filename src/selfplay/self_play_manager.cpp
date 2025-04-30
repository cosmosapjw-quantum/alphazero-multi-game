// src/selfplay/self_play_manager.cpp
#include "alphazero/types.h"
#include "alphazero/selfplay/self_play_manager.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include "alphazero/mcts/transposition_table.h"
#if defined(PYBIND11_MODULE)
#include <pybind11/pybind11.h>
#endif

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
      running_(false),
      batchSize_(64),
      batchTimeoutMs_(10) {
}

SelfPlayManager::~SelfPlayManager() {
    // Signal abort and wait for running threads to finish
    setAbort(true);
    while (isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::vector<GameRecord> SelfPlayManager::generateGames(
    core::GameType gameType,
    int boardSize,
    bool useVariantRules) {

    running_ = true;
    abort_ = false;
    
    // Reset counters
    completedGamesCount_.store(0, std::memory_order_relaxed);
    totalMovesCount_.store(0, std::memory_order_relaxed);

    if (saveGames_) {
        std::filesystem::create_directories(outputDir_);
    }

    {
        std::lock_guard<std::mutex> lock(gameRecordsMutex_);
        gameRecords_.clear();
    }

    // Create thread pool if not already created
    if (!threadPool_) {
        threadPool_ = std::make_unique<mcts::ThreadPool>(numThreads_);
    }

    // Launch tasks in the thread pool
    std::vector<std::future<GameRecord>> futures;
    futures.reserve(numGames_);

    for (int i = 0; i < numGames_; ++i) {
        if (abort_) break;
        
        // Enqueue the task to the thread pool
        futures.push_back(threadPool_->enqueue(
            &SelfPlayManager::playSingleGame, 
            this, 
            i, 
            gameType, 
            boardSize, 
            useVariantRules
        ));
    }

    // Collect results from futures
    for (auto& future : futures) {
        if (abort_) break;
        
        try {
            GameRecord rec = future.get();
            
            // Store the game record
            {
                std::lock_guard<std::mutex> lock(gameRecordsMutex_);
                gameRecords_.push_back(std::move(rec));
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in self-play game: " << e.what() << std::endl;
        }
    }

    running_ = false;
    
    // Return a copy of the game records
    std::lock_guard<std::mutex> lock(gameRecordsMutex_);
    return gameRecords_;
}

void SelfPlayManager::setExplorationParams(
    float dirichletAlpha,
    float dirichletEpsilon,
    float initialTemperature,
    int temperatureDropMove,
    float finalTemperature) {
    dirichletAlpha_     = dirichletAlpha;
    dirichletEpsilon_   = dirichletEpsilon;
    initialTemperature_ = initialTemperature;
    temperatureDropMove_= temperatureDropMove;
    finalTemperature_   = finalTemperature;
}

void SelfPlayManager::setProgressCallback(std::function<void(int,int,int,int)> callback) {
    progressCallback_ = callback;
}

void SelfPlayManager::setBatchConfig(int batchSize, int batchTimeoutMs) {
    batchSize_       = batchSize;
    batchTimeoutMs_  = batchTimeoutMs;
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

    auto state = core::createGameState(gameType, boardSize, useVariantRules);
    GameRecord record(gameType, state->getBoardSize(), useVariantRules);
    mcts::TranspositionTable tt(1048576);

    // Configure MCTS with our stored config
    mcts::MCTSConfig mctsConfig = mctsConfig_;
    
    // Apply some necessary overrides for this specific game
    mctsConfig.numThreads = std::min(numThreads_, mctsConfig.numThreads);
    mctsConfig.numSimulations = numSimulations_;
    
    // Ensure batch configuration is enabled
    mctsConfig.useBatchInference = true;
    mctsConfig.useBatchedMCTS = true;
    mctsConfig.batchSize = batchSize_;
    mctsConfig.batchTimeoutMs = batchTimeoutMs_;
    mctsConfig.searchMode = mcts::MCTSSearchMode::BATCHED;

    mcts::ParallelMCTS mcts(*state, mctsConfig, neuralNetwork_, &tt);
    
    // These settings will be used only if not already set in mctsConfig
    if (mctsConfig.cPuct <= 0.0f) mcts.setCPuct(1.5f);
    if (mctsConfig.fpuReduction < 0.0f) mcts.setFpuReduction(0.1f);
    if (mctsConfig.selectionStrategy == mcts::MCTSNodeSelection::UCB) 
        mcts.setSelectionStrategy(mcts::MCTSNodeSelection::PUCT);
    
    // Always add Dirichlet noise for root node to ensure exploration
    mcts.addDirichletNoise(dirichletAlpha_, dirichletEpsilon_);

    int moveNum = 0;
    while (!state->isTerminal() && !abort_) {
        if (progressCallback_) {
#if defined(PYBIND11_MODULE)
            // Acquire GIL before calling Python callback
            pybind11::gil_scoped_acquire acquire;
#endif
            // gameId, moveNum, totalGames, totalMoves
            progressCallback_(gameId, moveNum, numGames_, totalMovesCount_.load());
        }

        auto start = std::chrono::high_resolution_clock::now();
        mcts.search();
        auto end   = std::chrono::high_resolution_clock::now();

        int64_t thinkingTimeMs = std::chrono::duration_cast<
            std::chrono::milliseconds>(end - start).count();

        float temperature = getTemperature(moveNum);
        auto actionProbs  = mcts.getActionProbabilities(temperature);
        int action        = mcts.selectAction(true, temperature);
        float value       = mcts.getRootValue();

        record.addMove(action, actionProbs, value, thinkingTimeMs);
        state->makeMove(action);
        mcts.updateWithMove(action);

        if (moveNum % 2 == 0) {
            mcts.addDirichletNoise(dirichletAlpha_, dirichletEpsilon_);
        }
        ++moveNum;
        
        // Update total moves counter
        totalMovesCount_.fetch_add(1, std::memory_order_relaxed);
    }

    record.setResult(state->getGameResult());
    if (saveGames_) {
        std::ostringstream filename;
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&t);
        filename << outputDir_ << "/"
                 << std::setfill('0') << std::setw(3) << gameId << "_"
                 << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".json";
        record.saveToFile(filename.str());
    }
    
    // Update completed games counter
    completedGamesCount_.fetch_add(1, std::memory_order_relaxed);
    
    return record;
}

float SelfPlayManager::getTemperature(int moveNum) const {
    return (moveNum >= temperatureDropMove_)
        ? finalTemperature_
        : initialTemperature_;
}

void SelfPlayManager::setMctsConfig(const mcts::MCTSConfig& config) {
    mctsConfig_ = config;
    
    // Update batch configuration to match MCTS config
    if (config.useBatchedMCTS) {
        batchSize_ = config.batchSize;
        batchTimeoutMs_ = config.batchTimeoutMs;
    }
}

} // namespace selfplay
} // namespace alphazero