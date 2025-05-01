#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "alphazero/types.h"
#include "alphazero/core/game_factory.h"
#include "alphazero/nn/torch_neural_network.h"
#include "alphazero/nn/random_policy_network.h"
#include "alphazero/selfplay/self_play_manager.h"
#include "alphazero/mcts/transposition_table.h"

// Simple command line parser
class CommandLineParser {
public:
    CommandLineParser(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg[0] == '-') {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    args_[arg] = argv[i + 1];
                    ++i;
                } else {
                    args_[arg] = "true";
                }
            }
        }
    }

    bool has(const std::string& key) const {
        return args_.find(key) != args_.end();
    }

    std::string get(const std::string& key, const std::string& defaultValue = "") const {
        auto it = args_.find(key);
        if (it != args_.end()) {
            return it->second;
        }
        return defaultValue;
    }

    int getInt(const std::string& key, int defaultValue = 0) const {
        auto it = args_.find(key);
        if (it != args_.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    float getFloat(const std::string& key, float defaultValue = 0.0f) const {
        auto it = args_.find(key);
        if (it != args_.end()) {
            try {
                return std::stof(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    bool getBool(const std::string& key, bool defaultValue = false) const {
        auto it = args_.find(key);
        if (it != args_.end()) {
            std::string value = it->second;
            if (value == "true" || value == "yes" || value == "1") {
                return true;
            } else if (value == "false" || value == "no" || value == "0") {
                return false;
            }
            return defaultValue;
        }
        return defaultValue;
    }

private:
    std::unordered_map<std::string, std::string> args_;
};

void printUsage() {
    std::cout << "AlphaZero Multi-Game Self-Play Generator\n"
              << "Usage: self_play [options]\n\n"
              << "Options:\n"
              << "  --model PATH          Path to model file (LibTorch format)\n"
              << "  --game TYPE           Game type: gomoku, chess, go (default: gomoku)\n"
              << "  --size SIZE           Board size (default: depends on game)\n"
              << "  --num-games NUM       Number of games to generate (default: 100)\n"
              << "  --simulations SIMS    Number of MCTS simulations per move (default: 800)\n"
              << "  --threads THREADS     Number of threads (default: auto-detect)\n"
              << "  --output-dir DIR      Output directory (default: data/games)\n"
              << "  --temperature TEMP    Initial temperature (default: 1.0)\n"
              << "  --temp-drop MOVE      Move to drop temperature (default: 30)\n"
              << "  --final-temp TEMP     Final temperature (default: 0.0)\n"
              << "  --dirichlet-alpha A   Dirichlet noise alpha (default: 0.03)\n"
              << "  --dirichlet-epsilon E Dirichlet noise weight (default: 0.25)\n"
              << "  --variant             Use variant rules (Renju, Chess960, Chinese)\n"
              << "  --batch-size SIZE     Batch size for neural network inference (default: 8)\n"
              << "  --batch-timeout MS    Timeout for batch completion in milliseconds (default: 10)\n"
              << "  --no-gpu              Disable GPU acceleration\n"
              << "  --no-batched-search   Disable batched MCTS search\n"
              << "  --fp16                Use FP16 precision (faster but less accurate)\n"
              << "  --c-puct VALUE        Exploration constant (default: 1.5)\n"
              << "  --fpu-reduction VALUE First play urgency reduction (default: 0.1)\n"
              << "  --virtual-loss VALUE  Virtual loss amount (default: 3)\n"
              << "  --use-tt              Use transposition table in MCTS (default: true)\n"
              << "  --progressive-widening Use progressive widening in MCTS\n"
              << "  --help                Display this help message\n";
}

alphazero::core::GameType parseGameType(const std::string& gameTypeStr) {
    if (gameTypeStr == "chess") {
        return alphazero::core::GameType::CHESS;
    } else if (gameTypeStr == "go") {
        return alphazero::core::GameType::GO;
    } else {
        return alphazero::core::GameType::GOMOKU;
    }
}

// Simple progress callback for console display
void progressCallback(int gameId, int moveNum, int totalGames, int totalMoves) {
    static auto lastUpdate = std::chrono::steady_clock::now();
    static int lastGameId = 0;
    static int lastTotalMoves = 0;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count();
    
    // Update at most once per second
    if (elapsed >= 1 || gameId > lastGameId) {
        int gamesDelta = gameId - lastGameId;
        int movesDelta = totalMoves - lastTotalMoves;
        
        float gamesPerSec = elapsed > 0 ? static_cast<float>(gamesDelta) / elapsed : 0.0f;
        float movesPerSec = elapsed > 0 ? static_cast<float>(movesDelta) / elapsed : 0.0f;
        
        std::cout << "Progress: " << gameId << "/" << totalGames 
                  << " games | " << totalMoves << " total moves | "
                  << std::fixed << std::setprecision(2) << gamesPerSec << " games/sec | "
                  << std::fixed << std::setprecision(1) << movesPerSec << " moves/sec"
                  << "        \r" << std::flush;
        
        lastUpdate = now;
        lastGameId = gameId;
        lastTotalMoves = totalMoves;
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CommandLineParser args(argc, argv);
    
    // Show help if requested
    if (args.has("--help")) {
        printUsage();
        return 0;
    }
    
    // Parse arguments
    std::string modelPath = args.get("--model", "");
    std::string gameTypeStr = args.get("--game", "gomoku");
    int boardSize = args.getInt("--size", 0);
    int numGames = args.getInt("--num-games", 100);
    int numSimulations = args.getInt("--simulations", 800);
    int numThreads = args.getInt("--threads", 0);
    std::string outputDir = args.get("--output-dir", "data/games");
    float temperature = args.getFloat("--temperature", 1.0f);
    int tempDropMove = args.getInt("--temp-drop", 30);
    float finalTemperature = args.getFloat("--final-temp", 0.0f);
    float dirichletAlpha = args.getFloat("--dirichlet-alpha", 0.03f);
    float dirichletEpsilon = args.getFloat("--dirichlet-epsilon", 0.25f);
    bool useVariantRules = args.has("--variant");
    int batchSize = args.getInt("--batch-size", 8);
    int batchTimeoutMs = args.getInt("--batch-timeout", 10);
    bool useGpu = !args.has("--no-gpu");
    bool useBatchedSearch = !args.has("--no-batched-search");
    bool useFp16 = args.has("--fp16");
    float cPuct = args.getFloat("--c-puct", 1.5f);
    float fpuReduction = args.getFloat("--fpu-reduction", 0.1f);
    int virtualLoss = args.getInt("--virtual-loss", 3);
    bool useTranspositionTable = !args.has("--no-tt");
    bool useProgressiveWidening = args.has("--progressive-widening");
    
    // Parse game type
    alphazero::core::GameType gameType = parseGameType(gameTypeStr);
    
    // Set default board size if not specified
    if (boardSize <= 0) {
        switch (gameType) {
            case alphazero::core::GameType::GOMOKU:
                boardSize = 15;
                break;
            case alphazero::core::GameType::CHESS:
                boardSize = 8;
                break;
            case alphazero::core::GameType::GO:
                boardSize = 19;
                break;
            default:
                boardSize = 15;
                break;
        }
    }
    
    // Auto-detect thread count if not specified
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads <= 0) {
            numThreads = 4;  // Fallback if detection fails
        }
        // Use 75% of available cores
        numThreads = std::max(1, static_cast<int>(numThreads * 0.75));
    }
    
    // Create output directory
    std::filesystem::create_directories(outputDir);
    
    // Initialize neural network
    std::unique_ptr<alphazero::nn::NeuralNetwork> neuralNetwork = nullptr;
    
    if (!modelPath.empty()) {
        try {
            // Configure TorchNeuralNetwork
            alphazero::nn::TorchNeuralNetworkConfig nnConfig;
            nnConfig.useGpu = useGpu;
            nnConfig.useFp16 = useFp16;
            nnConfig.batchSize = batchSize;
            nnConfig.batchTimeoutMs = batchTimeoutMs;
            nnConfig.useAsyncExecution = true;
            nnConfig.useTensorCaching = true;
            
            std::cout << "Loading model from " << modelPath << std::endl;
            neuralNetwork = std::make_unique<alphazero::nn::TorchNeuralNetwork>(
                modelPath, gameType, boardSize, useGpu, nnConfig
            );
            
            std::cout << "Model loaded: " << neuralNetwork->getDeviceInfo() << std::endl;
            std::cout << "Batch size: " << neuralNetwork->getBatchSize() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            std::cerr << "Using random policy network instead." << std::endl;
            neuralNetwork = std::make_unique<alphazero::nn::RandomPolicyNetwork>(
                gameType, boardSize
            );
        }
    } else {
        std::cout << "No model specified. Using random policy network." << std::endl;
        neuralNetwork = std::make_unique<alphazero::nn::RandomPolicyNetwork>(
            gameType, boardSize
        );
    }
    
    // Create transposition table
    std::unique_ptr<alphazero::mcts::TranspositionTable> tt = nullptr;
    if (useTranspositionTable) {
        tt = std::make_unique<alphazero::mcts::TranspositionTable>(1048576);
    }
    
    // Create self-play manager
    alphazero::selfplay::SelfPlayManager selfPlay(
        neuralNetwork.get(),
        numGames,
        numSimulations,
        numThreads
    );
    
    // Set exploration parameters
    selfPlay.setExplorationParams(
        dirichletAlpha,
        dirichletEpsilon,
        temperature,
        tempDropMove,
        finalTemperature
    );
    
    // Configure batch settings
    selfPlay.setBatchConfig(batchSize, batchTimeoutMs);
    
    // Set save options
    selfPlay.setSaveGames(true, outputDir);
    
    // Set progress callback
    selfPlay.setProgressCallback(progressCallback);
    
    // Configure MCTS
    alphazero::mcts::MCTSConfig mctsConfig;
    mctsConfig.numThreads = numThreads;
    mctsConfig.numSimulations = numSimulations;
    mctsConfig.cPuct = cPuct;
    mctsConfig.fpuReduction = fpuReduction;
    mctsConfig.virtualLoss = virtualLoss;
    mctsConfig.useDirichletNoise = true;
    mctsConfig.dirichletAlpha = dirichletAlpha;
    mctsConfig.dirichletEpsilon = dirichletEpsilon;
    mctsConfig.useBatchInference = true;
    mctsConfig.useBatchedMCTS = useBatchedSearch;
    mctsConfig.batchSize = batchSize;
    mctsConfig.batchTimeoutMs = batchTimeoutMs;
    mctsConfig.searchMode = useBatchedSearch ? 
        alphazero::mcts::MCTSSearchMode::BATCHED : 
        alphazero::mcts::MCTSSearchMode::PARALLEL;
    mctsConfig.useProgressiveWidening = useProgressiveWidening;
    mctsConfig.useFmapCache = useTranspositionTable;
    
    selfPlay.setMctsConfig(mctsConfig);
    
    // Print configuration
    std::cout << "Starting self-play generation..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Game: " << gameTypeStr << std::endl;
    std::cout << "Board size: " << boardSize << "x" << boardSize << std::endl;
    std::cout << "Number of games: " << numGames << std::endl;
    std::cout << "Simulations per move: " << numSimulations << std::endl;
    std::cout << "Number of threads: " << numThreads << std::endl;
    std::cout << "Using GPU: " << (useGpu ? "yes" : "no") << std::endl;
    std::cout << "Using FP16: " << (useFp16 ? "yes" : "no") << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Batch timeout: " << batchTimeoutMs << " ms" << std::endl;
    std::cout << "Batched MCTS: " << (useBatchedSearch ? "enabled" : "disabled") << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    std::cout << "Model path: " << (modelPath.empty() ? "Random policy" : modelPath) << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Generate games
    auto gameRecords = selfPlay.generateGames(gameType, boardSize, useVariantRules);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    // Print results
    std::cout << "\nSelf-play completed!" << std::endl;
    std::cout << "Generated " << gameRecords.size() << " games in " << duration << " seconds" << std::endl;
    
    int totalMoves = selfPlay.getTotalMovesCount();
    float movesPerGame = gameRecords.empty() ? 0.0f : static_cast<float>(totalMoves) / gameRecords.size();
    float movesPerSecond = duration > 0 ? static_cast<float>(totalMoves) / duration : 0.0f;
    
    std::cout << "Total moves: " << totalMoves << std::endl;
    std::cout << "Average moves per game: " << std::fixed << std::setprecision(1) << movesPerGame << std::endl;
    std::cout << "Average moves per second: " << std::fixed << std::setprecision(1) << movesPerSecond << std::endl;
    
    // Save metadata to JSON file
    std::string metadataPath = outputDir + "/metadata_" + 
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".json";
    
    std::ofstream metadataFile(metadataPath);
    if (metadataFile.is_open()) {
        metadataFile << "{\n";
        metadataFile << "  \"game\": \"" << gameTypeStr << "\",\n";
        metadataFile << "  \"board_size\": " << boardSize << ",\n";
        metadataFile << "  \"num_games_requested\": " << numGames << ",\n";
        metadataFile << "  \"num_games_completed\": " << gameRecords.size() << ",\n";
        metadataFile << "  \"simulations\": " << numSimulations << ",\n";
        metadataFile << "  \"threads\": " << numThreads << ",\n";
        metadataFile << "  \"temperature\": " << temperature << ",\n";
        metadataFile << "  \"temp_drop\": " << tempDropMove << ",\n";
        metadataFile << "  \"final_temp\": " << finalTemperature << ",\n";
        metadataFile << "  \"dirichlet_alpha\": " << dirichletAlpha << ",\n";
        metadataFile << "  \"dirichlet_epsilon\": " << dirichletEpsilon << ",\n";
        metadataFile << "  \"variant\": " << (useVariantRules ? "true" : "false") << ",\n";
        metadataFile << "  \"model_path\": \"" << modelPath << "\",\n";
        metadataFile << "  \"total_moves\": " << totalMoves << ",\n";
        metadataFile << "  \"avg_moves_per_game\": " << movesPerGame << ",\n";
        metadataFile << "  \"total_time_seconds\": " << duration << ",\n";
        metadataFile << "  \"avg_moves_per_second\": " << movesPerSecond << ",\n";
        metadataFile << "  \"use_gpu\": " << (useGpu ? "true" : "false") << ",\n";
        metadataFile << "  \"batch_size\": " << batchSize << ",\n";
        metadataFile << "  \"batch_timeout\": " << batchTimeoutMs << ",\n";
        metadataFile << "  \"fp16_used\": " << (useFp16 ? "true" : "false") << ",\n";
        metadataFile << "  \"c_puct\": " << cPuct << ",\n";
        metadataFile << "  \"fpu_reduction\": " << fpuReduction << ",\n";
        metadataFile << "  \"virtual_loss\": " << virtualLoss << ",\n";
        metadataFile << "  \"use_transposition_table\": " << (useTranspositionTable ? "true" : "false") << ",\n";
        metadataFile << "  \"progressive_widening\": " << (useProgressiveWidening ? "true" : "false") << "\n";
        metadataFile << "}\n";
        metadataFile.close();
        
        std::cout << "Metadata saved to " << metadataPath << std::endl;
    }
    
    return 0;
}