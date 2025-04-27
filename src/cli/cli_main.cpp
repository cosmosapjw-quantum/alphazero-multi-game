// cli_main.cpp
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

#include "alphazero/core/igamestate.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/mcts/transposition_table.h"
#include "alphazero/nn/neural_network.h"

using namespace alphazero;

// Forward declarations
void playSingleGame(core::GameType gameType, int boardSize, int simulations, int threads);
void runBenchmark(core::GameType gameType, int boardSize, int simulations, int threads, int games);
void showHelp();

int main(int argc, char* argv[]) {
    std::cout << "AlphaZero Multi-Game AI Engine CLI" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Default parameters
    core::GameType gameType = core::GameType::GOMOKU;
    int boardSize = 0;  // 0 means use default for the game
    int simulations = 800;
    int threads = 4;
    bool benchmark = false;
    int benchmarkGames = 10;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            showHelp();
            return 0;
        } else if (arg == "--game" || arg == "-g") {
            if (i + 1 < argc) {
                std::string gameArg = argv[++i];
                if (gameArg == "gomoku") {
                    gameType = core::GameType::GOMOKU;
                } else if (gameArg == "chess") {
                    gameType = core::GameType::CHESS;
                } else if (gameArg == "go") {
                    gameType = core::GameType::GO;
                } else {
                    std::cerr << "Unknown game type: " << gameArg << std::endl;
                    return 1;
                }
            }
        } else if (arg == "--size" || arg == "-s") {
            if (i + 1 < argc) {
                boardSize = std::stoi(argv[++i]);
            }
        } else if (arg == "--simulations" || arg == "-n") {
            if (i + 1 < argc) {
                simulations = std::stoi(argv[++i]);
            }
        } else if (arg == "--threads" || arg == "-t") {
            if (i + 1 < argc) {
                threads = std::stoi(argv[++i]);
            }
        } else if (arg == "--benchmark" || arg == "-b") {
            benchmark = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                benchmarkGames = std::stoi(argv[++i]);
            }
        }
    }
    
    // Run selected mode
    if (benchmark) {
        runBenchmark(gameType, boardSize, simulations, threads, benchmarkGames);
    } else {
        playSingleGame(gameType, boardSize, simulations, threads);
    }
    
    return 0;
}

void playSingleGame(core::GameType gameType, int boardSize, int simulations, int threads) {
    // Print game info
    std::cout << "Starting new game:" << std::endl;
    std::cout << "  Game: " << (gameType == core::GameType::GOMOKU ? "Gomoku" : 
                              (gameType == core::GameType::CHESS ? "Chess" : "Go")) << std::endl;
    std::cout << "  Board size: " << (boardSize > 0 ? std::to_string(boardSize) : "default") << std::endl;
    std::cout << "  Simulations: " << simulations << std::endl;
    std::cout << "  Threads: " << threads << std::endl;
    std::cout << std::endl;
    
    // Create game state
    std::unique_ptr<core::IGameState> state = core::createGameState(gameType, boardSize, false);
    
    // Create neural network
    std::unique_ptr<nn::NeuralNetwork> nn = nn::NeuralNetwork::create("", gameType, boardSize, false);
    
    // Create transposition table
    mcts::TranspositionTable tt(1048576, 1024);
    
    // Create MCTS
    mcts::ParallelMCTS mcts(*state, nn.get(), &tt, threads, simulations);
    
    // Set progress callback
    mcts.setProgressCallback([](int current, int total) {
        std::cout << "\rThinking... " << current << "/" << total 
                 << " simulations (" << (100 * current / total) << "%)" << std::flush;
    });
    
    // Game loop
    while (!state->isTerminal()) {
        // Print board
        std::cout << std::endl;
        std::cout << state->toString() << std::endl;
        
        // Get current player
        int currentPlayer = state->getCurrentPlayer();
        
        if (currentPlayer == 1) {
            // Human player (Black)
            std::cout << "Your move (or 'quit' to exit): ";
            std::string input;
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "exit") {
                break;
            }
            
            // Try to parse move
            auto action = state->stringToAction(input);
            if (!action) {
                std::cout << "Invalid move format. Try again." << std::endl;
                continue;
            }
            
            // Check if move is legal
            if (!state->isLegalMove(*action)) {
                std::cout << "Illegal move. Try again." << std::endl;
                continue;
            }
            
            // Make the move
            state->makeMove(*action);
            
            // Update MCTS tree
            mcts.updateWithMove(*action);
            
        } else {
            // AI player (White)
            std::cout << "AI is thinking..." << std::endl;
            
            // Run search
            mcts.search();
            std::cout << std::endl;
            
            // Print search statistics
            mcts.printSearchStats();
            
            // Select move
            int action = mcts.selectAction(false, 0.0f);
            
            std::cout << "AI plays: " << state->actionToString(action) << std::endl;
            
            // Make move
            state->makeMove(action);
            
            // Update MCTS tree
            mcts.updateWithMove(action);
        }
    }
    
    // Print final board
    std::cout << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Print game result
    auto result = state->getGameResult();
    switch (result) {
        case core::GameResult::WIN_PLAYER1:
            std::cout << "Black (Player 1) wins!" << std::endl;
            break;
        case core::GameResult::WIN_PLAYER2:
            std::cout << "White (Player 2/AI) wins!" << std::endl;
            break;
        case core::GameResult::DRAW:
            std::cout << "Game ended in a draw." << std::endl;
            break;
        default:
            std::cout << "Game ended with unknown result." << std::endl;
            break;
    }
}

void runBenchmark(core::GameType gameType, int boardSize, int simulations, int threads, int games) {
    std::cout << "Running benchmark:" << std::endl;
    std::cout << "  Game: " << (gameType == core::GameType::GOMOKU ? "Gomoku" : 
                              (gameType == core::GameType::CHESS ? "Chess" : "Go")) << std::endl;
    std::cout << "  Board size: " << (boardSize > 0 ? std::to_string(boardSize) : "default") << std::endl;
    std::cout << "  Simulations: " << simulations << std::endl;
    std::cout << "  Threads: " << threads << std::endl;
    std::cout << "  Games: " << games << std::endl;
    std::cout << std::endl;
    
    // Create neural network
    std::unique_ptr<nn::NeuralNetwork> nn = nn::NeuralNetwork::create("", gameType, boardSize, false);
    
    // Create transposition table
    mcts::TranspositionTable tt(1048576, 1024);
    
    // Timing variables
    std::vector<double> searchTimes;
    std::vector<double> moveTimes;
    int totalNodes = 0;
    
    // Run games
    for (int game = 0; game < games; ++game) {
        std::cout << "Game " << (game + 1) << "/" << games << std::endl;
        
        // Create new game state
        std::unique_ptr<core::IGameState> state = core::createGameState(gameType, boardSize, false);
        
        // Create MCTS
        mcts::ParallelMCTS mcts(*state, nn.get(), &tt, threads, simulations);
        
        // Track nodes visited
        int gameNodes = 0;
        
        // Play game
        while (!state->isTerminal()) {
            // Measure search time
            auto searchStart = std::chrono::high_resolution_clock::now();
            
            // Run search
            mcts.search();
            
            auto searchEnd = std::chrono::high_resolution_clock::now();
            double searchTime = std::chrono::duration<double, std::milli>(searchEnd - searchStart).count();
            searchTimes.push_back(searchTime);
            
            // Select move
            auto moveStart = std::chrono::high_resolution_clock::now();
            int action = mcts.selectAction(false, 0.0f);
            auto moveEnd = std::chrono::high_resolution_clock::now();
            
            double moveTime = std::chrono::duration<double, std::milli>(moveEnd - moveStart).count();
            moveTimes.push_back(moveTime);
            
            // Make move
            state->makeMove(action);
            
            // Update MCTS tree
            mcts.updateWithMove(action);
            
            // Count nodes
            gameNodes += simulations;
            
            std::cout << "  Move " << state->getMoveHistory().size() 
                     << ": search=" << searchTime << "ms, move=" << moveTime << "ms" << std::endl;
        }
        
        totalNodes += gameNodes;
        
        // Print game result
        auto result = state->getGameResult();
        std::cout << "  Game result: " << static_cast<int>(result) << std::endl;
        std::cout << "  Total moves: " << state->getMoveHistory().size() << std::endl;
        std::cout << std::endl;
    }
    
    // Calculate statistics
    double totalSearchTime = 0.0;
    double minSearchTime = searchTimes.empty() ? 0.0 : searchTimes[0];
    double maxSearchTime = searchTimes.empty() ? 0.0 : searchTimes[0];
    
    for (double time : searchTimes) {
        totalSearchTime += time;
        minSearchTime = std::min(minSearchTime, time);
        maxSearchTime = std::max(maxSearchTime, time);
    }
    
    double avgSearchTime = searchTimes.empty() ? 0.0 : totalSearchTime / searchTimes.size();
    double nodesPerSecond = totalSearchTime > 0.0 ? 
                          (totalNodes * 1000.0) / totalSearchTime : 0.0;
    
    // Print results
    std::cout << "Benchmark results:" << std::endl;
    std::cout << "  Total searches: " << searchTimes.size() << std::endl;
    std::cout << "  Total nodes: " << totalNodes << std::endl;
    std::cout << "  Search time (avg): " << avgSearchTime << " ms" << std::endl;
    std::cout << "  Search time (min): " << minSearchTime << " ms" << std::endl;
    std::cout << "  Search time (max): " << maxSearchTime << " ms" << std::endl;
    std::cout << "  Nodes per second: " << nodesPerSecond << std::endl;
    std::cout << std::endl;
    
    // Transposition table stats
    std::cout << "Transposition table stats:" << std::endl;
    std::cout << "  Size: " << tt.getSize() << std::endl;
    std::cout << "  Entries: " << tt.getEntryCount() << std::endl;
    std::cout << "  Lookups: " << tt.getLookups() << std::endl;
    std::cout << "  Hits: " << tt.getHits() << std::endl;
    std::cout << "  Hit rate: " << (tt.getHitRate() * 100.0) << "%" << std::endl;
    std::cout << "  Memory: " << (tt.getMemoryUsageBytes() / 1024.0 / 1024.0) << " MB" << std::endl;
}

void showHelp() {
    std::cout << "Usage: alphazero_cli [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help                 Show this help message" << std::endl;
    std::cout << "  -g, --game GAME            Set game type (gomoku, chess, go)" << std::endl;
    std::cout << "  -s, --size SIZE            Set board size" << std::endl;
    std::cout << "  -n, --simulations NUM      Set number of MCTS simulations" << std::endl;
    std::cout << "  -t, --threads NUM          Set number of threads" << std::endl;
    std::cout << "  -b, --benchmark [GAMES]    Run benchmark with GAMES games (default: 10)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  alphazero_cli -g gomoku -s 15 -n 1000 -t 4   # Play Gomoku 15x15 with 1000 simulations" << std::endl;
    std::cout << "  alphazero_cli -g go -b 5                     # Benchmark Go with 5 games" << std::endl;
}