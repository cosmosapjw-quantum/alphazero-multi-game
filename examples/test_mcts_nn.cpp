#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <iomanip>

#include "alphazero/core/igamestate.h"
#include "alphazero/games/gomoku/gomoku_state.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/mcts/transposition_table.h"

using namespace alphazero;

// Format time duration for display
std::string formatDuration(std::chrono::milliseconds ms) {
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(ms);
    ms -= std::chrono::duration_cast<std::chrono::milliseconds>(secs);
    
    std::stringstream ss;
    ss << secs.count() << "." << std::setfill('0') << std::setw(3) << ms.count() << "s";
    return ss.str();
}

int main(int argc, char** argv) {
    std::cout << "AlphaZero MCTS with Neural Network Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Parse command line arguments
    std::string modelPath = "";
    bool useGpu = false;
    int numSimulations = 1600;
    int numThreads = 4;
    int boardSize = 15;
    bool useRenju = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--use-gpu") {
            useGpu = true;
        } else if (arg == "--simulations" && i + 1 < argc) {
            numSimulations = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else if (arg == "--board-size" && i + 1 < argc) {
            boardSize = std::stoi(argv[++i]);
        } else if (arg == "--use-renju") {
            useRenju = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: test_mcts_nn [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --model PATH       Path to neural network model" << std::endl;
            std::cout << "  --use-gpu          Use GPU for neural network inference" << std::endl;
            std::cout << "  --simulations N    Number of MCTS simulations (default: 1600)" << std::endl;
            std::cout << "  --threads N        Number of threads (default: 4)" << std::endl;
            std::cout << "  --board-size N     Board size (default: 15)" << std::endl;
            std::cout << "  --use-renju        Use Renju rules (default: false)" << std::endl;
            std::cout << "  --help, -h         Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create Gomoku state
    auto state = std::make_unique<gomoku::GomokuState>(boardSize, useRenju);
    
    // Create neural network
    std::cout << "Creating neural network..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto nn = nn::NeuralNetwork::create(modelPath, core::GameType::GOMOKU, boardSize, useGpu);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "Neural network created in " << formatDuration(duration) << std::endl;
    std::cout << "  Model: " << (modelPath.empty() ? "Random policy (no model loaded)" : modelPath) << std::endl;
    std::cout << "  Device: " << nn->getDeviceInfo() << std::endl;
    std::cout << "  Model info: " << nn->getModelInfo() << std::endl;
    
    // Create transposition table
    std::cout << "Creating transposition table..." << std::endl;
    mcts::TranspositionTable tt(1048576, 1024); // 1M entries, 1024 shards
    
    // Create MCTS
    std::cout << "Creating MCTS with " << numThreads << " threads and "
              << numSimulations << " simulations..." << std::endl;
    mcts::ParallelMCTS mcts(*state, nn.get(), &tt, numThreads, numSimulations);
    
    // Add Dirichlet noise for exploration
    mcts.addDirichletNoise(0.03f, 0.25f);
    
    // Set progress callback
    mcts.setProgressCallback([](int current, int total) {
        static int lastPercent = -1;
        int percent = (100 * current) / total;
        
        if (percent != lastPercent && percent % 10 == 0) {
            std::cout << "\rProgress: " << percent << "% (" 
                     << current << "/" << total << " simulations)" << std::flush;
            lastPercent = percent;
        }
    });
    
    // Display initial board
    std::cout << "\nInitial board:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Game loop
    int moveCount = 0;
    while (!state->isTerminal()) {
        std::cout << "\nMove " << moveCount + 1 << " (Player " 
                 << state->getCurrentPlayer() << "):" << std::endl;
        
        // Run MCTS search
        startTime = std::chrono::high_resolution_clock::now();
        mcts.search();
        endTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "\nSearch completed in " << formatDuration(duration) << std::endl;
        
        // Print search statistics
        std::cout << "Search stats:" << std::endl;
        std::cout << mcts.getSearchInfo() << std::endl;
        
        // Select the best move
        float temperature = moveCount < 30 ? 1.0f : 0.1f;
        int action = mcts.selectAction(true, temperature);
        
        std::cout << "Selected move: " << state->actionToString(action) << std::endl;
        
        // Make the move
        state->makeMove(action);
        
        // Update MCTS tree
        mcts.updateWithMove(action);
        
        // Display board
        std::cout << state->toString() << std::endl;
        
        // Add Dirichlet noise again for the new root
        mcts.addDirichletNoise(0.03f, 0.25f);
        
        // Increment move counter
        moveCount++;
        
        // Print transposition table stats
        if (moveCount % 5 == 0) {
            std::cout << "Transposition table stats:" << std::endl;
            std::cout << "  Size: " << tt.getSize() << std::endl;
            std::cout << "  Entries: " << tt.getEntryCount() << std::endl;
            std::cout << "  Lookups: " << tt.getLookups() << std::endl;
            std::cout << "  Hits: " << tt.getHits() << std::endl;
            std::cout << "  Hit rate: " << (tt.getHitRate() * 100.0) << "%" << std::endl;
            std::cout << "  Memory: " << (tt.getMemoryUsageBytes() / 1024.0 / 1024.0) << " MB" << std::endl;
        }
    }
    
    // Print game result
    std::cout << "\nGame over after " << moveCount << " moves." << std::endl;
    
    switch (state->getGameResult()) {
        case core::GameResult::WIN_PLAYER1:
            std::cout << "Player 1 (Black) wins!" << std::endl;
            break;
        case core::GameResult::WIN_PLAYER2:
            std::cout << "Player 2 (White) wins!" << std::endl;
            break;
        case core::GameResult::DRAW:
            std::cout << "Game ended in a draw." << std::endl;
            break;
        default:
            std::cout << "Unknown result." << std::endl;
            break;
    }
    
    return 0;
}