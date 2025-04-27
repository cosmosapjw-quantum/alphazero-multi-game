#include <iostream>
#include <memory>
#include "alphazero/core/igamestate.h"
#include "alphazero/games/gomoku/gomoku_state.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"

using namespace alphazero;

int main() {
    std::cout << "Gomoku Example" << std::endl;
    std::cout << "=============" << std::endl;
    
    // Create a Gomoku game state
    auto state = core::createGameState(core::GameType::GOMOKU, 15);
    
    // Display the initial board
    std::cout << "Initial board:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Create a neural network
    auto nn = nn::NeuralNetwork::create("", core::GameType::GOMOKU);
    
    // Create MCTS
    mcts::ParallelMCTS mcts(*state, nn.get(), nullptr, 2, 100);
    
    // Make a move at center
    int center = 7 * 15 + 7;  // Center position for 15x15 board
    std::cout << "Making a move at " << state->actionToString(center) << std::endl;
    state->makeMove(center);
    
    // Display board after move
    std::cout << "Board after move:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Let MCTS search for best response
    std::cout << "MCTS searching for best response..." << std::endl;
    mcts.updateWithMove(center);
    mcts.search();
    
    // Get and make the best move
    int bestMove = mcts.selectAction();
    std::cout << "MCTS selected move: " << state->actionToString(bestMove) << std::endl;
    state->makeMove(bestMove);
    
    // Display final board
    std::cout << "Board after MCTS move:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    return 0;
}