#include <iostream>
#include <memory>
#include "alphazero/core/igamestate.h"
#include "alphazero/games/go/go_state.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"

using namespace alphazero;

int main() {
    std::cout << "Go Example" << std::endl;
    std::cout << "==========" << std::endl;
    
    // Create a Go game state (9x9 board)
    auto state = core::createGameState(core::GameType::GO, 9);
    
    // Display the initial board
    std::cout << "Initial board:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Create a neural network
    auto nn = nn::NeuralNetwork::create("", core::GameType::GO);
    
    // Create MCTS
    mcts::ParallelMCTS mcts(*state, nn.get(), nullptr, 2, 100);
    
    // Make a move at position 3,3 (star point)
    int pos = 3 * 9 + 3;  // Convert to linear position
    std::cout << "Making a move at " << state->actionToString(pos) << std::endl;
    state->makeMove(pos);
    
    // Display board after move
    std::cout << "Board after move:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Let MCTS search for best response
    std::cout << "MCTS searching for best response..." << std::endl;
    mcts.updateWithMove(pos);
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
