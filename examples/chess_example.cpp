#include <iostream>
#include <memory>
#include <optional>
#include "alphazero/core/igamestate.h"
#include "alphazero/games/chess/chess_state.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"

using namespace alphazero;

int main() {
    std::cout << "Chess Example" << std::endl;
    std::cout << "=============" << std::endl;
    
    // Create a Chess game state
    auto state = core::createGameState(core::GameType::CHESS);
    
    // Display the initial board
    std::cout << "Initial board:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Create a neural network
    auto nn = nn::NeuralNetwork::create("", core::GameType::CHESS);
    
    // Create MCTS
    mcts::ParallelMCTS mcts(*state, nn.get(), nullptr, 2, 100);
    
    // Make a standard opening move (e4)
    std::optional<int> e4Opt = state->stringToAction("e2e4");
    if (!e4Opt) {
        std::cerr << "Invalid move: e2e4" << std::endl;
        return 1;
    }
    int e4 = *e4Opt;
    
    std::cout << "Making a move: " << state->actionToString(e4) << std::endl;
    state->makeMove(e4);
    
    // Display board after move
    std::cout << "Board after move:" << std::endl;
    std::cout << state->toString() << std::endl;
    
    // Let MCTS search for best response
    std::cout << "MCTS searching for best response..." << std::endl;
    mcts.updateWithMove(e4);
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
