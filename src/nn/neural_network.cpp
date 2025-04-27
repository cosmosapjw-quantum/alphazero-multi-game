// neural_network.cpp
#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/random_policy_network.h"

namespace alphazero {
namespace nn {

std::unique_ptr<NeuralNetwork> NeuralNetwork::create(
    const std::string& modelPath,
    core::GameType gameType,
    int boardSize,
    bool useGpu
) {
    // TODO: Implement actual neural network loading
    
    // For now, return a dummy random policy network for testing
    return std::make_unique<RandomPolicyNetwork>(gameType, boardSize);
}

} // namespace nn
} // namespace alphazero