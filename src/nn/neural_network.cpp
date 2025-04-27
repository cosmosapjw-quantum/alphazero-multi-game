// neural_network.cpp
#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/random_policy_network.h"
#include "alphazero/nn/torch_neural_network.h"

namespace alphazero {
namespace nn {

std::unique_ptr<NeuralNetwork> NeuralNetwork::create(
    const std::string& modelPath,
    core::GameType gameType,
    int boardSize,
    bool useGpu
) {
    try {
        // If model path provided, try to load a TorchNeuralNetwork
        if (!modelPath.empty()) {
            return std::make_unique<TorchNeuralNetwork>(modelPath, gameType, boardSize, useGpu);
        }
        
        // If no model path, fall back to random policy network
        return std::make_unique<RandomPolicyNetwork>(gameType, boardSize);
    } catch (const std::exception& e) {
        // Log error and fall back to RandomPolicyNetwork
        std::cerr << "Failed to create TorchNeuralNetwork: " << e.what() << std::endl;
        std::cerr << "Falling back to RandomPolicyNetwork" << std::endl;
        return std::make_unique<RandomPolicyNetwork>(gameType, boardSize);
    }
}

} // namespace nn
} // namespace alphazero