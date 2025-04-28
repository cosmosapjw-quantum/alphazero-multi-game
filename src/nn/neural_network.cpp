// neural_network.cpp
#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/random_policy_network.h"
#include <iostream>

// Only include TorchNeuralNetwork if LibTorch is enabled
#ifndef LIBTORCH_OFF
#include "alphazero/nn/torch_neural_network.h"
#endif

namespace alphazero {
namespace nn {

std::unique_ptr<NeuralNetwork> NeuralNetwork::create(
    const std::string& modelPath,
    core::GameType gameType,
    int boardSize,
    bool useGpu
) {
    try {
        // If model path provided and LibTorch is available, try to load a TorchNeuralNetwork
        if (!modelPath.empty()) {
#ifndef LIBTORCH_OFF
            return std::make_unique<TorchNeuralNetwork>(modelPath, gameType, boardSize, useGpu);
#else
            // If LibTorch is disabled, log a message and fall back
            std::cerr << "LibTorch is disabled. Falling back to RandomPolicyNetwork." << std::endl;
            return std::make_unique<RandomPolicyNetwork>(gameType, boardSize);
#endif
        }
        
        // If no model path, fall back to random policy network
        return std::make_unique<RandomPolicyNetwork>(gameType, boardSize);
    } catch (const std::exception& e) {
        // Log error and fall back to RandomPolicyNetwork
        std::cerr << "Failed to create Neural Network: " << e.what() << std::endl;
        std::cerr << "Falling back to RandomPolicyNetwork" << std::endl;
        return std::make_unique<RandomPolicyNetwork>(gameType, boardSize);
    }
}

} // namespace nn
} // namespace alphazero