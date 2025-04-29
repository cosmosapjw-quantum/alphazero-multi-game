#include "alphazero/AlphaZero.h"
#include <torch/torch.h>
#include <iostream>

namespace alphazero {

// Pimpl implementation
class AlphaZero::Impl {
public:
    Impl() {
        std::cout << "AlphaZero initialized" << std::endl;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Using GPU." << std::endl;
            std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
            
            // Set default device 0
            int device_id = 0;
            auto device = torch::Device(torch::kCUDA, device_id);
            std::cout << "Using CUDA device: " << device_id << std::endl;
            
            // Create a test tensor to verify CUDA works
            torch::Tensor test_tensor = torch::ones({1, 1}, device);
            std::cout << "Test tensor on CUDA: " << test_tensor << std::endl;
        } else {
            std::cout << "CUDA is not available. Using CPU." << std::endl;
        }
    }
    
    bool initialize(const std::string& configFile) {
        std::cout << "Initializing AlphaZero with config: " << configFile << std::endl;
        return true;
    }
    
    bool runSelfPlay(int numGames) {
        std::cout << "Running " << numGames << " self-play games" << std::endl;
        return true;
    }
    
    bool trainNetwork(const std::string& datasetPath) {
        std::cout << "Training neural network with dataset: " << datasetPath << std::endl;
        return true;
    }
    
    double evaluateModel(const std::string& modelPath, int numGames) {
        std::cout << "Evaluating model " << modelPath << " with " << numGames << " games" << std::endl;
        return 0.5;  // Placeholder win rate
    }
    
    int getBestAction(const std::vector<float>& state) {
        std::cout << "Getting best action for state of size " << state.size() << std::endl;
        return 0;  // Placeholder action
    }
};

// AlphaZero implementation
AlphaZero::AlphaZero() : pImpl(std::make_unique<Impl>()) {}

AlphaZero::~AlphaZero() = default;

bool AlphaZero::initialize(const std::string& configFile) {
    return pImpl->initialize(configFile);
}

bool AlphaZero::runSelfPlay(int numGames) {
    return pImpl->runSelfPlay(numGames);
}

bool AlphaZero::trainNetwork(const std::string& datasetPath) {
    return pImpl->trainNetwork(datasetPath);
}

double AlphaZero::evaluateModel(const std::string& modelPath, int numGames) {
    return pImpl->evaluateModel(modelPath, numGames);
}

int AlphaZero::getBestAction(const std::vector<float>& state) {
    return pImpl->getBestAction(state);
}

} // namespace alphazero