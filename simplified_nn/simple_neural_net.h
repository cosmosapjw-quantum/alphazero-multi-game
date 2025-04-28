#pragma once
#include <vector>
#include <string>

namespace alphazero {

class SimpleNeuralNetwork {
public:
    SimpleNeuralNetwork() {}
    ~SimpleNeuralNetwork() {}
    
    bool initialize() { return true; }
    bool load(const std::string& path) { return true; }
    std::vector<float> forward(const std::vector<float>& input) { 
        // Return dummy predictions
        return std::vector<float>(input.size(), 0.5f); 
    }
};

} // namespace alphazero
