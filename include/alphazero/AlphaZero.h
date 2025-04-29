#pragma once

#include <string>
#include <memory>
#include <vector>

namespace alphazero {

class AlphaZero {
public:
    AlphaZero();
    ~AlphaZero();

    // Initialize the AlphaZero system
    bool initialize(const std::string& configFile);

    // Run a self-play session
    bool runSelfPlay(int numGames);

    // Train the neural network
    bool trainNetwork(const std::string& datasetPath);

    // Evaluate a model
    double evaluateModel(const std::string& modelPath, int numGames);

    // Run MCTS with current model and return best action
    int getBestAction(const std::vector<float>& state);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace alphazero