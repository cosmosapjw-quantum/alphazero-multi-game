// random_policy_network.h
#ifndef RANDOM_POLICY_NETWORK_H
#define RANDOM_POLICY_NETWORK_H

#include "alphazero/nn/neural_network.h"
#include <random>

namespace alphazero {
namespace nn {

/**
 * @brief Neural network with random policy for testing
 * 
 * This class implements a neural network that returns random
 * policy vectors and values for testing purposes.
 */
class RandomPolicyNetwork : public NeuralNetwork {
public:
    /**
     * @brief Constructor
     * 
     * @param gameType Type of game
     * @param boardSize Board size (0 for default)
     * @param seed Random seed (0 for random)
     */
    RandomPolicyNetwork(core::GameType gameType, int boardSize = 0, unsigned int seed = 0);
    
    /**
     * @brief Evaluate a single state
     * 
     * @param state The game state to evaluate
     * @return Pair of (policy, value) where policy is a vector of move probabilities
     */
    std::pair<std::vector<float>, float> predict(const core::IGameState& state) override;
    
    /**
     * @brief Evaluate multiple states in a batch
     * 
     * @param states Vector of game states to evaluate
     * @param policies Output vector of policy vectors
     * @param values Output vector of value estimates
     */
    void predictBatch(
        const std::vector<std::reference_wrapper<const core::IGameState>>& states,
        std::vector<std::vector<float>>& policies,
        std::vector<float>& values
    ) override;
    
    /**
     * @brief Evaluate a single state asynchronously
     * 
     * @param state The game state to evaluate
     * @return Future for (policy, value) pair
     */
    std::future<std::pair<std::vector<float>, float>> predictAsync(
        const core::IGameState& state
    ) override;
    
    /**
     * @brief Check if GPU is available
     * 
     * @return false (no GPU used)
     */
    bool isGpuAvailable() const override { return false; }
    
    /**
     * @brief Get information about the device
     * 
     * @return Device information string
     */
    std::string getDeviceInfo() const override { return "CPU (Random)"; }
    
    /**
     * @brief Get average inference time in milliseconds
     * 
     * @return Average inference time
     */
    float getInferenceTimeMs() const override { return 0.1f; }
    
    /**
     * @brief Get batch size for inference
     * 
     * @return Batch size
     */
    int getBatchSize() const override { return 128; }
    
    /**
     * @brief Get information about the model
     * 
     * @return Model information string
     */
    std::string getModelInfo() const override { return "Random policy network"; }
    
    /**
     * @brief Get model size in bytes
     * 
     * @return Model size in bytes
     */
    size_t getModelSizeBytes() const override { return 0; }
    
    /**
     * @brief Run benchmark to measure performance
     * 
     * @param numIterations Number of iterations
     * @param batchSize Batch size to use
     */
    void benchmark(int numIterations = 100, int batchSize = 16) override { /* No-op */ }
    
    /**
     * @brief Enable debug mode for extra logging
     * 
     * @param enable Whether to enable debug mode
     */
    void enableDebugMode(bool enable) override { /* No-op */ }
    
    /**
     * @brief Print model summary to stdout
     */
    void printModelSummary() const override { /* No-op */ }
    
private:
    core::GameType gameType_;   // Type of game
    int boardSize_;             // Board size
    std::mt19937 rng_;          // Random number generator
    
    /**
     * @brief Get default board size for game type
     * 
     * @param gameType Type of game
     * @return Default board size
     */
    int getDefaultBoardSize(core::GameType gameType) const;
    
    /**
     * @brief Generate a random policy for a state
     * 
     * @param state The game state
     * @return Random policy vector
     */
    std::vector<float> generateRandomPolicy(const core::IGameState& state) const;
    
    /**
     * @brief Generate a random value
     * 
     * @return Random value in [-0.1, 0.1]
     */
    float generateRandomValue() const;
};

} // namespace nn
} // namespace alphazero

#endif // RANDOM_POLICY_NETWORK_H