// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <future>
#include <functional>
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace nn {

/**
 * @brief Interface for neural network implementations
 * 
 * This abstract class defines the interface for neural networks
 * used for evaluating game states.
 */
class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;
    
    /**
     * @brief Evaluate a single state
     * 
     * @param state The game state to evaluate
     * @return Pair of (policy, value) where policy is a vector of move probabilities
     */
    virtual std::pair<std::vector<float>, float> predict(const core::IGameState& state) = 0;
    
    /**
     * @brief Evaluate multiple states in a batch
     * 
     * @param states Vector of game states to evaluate
     * @param policies Output vector of policy vectors
     * @param values Output vector of value estimates
     */
    virtual void predictBatch(
        const std::vector<std::reference_wrapper<const core::IGameState>>& states,
        std::vector<std::vector<float>>& policies,
        std::vector<float>& values
    ) = 0;
    
    /**
     * @brief Evaluate a single state asynchronously
     * 
     * @param state The game state to evaluate
     * @return Future for (policy, value) pair
     */
    virtual std::future<std::pair<std::vector<float>, float>> predictAsync(
        const core::IGameState& state
    ) = 0;
    
    /**
     * @brief Check if GPU is available
     * 
     * @return true if GPU is available, false otherwise
     */
    virtual bool isGpuAvailable() const = 0;
    
    /**
     * @brief Get information about the device
     * 
     * @return String with device information
     */
    virtual std::string getDeviceInfo() const = 0;
    
    /**
     * @brief Get average inference time in milliseconds
     * 
     * @return Average inference time
     */
    virtual float getInferenceTimeMs() const = 0;
    
    /**
     * @brief Get batch size for inference
     * 
     * @return Batch size
     */
    virtual int getBatchSize() const = 0;
    
    /**
     * @brief Get information about the model
     * 
     * @return String with model information
     */
    virtual std::string getModelInfo() const = 0;
    
    /**
     * @brief Get model size in bytes
     * 
     * @return Model size in bytes
     */
    virtual size_t getModelSizeBytes() const = 0;
    
    /**
     * @brief Run benchmark to measure performance
     * 
     * @param numIterations Number of iterations
     * @param batchSize Batch size to use
     */
    virtual void benchmark(int numIterations = 100, int batchSize = 16) = 0;
    
    /**
     * @brief Enable debug mode for extra logging
     * 
     * @param enable Whether to enable debug mode
     */
    virtual void enableDebugMode(bool enable) = 0;
    
    /**
     * @brief Print model summary to stdout
     */
    virtual void printModelSummary() const = 0;
    
    /**
     * @brief Create a neural network
     * 
     * @param modelPath Path to model file
     * @param gameType Game type
     * @param boardSize Board size (0 for default)
     * @param useGpu Whether to use GPU
     * @return Unique pointer to created neural network
     */
    static std::unique_ptr<NeuralNetwork> create(
        const std::string& modelPath,
        core::GameType gameType,
        int boardSize = 0,
        bool useGpu = true
    );
};

} // namespace nn
} // namespace alphazero

#endif // NEURAL_NETWORK_H