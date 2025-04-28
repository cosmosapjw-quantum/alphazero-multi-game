// include/alphazero/nn/torch_neural_network.h
#ifndef TORCH_NEURAL_NETWORK_H
#define TORCH_NEURAL_NETWORK_H

#include "alphazero/nn/neural_network.h"

// Only include torch headers if LibTorch is enabled
#ifndef LIBTORCH_OFF
#include <torch/torch.h>
#endif

#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <future>

namespace alphazero {
namespace nn {

/**
 * @brief PyTorch implementation of the neural network
 */
class TorchNeuralNetwork : public NeuralNetwork {
public:
    /**
     * @brief Constructor
     * 
     * @param modelPath Path to model file
     * @param gameType Game type
     * @param boardSize Board size
     * @param useGpu Whether to use GPU
     */
    TorchNeuralNetwork(const std::string& modelPath, 
                      core::GameType gameType,
                      int boardSize = 0,
                      bool useGpu = true);
    
    /**
     * @brief Destructor
     */
    ~TorchNeuralNetwork();
    
    // NeuralNetwork interface implementation
    std::pair<std::vector<float>, float> predict(const core::IGameState& state) override;
    
    void predictBatch(
        const std::vector<std::reference_wrapper<const core::IGameState>>& states,
        std::vector<std::vector<float>>& policies,
        std::vector<float>& values
    ) override;
    
    std::future<std::pair<std::vector<float>, float>> predictAsync(
        const core::IGameState& state
    ) override;
    
    bool isGpuAvailable() const override;
    std::string getDeviceInfo() const override;
    float getInferenceTimeMs() const override;
    int getBatchSize() const override;
    std::string getModelInfo() const override;
    size_t getModelSizeBytes() const override;
    void benchmark(int numIterations = 100, int batchSize = 16) override;
    void enableDebugMode(bool enable) override;
    void printModelSummary() const override;
    
private:
    // Game and board information
    core::GameType gameType_;
    int boardSize_;
    int inputChannels_;
    int actionSpaceSize_;
    
#ifndef LIBTORCH_OFF
    // PyTorch model
    torch::jit::script::Module model_;
    torch::Device device_;
#endif

    bool isGpu_;
    bool debugMode_;
    
    // Performance tracking
    mutable std::mutex mutex_;
    float avgInferenceTimeMs_;
    int batchSize_;
    
    // Batch processing
#ifndef LIBTORCH_OFF
    std::queue<std::pair<torch::Tensor, std::promise<std::pair<std::vector<float>, float>>>> batchQueue_;
#else
    std::queue<std::pair<int, std::promise<std::pair<std::vector<float>, float>>>> batchQueue_;
#endif
    std::mutex batchMutex_;
    std::condition_variable batchCondVar_;
    std::thread batchThread_;
    bool stopBatchThread_;
    
#ifndef LIBTORCH_OFF
    // Helper methods
    torch::Tensor stateTensor(const core::IGameState& state) const;
    std::pair<std::vector<float>, float> processOutput(const torch::jit::IValue& output, int actionSize) const;
#endif
    void batchProcessingLoop();
    void createModel();
};

} // namespace nn
} // namespace alphazero

#endif // TORCH_NEURAL_NETWORK_H