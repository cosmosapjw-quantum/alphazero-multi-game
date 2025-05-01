// include/alphazero/nn/torch_neural_network.h
#ifndef TORCH_NEURAL_NETWORK_H
#define TORCH_NEURAL_NETWORK_H

#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/batch_queue.h"

// Only include torch headers if LibTorch is enabled
#ifndef LIBTORCH_OFF
#include <torch/torch.h>
#include "alphazero/nn/ddw_randwire_resnet.h"
#endif

#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <future>
#include <unordered_map>

namespace alphazero {
namespace nn {

/**
 * @brief Configuration for TorchNeuralNetwork
 */
struct TorchNeuralNetworkConfig {
    bool useGpu = true;                    // Use GPU if available
    bool useFp16 = false;                  // Use half precision (FP16)
    int batchSize = 16;                    // Default batch size
    int batchTimeoutMs = 10;               // Maximum time to wait for a batch (milliseconds)
    bool useTensorCaching = true;          // Cache tensors to avoid repeated conversions
    bool useJitScripting = true;           // Use JIT scripting for model optimization
    bool useAsyncExecution = true;         // Use asynchronous execution
    bool useOutputCompression = false;     // Apply output compression for less memory usage
    int maxCacheSize = 2048;               // Maximum tensor cache size
    int maxQueueSize = 512;                // Maximum queue size for async execution
    std::string modelBackend = "default";  // Model backend
    bool useNhwcFormat = false;            // Use NHWC format instead of NCHW
    bool useWarmup = true;                 // Perform warmup inferences at startup
    int numWarmupIterations = 10;          // Number of warmup iterations
    bool usePinnedMemory = true;           // Use pinned memory for better CPU-GPU transfers
    bool useThreadedInference = true;      // Use threaded inference for CPU backend
    int numThreads = 4;                    // Number of threads for CPU backend inference
};

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
     * @param config TorchNeuralNetwork configuration
     */
    TorchNeuralNetwork(const std::string& modelPath, 
                      core::GameType gameType,
                      int boardSize = 0,
                      bool useGpu = true,
                      const TorchNeuralNetworkConfig& config = TorchNeuralNetworkConfig());
    
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
    
    /**
     * @brief Get configuration
     * 
     * @return Current configuration
     */
    const TorchNeuralNetworkConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set configuration
     * 
     * @param config New configuration
     */
    void setConfig(const TorchNeuralNetworkConfig& config);
    
    /**
     * @brief Clear tensor cache
     */
    void clearCache();
    
    /**
     * @brief Get cache statistics
     * 
     * @return String with cache statistics
     */
    std::string getCacheStats() const;
    
    /**
     * @brief Export model to ONNX format
     * 
     * @param outputPath Output path
     * @return true if successful
     */
    bool exportToOnnx(const std::string& outputPath) const;
    
    /**
     * @brief Clear the batch queue
     */
    void clearBatchQueue();
    
    /**
     * @brief Get the batch queue stats
     * 
     * @return String with batch queue stats
     */
    std::string getBatchQueueStats() const;
    
    /**
     * @brief Create a new DDWRandWireResNet model
     * 
     * @param input_channels Number of input channels
     * @param output_size Size of policy output
     * @param channels Number of channels in the network (default: 128)
     * @param num_blocks Number of random wire blocks (default: 20)
     * @return Shared pointer to the created model
     */
    static std::shared_ptr<DDWRandWireResNet> createDDWRandWireResNet(
        int64_t input_channels, 
        int64_t output_size, 
        int64_t channels = 128, 
        int64_t num_blocks = 20);
    
    /**
     * @brief Export model in TorchScript format for CPU/GPU inference
     * 
     * @param modelPath Path to export the model
     * @return True if successful, false otherwise
     */
    bool exportToTorchScript(const std::string& modelPath) const;
    
private:
    // Game and board information
    core::GameType gameType_;
    int boardSize_;
    int inputChannels_;
    int actionSpaceSize_;
    TorchNeuralNetworkConfig config_;
    
#ifndef LIBTORCH_OFF
    // PyTorch model
    torch::jit::script::Module model_;
    torch::Device device_{torch::kCPU};
    
    // Tensor cache
    struct CacheEntry {
        torch::Tensor tensor;
        std::chrono::steady_clock::time_point lastAccess;
        
        CacheEntry(torch::Tensor t) 
            : tensor(std::move(t)), lastAccess(std::chrono::steady_clock::now()) {}
    };
    
    mutable std::mutex cacheMutex_;
    mutable std::unordered_map<uint64_t, CacheEntry> tensorCache_;
    
    // Cache statistics
    mutable std::atomic<size_t> cacheHits_{0};
    mutable std::atomic<size_t> cacheMisses_{0};
    mutable std::atomic<size_t> cacheSize_{0};
#endif

    bool isGpu_;
    bool debugMode_;
    
    // Performance tracking
    mutable std::mutex mutex_;
    float avgInferenceTimeMs_;
    int batchSize_;
    
    // Batch processing
    std::unique_ptr<BatchQueue> batchQueue_;
    
#ifndef LIBTORCH_OFF
    // Helper methods
    torch::Tensor stateTensor(const core::IGameState& state) const;
    torch::Tensor getCachedStateTensor(const core::IGameState& state) const;
    std::pair<std::vector<float>, float> processOutput(const torch::jit::IValue& output, int actionSize) const;
    void createModel();
    void performWarmup();
#endif

    // Feature map compression and decompression
    std::vector<float> compressPolicy(const std::vector<float>& policy) const;
    std::vector<float> decompressPolicy(const std::vector<float>& compressedPolicy, int actionSize) const;
};

} // namespace nn
} // namespace alphazero

#endif // TORCH_NEURAL_NETWORK_H