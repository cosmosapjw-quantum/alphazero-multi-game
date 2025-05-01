// include/alphazero/nn/batch_queue.h
#ifndef BATCH_QUEUE_H
#define BATCH_QUEUE_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>
#include <functional>
#include <memory>
#include <atomic>
#include <chrono>
#include <sstream>
#include <string>
#include "alphazero/core/igamestate.h"
#include "alphazero/nn/neural_network.h"

// Remove pybind11 dependency

namespace alphazero {
namespace nn {

/**
 * @brief Configuration for the batch queue
 */
struct BatchQueueConfig {
    int batchSize = 16;                // Maximum batch size
    int timeoutMs = 10;                // Maximum time to wait for a full batch (milliseconds)
    int maxQueueSize = 1024;           // Maximum queue size
    int numWorkerThreads = 1;          // Number of worker threads
    bool prioritizeBatchSize = true;   // Prioritize full batches over latency
    int minBatchSize = 1;              // Minimum batch size to process
    bool useAdaptiveBatching = true;   // Adapt batch size based on queue pressure
    int adaptiveBatchInterval = 100;   // Interval to adapt batch size (milliseconds)
    int maxAdaptiveBatchSize = 64;     // Maximum adaptive batch size
};

/**
 * @brief Statistics for the batch queue
 */
struct BatchQueueStats {
    std::atomic<size_t> totalRequests{0};
    std::atomic<size_t> totalBatches{0};
    std::atomic<size_t> totalTimedOutBatches{0};
    std::atomic<size_t> avgBatchSize{0};
    std::atomic<size_t> maxQueueSize{0};
    std::atomic<size_t> avgQueueWaitTimeMs{0};
    std::atomic<size_t> avgProcessingTimeMs{0};
    std::atomic<size_t> droppedRequests{0};
    
    void reset() {
        totalRequests = 0;
        totalBatches = 0;
        totalTimedOutBatches = 0;
        avgBatchSize = 0;
        maxQueueSize = 0;
        avgQueueWaitTimeMs = 0;
        avgProcessingTimeMs = 0;
        droppedRequests = 0;
    }
    
    std::string toString() const {
        std::stringstream ss;
        ss << "Batch Queue Stats:" << std::endl;
        ss << "  Total requests: " << totalRequests << std::endl;
        ss << "  Total batches: " << totalBatches << std::endl;
        ss << "  Timed out batches: " << totalTimedOutBatches << std::endl;
        ss << "  Avg batch size: " << (totalBatches > 0 ? avgBatchSize.load() / totalBatches.load() : 0) << std::endl;
        ss << "  Max queue size: " << maxQueueSize << std::endl;
        ss << "  Avg queue wait time: " << 
            (totalRequests > 0 ? avgQueueWaitTimeMs.load() / totalRequests.load() : 0) << " ms" << std::endl;
        ss << "  Avg processing time: " << 
            (totalBatches > 0 ? avgProcessingTimeMs.load() / totalBatches.load() : 0) << " ms" << std::endl;
        ss << "  Dropped requests: " << droppedRequests << std::endl;
        return ss.str();
    }
};

/**
 * @brief Queue for batching neural network inference requests
 * 
 * This class collects individual inference requests and processes them
 * in batches for improved efficiency on GPU.
 * 
 * WARNING: Using a Python-backed neural network with this class will limit parallelism
 * due to the Python Global Interpreter Lock (GIL). For best performance, use a
 * C++-based neural network implementation like TorchNeuralNetwork that directly
 * uses LibTorch without Python bindings.
 * 
 * If you need to use a Python model, export it to LibTorch format first to achieve
 * proper parallelism.
 */
class BatchQueue {
public:
    /**
     * @brief Constructor
     * 
     * @param neuralNetwork Neural network to use for inference
     * @param config Batch queue configuration
     */
    BatchQueue(NeuralNetwork* neuralNetwork, const BatchQueueConfig& config = BatchQueueConfig());
    
    /**
     * @brief Constructor with basic parameters
     * 
     * @param neuralNetwork Neural network to use for inference
     * @param batchSize Maximum batch size
     * @param timeoutMs Maximum time to wait for a full batch (milliseconds)
     */
    BatchQueue(NeuralNetwork* neuralNetwork, int batchSize = 16, int timeoutMs = 10);
    
    /**
     * @brief Destructor
     */
    ~BatchQueue();
    
    /**
     * @brief Enqueue a state for inference
     * 
     * @param state Game state to evaluate
     * @param priority Priority level (higher is processed first)
     * @return Future containing the inference result
     */
    std::future<std::pair<std::vector<float>, float>> enqueue(
        const core::IGameState& state, int priority = 0);
    
    /**
     * @brief Get batch queue configuration
     * 
     * @return Current configuration
     */
    const BatchQueueConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set batch queue configuration
     * 
     * @param config New configuration
     */
    void setConfig(const BatchQueueConfig& config);
    
    /**
     * @brief Set batch size
     * 
     * @param batchSize New batch size
     */
    void setBatchSize(int batchSize);
    
    /**
     * @brief Set timeout in milliseconds
     * 
     * @param timeoutMs Timeout in milliseconds
     */
    void setTimeout(int timeoutMs) { config_.timeoutMs = timeoutMs; }
    
    /**
     * @brief Get current batch size
     * 
     * @return Current batch size
     */
    int getBatchSize() const { return config_.batchSize; }
    
    /**
     * @brief Get current timeout
     * 
     * @return Timeout in milliseconds
     */
    int getTimeout() const { return config_.timeoutMs; }
    
    /**
     * @brief Get number of pending requests
     * 
     * @return Number of pending requests
     */
    int getPendingRequests() const;
    
    /**
     * @brief Get statistics
     * 
     * @return Batch queue statistics
     */
    const BatchQueueStats& getStats() const { return stats_; }
    
    /**
     * @brief Reset statistics
     */
    void resetStats() { stats_.reset(); }
    
    /**
     * @brief Get neural network
     * 
     * @return Neural network pointer
     */
    NeuralNetwork* getNeuralNetwork() const { return neuralNetwork_; }
    
    /**
     * @brief Set neural network
     * 
     * @param neuralNetwork Neural network pointer
     */
    void setNeuralNetwork(NeuralNetwork* neuralNetwork) { neuralNetwork_ = neuralNetwork; }

private:
    /**
     * @brief Request structure
     */
    struct Request {
        std::unique_ptr<core::IGameState> state;
        std::promise<std::pair<std::vector<float>, float>> promise;
        int priority;
        std::chrono::steady_clock::time_point enqueueTime;
        
        Request(std::unique_ptr<core::IGameState> s, int p = 0)
            : state(std::move(s)), priority(p), enqueueTime(std::chrono::steady_clock::now()) {}
        
        // Comparison for priority queue
        bool operator<(const Request& other) const {
            return priority < other.priority;
        }
    };
    
    // Compare for priority queue (higher priority first)
    struct RequestCompare {
        bool operator()(const std::unique_ptr<Request>& a, const std::unique_ptr<Request>& b) const {
            return a->priority < b->priority;
        }
    };
    
    /**
     * @brief State batch structure for processing
     */
    struct StateBatch {
        std::vector<std::reference_wrapper<const core::IGameState>> states;
        std::vector<std::promise<std::pair<std::vector<float>, float>>> promises;
        std::vector<std::chrono::steady_clock::time_point> enqueueTimes;
    };
    
    NeuralNetwork* neuralNetwork_;
    BatchQueueConfig config_;
    
    // Prioritized request queue
    std::priority_queue<std::unique_ptr<Request>, 
                       std::vector<std::unique_ptr<Request>>, 
                       RequestCompare> requestQueue_;
    mutable std::mutex queueMutex_;
    std::condition_variable queueCondVar_;
    std::atomic<bool> stopProcessing_;
    
    // Adaptive batching
    int currentBatchSize_;
    std::atomic<int> queuePressure_{0};
    std::chrono::steady_clock::time_point lastBatchSizeUpdate_;
    
    // Worker threads
    std::vector<std::thread> processingThreads_;
    
    // Statistics
    BatchQueueStats stats_;
    
    // Helper methods
    void processingLoop();
    void processBatch(StateBatch& batch);
    void updateAdaptiveBatchSize();
    int calculateOptimalBatchSize() const;
};

} // namespace nn
} // namespace alphazero

#endif // BATCH_QUEUE_H