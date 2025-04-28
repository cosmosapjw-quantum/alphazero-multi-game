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
#include "alphazero/core/igamestate.h"
#include "alphazero/nn/neural_network.h"

namespace alphazero {
namespace nn {

/**
 * @brief Queue for batching neural network inference requests
 * 
 * This class collects individual inference requests and processes them
 * in batches for improved efficiency on GPU.
 */
class BatchQueue {
public:
    /**
     * @brief Constructor
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
     * @return Future containing the inference result
     */
    std::future<std::pair<std::vector<float>, float>> enqueue(
        const core::IGameState& state);
    
    /**
     * @brief Get current batch size
     * 
     * @return Current batch size
     */
    int getBatchSize() const { return batchSize_; }
    
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
    void setTimeout(int timeoutMs) { timeoutMs_ = timeoutMs; }
    
    /**
     * @brief Get current timeout
     * 
     * @return Timeout in milliseconds
     */
    int getTimeout() const { return timeoutMs_; }
    
    /**
     * @brief Get number of pending requests
     * 
     * @return Number of pending requests
     */
    int getPendingRequests() const;
    
    /**
     * @brief Get total number of processed requests
     * 
     * @return Total number of processed requests
     */
    int getProcessedRequestsCount() const { return processedRequests_.load(); }
    
    /**
     * @brief Get total number of batches processed
     * 
     * @return Total number of batches processed
     */
    int getProcessedBatchesCount() const { return processedBatches_.load(); }
    
private:
    struct Request {
        std::unique_ptr<core::IGameState> state;
        std::promise<std::pair<std::vector<float>, float>> promise;
    };
    
    NeuralNetwork* neuralNetwork_;
    int batchSize_;
    int timeoutMs_;
    
    std::queue<std::unique_ptr<Request>> requestQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCondVar_;
    std::atomic<bool> stopProcessing_;
    
    std::thread processingThread_;
    std::atomic<int> processedRequests_{0};
    std::atomic<int> processedBatches_{0};
    
    void processingLoop();
    void processBatch(std::vector<std::unique_ptr<Request>>& batch);
};

} // namespace nn
} // namespace alphazero

#endif // BATCH_QUEUE_H