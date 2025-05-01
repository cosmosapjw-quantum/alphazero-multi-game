// src/nn/batch_queue.cpp
#include "alphazero/nn/batch_queue.h"
#include <algorithm>
#include <numeric>
#include <iostream>

namespace alphazero {
namespace nn {

BatchQueue::BatchQueue(NeuralNetwork* neuralNetwork, const BatchQueueConfig& config)
    : neuralNetwork_(neuralNetwork), 
      config_(config),
      currentBatchSize_(config.batchSize),
      lastBatchSizeUpdate_(std::chrono::steady_clock::now()),
      stopProcessing_(false),
      queuePressure_(0) {
    
    // Start the processing threads
    for (int i = 0; i < config_.numWorkerThreads; ++i) {
        processingThreads_.emplace_back(&BatchQueue::processingLoop, this);
    }
}

BatchQueue::BatchQueue(NeuralNetwork* neuralNetwork, int batchSize, int timeoutMs)
    : neuralNetwork_(neuralNetwork),
      currentBatchSize_(batchSize),
      lastBatchSizeUpdate_(std::chrono::steady_clock::now()),
      stopProcessing_(false),
      queuePressure_(0) {
    
    // Set configuration
    config_.batchSize = batchSize;
    config_.timeoutMs = timeoutMs;
    
    // Start the processing thread
    processingThreads_.emplace_back(&BatchQueue::processingLoop, this);
}

BatchQueue::~BatchQueue() {
    // Signal the processing threads to stop
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        stopProcessing_ = true;
    }
    queueCondVar_.notify_all();
    
    // Wait for the threads to finish
    for (auto& thread : processingThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Clear any remaining requests to avoid memory leaks
    std::unique_lock<std::mutex> lock(queueMutex_);
    while (!requestQueue_.empty()) {
        requestQueue_.pop();
    }
}

std::future<std::pair<std::vector<float>, float>> BatchQueue::enqueue(
    const core::IGameState& state, int priority) {
    
    if (!neuralNetwork_) {
        // Create a promise to return a default result
        std::promise<std::pair<std::vector<float>, float>> promise;
        std::vector<float> policy(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
        promise.set_value({policy, 0.0f});
        return promise.get_future();
    }
    
    // Record enqueue time for statistics
    auto enqueueTime = std::chrono::steady_clock::now();
    
    // Check if queue is full
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        if (requestQueue_.size() >= static_cast<size_t>(config_.maxQueueSize)) {
            stats_.droppedRequests.fetch_add(1, std::memory_order_relaxed);
            
            // Return a future that immediately sets a default result
            std::promise<std::pair<std::vector<float>, float>> promise;
            std::vector<float> policy(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
            promise.set_value({policy, 0.0f});
            return promise.get_future();
        }
    }
    
    try {
        // Create a request with a deep copy of the state
        auto request = std::make_unique<Request>(state.clone(), priority);
        std::future<std::pair<std::vector<float>, float>> future = request->promise.get_future();
        
        // Add to queue
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            
            // Update max queue size statistic
            size_t queueSize = requestQueue_.size();
            size_t currentMax = stats_.maxQueueSize.load(std::memory_order_relaxed);
            if (queueSize > currentMax) {
                stats_.maxQueueSize.store(queueSize, std::memory_order_relaxed);
            }
            
            // Add to priority queue
            requestQueue_.push(std::move(request));
            
            // Update queue pressure for adaptive batching
            queuePressure_.store(static_cast<int>(requestQueue_.size()), std::memory_order_relaxed);
            
            // Update statistics
            stats_.totalRequests.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Notify processing thread
        queueCondVar_.notify_one();
        
        return future;
    } catch (const std::exception& e) {
        // Handle clone failure gracefully
        std::promise<std::pair<std::vector<float>, float>> promise;
        std::vector<float> policy(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
        promise.set_value({policy, 0.0f});
        return promise.get_future();
    }
}

void BatchQueue::setConfig(const BatchQueueConfig& config) {
    std::unique_lock<std::mutex> lock(queueMutex_);
    
    // Update configuration
    config_ = config;
    currentBatchSize_ = config.batchSize;
    
    // Update number of worker threads if needed
    if (processingThreads_.size() != static_cast<size_t>(config.numWorkerThreads)) {
        // Stop all threads first
        stopProcessing_ = true;
        queueCondVar_.notify_all();
        
        lock.unlock();
        
        // Wait for threads to finish
        for (auto& thread : processingThreads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // Clear and restart
        processingThreads_.clear();
        
        lock.lock();
        
        stopProcessing_ = false;
        
        // Start new threads
        for (int i = 0; i < config.numWorkerThreads; ++i) {
            processingThreads_.emplace_back(&BatchQueue::processingLoop, this);
        }
    }
}

void BatchQueue::setBatchSize(int batchSize) {
    if (batchSize <= 0) {
        throw std::invalid_argument("Batch size must be positive");
    }
    
    std::unique_lock<std::mutex> lock(queueMutex_);
    config_.batchSize = batchSize;
    currentBatchSize_ = batchSize;
}

int BatchQueue::getPendingRequests() const {
    std::unique_lock<std::mutex> lock(queueMutex_);
    return static_cast<int>(requestQueue_.size());
}

void BatchQueue::processingLoop() {
    while (!stopProcessing_) {
        StateBatch batch;
        bool batchTimedOut = false;
        
        // Wait for items to process or update batch size
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            
            // Update adaptive batch size periodically
            if (config_.useAdaptiveBatching) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - lastBatchSizeUpdate_).count();
                
                if (elapsed >= config_.adaptiveBatchInterval) {
                    updateAdaptiveBatchSize();
                    lastBatchSizeUpdate_ = now;
                }
            }
            
            // Wait for minimum batch size or timeout
            auto batchSize = std::min(currentBatchSize_, static_cast<int>(requestQueue_.size()));
            
            // If queue is empty, wait for new items
            if (requestQueue_.empty()) {
                queueCondVar_.wait_for(lock, std::chrono::milliseconds(config_.timeoutMs),
                    [this] { return !requestQueue_.empty() || stopProcessing_; });
                
                // Check if we should exit
                if (stopProcessing_) {
                    break;
                }
                
                // If still empty after waiting, continue waiting
                if (requestQueue_.empty()) {
                    continue;
                }
            }
            
            // Collect items for batch
            auto startTime = std::chrono::steady_clock::now();
            
            // Try to fill batch up to target size or timeout
            while (batch.states.size() < static_cast<size_t>(currentBatchSize_) && !requestQueue_.empty()) {
                // Check timeout - try to form the largest batch possible within timeout
                auto now = std::chrono::steady_clock::now();
                auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                
                if (elapsedMs >= config_.timeoutMs && batch.states.size() >= static_cast<size_t>(config_.minBatchSize)) {
                    // If we have at least minimum batch size and timeout has elapsed, process what we have
                    batchTimedOut = true;
                    break;
                }
                
                // No timeout yet, process more items if available
                if (!requestQueue_.empty()) {
                    // Get next request - properly handling unique_ptr
                    // We need to get the top item, then pop it, then use it
                    // We can't move directly from top() as it returns a const reference
                    auto& topRequest = requestQueue_.top();
                    
                    // Add to batch
                    batch.states.push_back(*topRequest->state);
                    batch.promises.push_back(std::move(topRequest->promise));
                    batch.enqueueTimes.push_back(topRequest->enqueueTime);
                    
                    // Now we can pop it from the queue
                    requestQueue_.pop();
                }
                
                // If queue becomes empty, check if we should wait more or process what we have
                if (requestQueue_.empty() && batch.states.size() < static_cast<size_t>(config_.minBatchSize)) {
                    // Wait a bit more to see if we get more items
                    bool gotMore = queueCondVar_.wait_for(lock, 
                        std::chrono::milliseconds(std::max(1, config_.timeoutMs / 4)),
                        [this] { return !requestQueue_.empty() || stopProcessing_; });
                    
                    if (stopProcessing_) {
                        break;
                    }
                    
                    // If still empty and we have at least one item, just process what we have
                    if (!gotMore && !batch.states.empty()) {
                        batchTimedOut = true;
                        break;
                    }
                }
            }
        }
        
        // Process the batch if it's not empty
        if (!batch.states.empty()) {
            processBatch(batch);
        }
    }
}

void BatchQueue::processBatch(StateBatch& batch) {
    if (batch.states.empty() || !neuralNetwork_) {
        return;
    }
    
    // Prepare output containers
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
    
    // No Python GIL management required in C++ standalone implementation
    
    // Process batch
    try {
        // Perform batch inference
        neuralNetwork_->predictBatch(batch.states, policies, values);
        
        // Set results in promises
        for (size_t i = 0; i < batch.states.size(); ++i) {
            if (i < policies.size() && i < values.size()) {
                batch.promises[i].set_value(std::make_pair(policies[i], values[i]));
            } else {
                // Handle error case (should never happen)
                std::vector<float> emptyPolicy(batch.states[i].get().getActionSpaceSize(), 0.0f);
                batch.promises[i].set_value(std::make_pair(emptyPolicy, 0.0f));
            }
        }
    } catch (const std::exception& e) {
        // Handle exceptions during inference
        for (size_t i = 0; i < batch.promises.size(); ++i) {
            try {
                // Create a minimal policy for error cases
                std::vector<float> uniformPolicy;
                
                if (i < batch.states.size()) {
                    uniformPolicy.resize(batch.states[i].get().getActionSpaceSize(), 
                                      1.0f / batch.states[i].get().getActionSpaceSize());
                } else {
                    // Fallback if state reference is invalid
                    uniformPolicy.resize(10, 0.1f);
                }
                
                batch.promises[i].set_value(std::make_pair(uniformPolicy, 0.0f));
            } catch (...) {
                // Ignore broken promises
            }
        }
    }
}

void BatchQueue::updateAdaptiveBatchSize() {
    // Calculate optimal batch size based on queue pressure and neural network
    int optimalBatchSize = calculateOptimalBatchSize();
    
    // Update current batch size gradually to avoid rapid fluctuations
    if (optimalBatchSize > currentBatchSize_) {
        currentBatchSize_ = std::min(optimalBatchSize, currentBatchSize_ + 2);
    } else if (optimalBatchSize < currentBatchSize_) {
        currentBatchSize_ = std::max(optimalBatchSize, currentBatchSize_ - 1);
    }
    
    // Ensure within limits
    currentBatchSize_ = std::max(config_.minBatchSize, 
                               std::min(currentBatchSize_, config_.maxAdaptiveBatchSize));
}

int BatchQueue::calculateOptimalBatchSize() const {
    // Simple heuristic for batch size based on queue pressure
    int queueSize = queuePressure_.load(std::memory_order_relaxed);
    
    // If queue is nearly empty, use small batches for low latency
    if (queueSize <= config_.minBatchSize) {
        return config_.minBatchSize;
    }
    
    // If queue is growing, increase batch size for throughput
    if (queueSize > 2 * currentBatchSize_) {
        return std::min(queueSize / 2, config_.maxAdaptiveBatchSize);
    }
    
    // If neural network is GPU-accelerated, prefer larger batches
    if (neuralNetwork_ && neuralNetwork_->isGpuAvailable()) {
        return std::min(config_.batchSize * 2, config_.maxAdaptiveBatchSize);
    }
    
    // Default to configured batch size
    return config_.batchSize;
}

} // namespace nn
} // namespace alphazero