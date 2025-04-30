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
            
            // If queue is empty and not stopping, wait for new items
            if (requestQueue_.empty() && !stopProcessing_) {
                queueCondVar_.wait(lock, [this] { 
                    return !requestQueue_.empty() || stopProcessing_; 
                });
            }
            
            // Exit if stopping and queue is empty
            if (stopProcessing_ && requestQueue_.empty()) {
                break;
            }
            
            // If queue has enough items for a batch or we prioritize batch size
            if (!requestQueue_.empty() && 
                (requestQueue_.size() >= static_cast<size_t>(currentBatchSize_) || !config_.prioritizeBatchSize)) {
                // Collect batch
                int batchSize = std::min(static_cast<int>(requestQueue_.size()), currentBatchSize_);
                
                // Ensure minimum batch size
                if (batchSize < config_.minBatchSize && requestQueue_.size() < static_cast<size_t>(config_.minBatchSize)) {
                    // Wait for more items or timeout
                    auto deadline = std::chrono::steady_clock::now() + 
                                   std::chrono::milliseconds(config_.timeoutMs);
                    
                    bool timedOut = !queueCondVar_.wait_until(lock, deadline, [this] {
                        return requestQueue_.size() >= static_cast<size_t>(config_.minBatchSize) || 
                               stopProcessing_;
                    });
                    
                    if (timedOut) {
                        batchTimedOut = true;
                    }
                    
                    // Recalculate batch size
                    batchSize = std::min(static_cast<int>(requestQueue_.size()), currentBatchSize_);
                }
                
                // Skip if no items or stopping
                if (batchSize <= 0 || stopProcessing_) {
                    continue;
                }
                
                // Prepare batch vectors
                batch.states.reserve(batchSize);
                batch.promises.reserve(batchSize);
                batch.enqueueTimes.reserve(batchSize);
                
                // Collect items into batch safely
                for (int i = 0; i < batchSize && !requestQueue_.empty(); ++i) {
                    auto item = std::move(const_cast<std::unique_ptr<Request>&>(requestQueue_.top()));
                    requestQueue_.pop();
                    
                    if (item && item->state) {
                        batch.states.push_back(std::cref(*item->state));
                        batch.promises.push_back(std::move(item->promise));
                        batch.enqueueTimes.push_back(item->enqueueTime);
                    }
                }
                
                // Update queue pressure
                queuePressure_.store(static_cast<int>(requestQueue_.size()), std::memory_order_relaxed);
            } else if (!requestQueue_.empty()) {
                // Not enough items for a full batch, wait for more items or timeout
                auto deadline = std::chrono::steady_clock::now() + 
                               std::chrono::milliseconds(config_.timeoutMs);
                
                bool timedOut = !queueCondVar_.wait_until(lock, deadline, [this] {
                    return requestQueue_.size() >= static_cast<size_t>(currentBatchSize_) || 
                           stopProcessing_;
                });
                
                if (timedOut) {
                    batchTimedOut = true;
                    
                    // Process what we have if prioritizing latency or have minimum batch size
                    if (!config_.prioritizeBatchSize || 
                        requestQueue_.size() >= static_cast<size_t>(config_.minBatchSize)) {
                        int batchSize = std::min(static_cast<int>(requestQueue_.size()), currentBatchSize_);
                        
                        // Prepare batch vectors
                        batch.states.reserve(batchSize);
                        batch.promises.reserve(batchSize);
                        batch.enqueueTimes.reserve(batchSize);
                        
                        // Collect items into batch safely
                        for (int i = 0; i < batchSize && !requestQueue_.empty(); ++i) {
                            auto item = std::move(const_cast<std::unique_ptr<Request>&>(requestQueue_.top()));
                            requestQueue_.pop();
                            
                            if (item && item->state) {
                                batch.states.push_back(std::cref(*item->state));
                                batch.promises.push_back(std::move(item->promise));
                                batch.enqueueTimes.push_back(item->enqueueTime);
                            }
                        }
                        
                        // Update queue pressure
                        queuePressure_.store(static_cast<int>(requestQueue_.size()), std::memory_order_relaxed);
                    }
                }
            }
        } // End of mutex scope
        
        // Process the batch if not empty
        if (!batch.states.empty()) {
            auto batchStartTime = std::chrono::steady_clock::now();
            
            // Calculate queue wait times for statistics
            for (const auto& enqueueTime : batch.enqueueTimes) {
                auto waitTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    batchStartTime - enqueueTime).count();
                stats_.avgQueueWaitTimeMs.fetch_add(waitTime, std::memory_order_relaxed);
            }
            
            // Process the batch
            processBatch(batch);
            
            // Calculate processing time for statistics
            auto batchEndTime = std::chrono::steady_clock::now();
            auto processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                batchEndTime - batchStartTime).count();
            
            // Update statistics
            stats_.totalBatches.fetch_add(1, std::memory_order_relaxed);
            stats_.avgBatchSize.fetch_add(batch.states.size(), std::memory_order_relaxed);
            stats_.avgProcessingTimeMs.fetch_add(processingTime, std::memory_order_relaxed);
            
            if (batchTimedOut) {
                stats_.totalTimedOutBatches.fetch_add(1, std::memory_order_relaxed);
            }
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
    
#ifdef PYBIND11_MODULE
    // Release GIL during batch processing if this is being called from Python
    pybind11::gil_scoped_release release;
#endif
    
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