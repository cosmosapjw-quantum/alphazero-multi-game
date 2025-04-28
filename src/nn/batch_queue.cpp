// src/nn/batch_queue.cpp
#include "alphazero/nn/batch_queue.h"
#include <chrono>
#include <algorithm>

namespace alphazero {
namespace nn {

BatchQueue::BatchQueue(NeuralNetwork* neuralNetwork, int batchSize, int timeoutMs)
    : neuralNetwork_(neuralNetwork), 
      batchSize_(batchSize), 
      timeoutMs_(timeoutMs),
      stopProcessing_(false) {
    
    // Start the processing thread
    processingThread_ = std::thread(&BatchQueue::processingLoop, this);
}

BatchQueue::~BatchQueue() {
    // Signal the processing thread to stop
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        stopProcessing_ = true;
    }
    queueCondVar_.notify_all();
    
    // Wait for the thread to finish
    if (processingThread_.joinable()) {
        processingThread_.join();
    }
}

std::future<std::pair<std::vector<float>, float>> BatchQueue::enqueue(
    const core::IGameState& state) {
    
    // Create a request
    auto request = std::make_unique<Request>();
    request->state = state.clone();
    std::future<std::pair<std::vector<float>, float>> future = request->promise.get_future();
    
    // Add to queue
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        requestQueue_.push(std::move(request));
    }
    
    // Notify processing thread
    queueCondVar_.notify_one();
    
    return future;
}

void BatchQueue::setBatchSize(int batchSize) {
    if (batchSize <= 0) {
        throw std::invalid_argument("Batch size must be positive");
    }
    
    std::unique_lock<std::mutex> lock(queueMutex_);
    batchSize_ = batchSize;
}

int BatchQueue::getPendingRequests() const {
    std::unique_lock<std::mutex> lock(queueMutex_);
    return static_cast<int>(requestQueue_.size());
}

void BatchQueue::processingLoop() {
    while (!stopProcessing_) {
        std::vector<std::unique_ptr<Request>> batch;
        
        // Wait for items to process
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            
            if (requestQueue_.empty() && !stopProcessing_) {
                // Wait for new items or stop signal
                queueCondVar_.wait(lock, [this] { 
                    return !requestQueue_.empty() || stopProcessing_; 
                });
            }
            
            if (stopProcessing_ && requestQueue_.empty()) {
                break;
            }
            
            // If the queue is not empty, collect items for a batch
            if (!requestQueue_.empty()) {
                if (requestQueue_.size() >= static_cast<size_t>(batchSize_)) {
                    // Queue has enough items for a full batch
                    for (int i = 0; i < batchSize_ && !requestQueue_.empty(); ++i) {
                        batch.push_back(std::move(requestQueue_.front()));
                        requestQueue_.pop();
                    }
                } else {
                    // Wait for more items or timeout
                    auto deadline = std::chrono::steady_clock::now() + 
                                  std::chrono::milliseconds(timeoutMs_);
                    
                    while (requestQueue_.size() < static_cast<size_t>(batchSize_) && !stopProcessing_) {
                        // Wait until deadline
                        if (queueCondVar_.wait_until(lock, deadline) == std::cv_status::timeout) {
                            break;
                        }
                    }
                    
                    // Process what we have after timeout
                    while (!requestQueue_.empty() && batch.size() < static_cast<size_t>(batchSize_)) {
                        batch.push_back(std::move(requestQueue_.front()));
                        requestQueue_.pop();
                    }
                }
            }
        }
        
        // Process the batch
        if (!batch.empty()) {
            processBatch(batch);
            
            // Update statistics
            processedRequests_.fetch_add(batch.size(), std::memory_order_relaxed);
            processedBatches_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void BatchQueue::processBatch(std::vector<std::unique_ptr<Request>>& batch) {
    if (batch.empty()) {
        return;
    }
    
    // Prepare input batch
    std::vector<std::reference_wrapper<const core::IGameState>> states;
    states.reserve(batch.size());
    
    for (const auto& request : batch) {
        states.push_back(std::cref(*request->state));
    }
    
    // Prepare output containers
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
    
    // Perform batch inference
    neuralNetwork_->predictBatch(states, policies, values);
    
    // Set results in promises
    for (size_t i = 0; i < batch.size(); ++i) {
        if (i < policies.size() && i < values.size()) {
            batch[i]->promise.set_value(std::make_pair(policies[i], values[i]));
        } else {
            // Handle error case (should never happen)
            std::vector<float> emptyPolicy(batch[i]->state->getActionSpaceSize(), 0.0f);
            batch[i]->promise.set_value(std::make_pair(emptyPolicy, 0.0f));
        }
    }
}

} // namespace nn
} // namespace alphazero