#include <gtest/gtest.h>
#include "alphazero/nn/batch_queue.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace nn {

class BatchQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a game state
        state = std::make_unique<gomoku::GomokuState>(15, false);
        
        // Create a neural network
        network = NeuralNetwork::create("", core::GameType::GOMOKU, 15);
        
        // Create a batch queue with batch size 4 and timeout 20ms
        queue = std::make_unique<BatchQueue>(network.get(), 4, 20);
    }
    
    std::unique_ptr<gomoku::GomokuState> state;
    std::unique_ptr<NeuralNetwork> network;
    std::unique_ptr<BatchQueue> queue;
};

TEST_F(BatchQueueTest, EnqueueSingle) {
    // Enqueue a single state
    auto future = queue->enqueue(*state);
    
    // Wait for result
    auto [policy, value] = future.get();
    
    // Should have valid result
    EXPECT_EQ(policy.size(), state->getActionSpaceSize());
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
}

TEST_F(BatchQueueTest, EnqueueMultiple) {
    // Enqueue multiple states
    std::vector<std::future<std::pair<std::vector<float>, float>>> futures;
    
    for (int i = 0; i < 10; ++i) {
        futures.push_back(queue->enqueue(*state));
    }
    
    // Wait for all results
    for (auto& future : futures) {
        auto [policy, value] = future.get();
        
        // Should have valid result
        EXPECT_EQ(policy.size(), state->getActionSpaceSize());
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 1.0f);
    }
}

TEST_F(BatchQueueTest, SetBatchSize) {
    // Set batch size
    queue->setBatchSize(8);
    
    // Should reflect the new value
    EXPECT_EQ(queue->getBatchSize(), 8);
    
    // Test with invalid value
    EXPECT_THROW(queue->setBatchSize(0), std::invalid_argument);
    
    // Batch size should remain unchanged after error
    EXPECT_EQ(queue->getBatchSize(), 8);
}

TEST_F(BatchQueueTest, SetTimeout) {
    // Set timeout
    queue->setTimeout(50);
    
    // Should reflect the new value
    EXPECT_EQ(queue->getTimeout(), 50);
}

TEST_F(BatchQueueTest, PendingRequests) {
    // Should start with 0 pending requests
    EXPECT_EQ(queue->getPendingRequests(), 0);
    
    // Enqueue some requests in a non-blocking way
    std::vector<std::future<std::pair<std::vector<float>, float>>> futures;
    const int numRequests = 10;
    
    for (int i = 0; i < numRequests; ++i) {
        futures.push_back(queue->enqueue(*state));
    }
    
    // Wait a short time for processing to begin
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    // Should have some pending or processed requests
    EXPECT_LE(queue->getPendingRequests(), numRequests);
    
    // Wait for all results
    for (auto& future : futures) {
        future.get();
    }
    
    // Wait a bit for queue to process everything
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Should have 0 pending requests again
    EXPECT_EQ(queue->getPendingRequests(), 0);
}

TEST_F(BatchQueueTest, ProcessedCounts) {
    // Initially zero processed requests and batches
    EXPECT_EQ(queue->getProcessedRequestsCount(), 0);
    EXPECT_EQ(queue->getProcessedBatchesCount(), 0);
    
    // Process some requests
    std::vector<std::future<std::pair<std::vector<float>, float>>> futures;
    const int numRequests = 10;
    
    for (int i = 0; i < numRequests; ++i) {
        futures.push_back(queue->enqueue(*state));
    }
    
    // Wait for all results
    for (auto& future : futures) {
        future.get();
    }
    
    // Should have processed all requests
    EXPECT_EQ(queue->getProcessedRequestsCount(), numRequests);
    
    // Should have processed some batches (based on batch size)
    int expectedBatches = (numRequests + queue->getBatchSize() - 1) / queue->getBatchSize();
    EXPECT_EQ(queue->getProcessedBatchesCount(), expectedBatches);
}

TEST_F(BatchQueueTest, DifferentStates) {
    // Create states with different moves
    std::vector<std::unique_ptr<gomoku::GomokuState>> states;
    std::vector<std::future<std::pair<std::vector<float>, float>>> futures;
    
    // Create 5 different states
    for (int i = 0; i < 5; ++i) {
        auto newState = std::make_unique<gomoku::GomokuState>(15, false);
        
        // Make some moves
        for (int j = 0; j < i; ++j) {
            newState->makeMove(j * 15 + j);
        }
        
        // Enqueue the state
        futures.push_back(queue->enqueue(*newState));
        
        // Store for lifetime
        states.push_back(std::move(newState));
    }
    
    // Wait for all results
    for (auto& future : futures) {
        auto [policy, value] = future.get();
        
        // Should have valid result
        EXPECT_EQ(policy.size(), state->getActionSpaceSize());
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 1.0f);
    }
}

TEST_F(BatchQueueTest, Stress) {
    // Stress test with many concurrent requests
    const int numThreads = 4;
    const int requestsPerThread = 25;
    
    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};
    
    // Start threads
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, requestsPerThread, &successCount]() {
            for (int i = 0; i < requestsPerThread; ++i) {
                try {
                    auto future = queue->enqueue(*state);
                    auto [policy, value] = future.get();
                    
                    // Verify result
                    if (policy.size() == state->getActionSpaceSize() &&
                        value >= -1.0f && value <= 1.0f) {
                        successCount++;
                    }
                } catch (...) {
                    // Count failed
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All requests should have succeeded
    EXPECT_EQ(successCount, numThreads * requestsPerThread);
}

} // namespace nn
} // namespace alphazero