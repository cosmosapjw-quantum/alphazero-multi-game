#include <gtest/gtest.h>
#include "alphazero/nn/batch_queue.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/random_policy_network.h"
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace nn {

// Mock Neural Network for testing BatchQueue
class MockNeuralNetwork : public NeuralNetwork {
public:
    MockNeuralNetwork() {}
    
    std::pair<std::vector<float>, float> predict(const core::IGameState& state) override {
        std::vector<float> policy(state.getActionSpaceSize(), 0.1f);
        return {policy, 0.0f};
    }
    
    void predictBatch(
        const std::vector<std::reference_wrapper<const core::IGameState>>& states,
        std::vector<std::vector<float>>& policies,
        std::vector<float>& values
    ) override {
        policies.clear();
        values.clear();
        
        for (const auto& state : states) {
            policies.push_back(std::vector<float>(state.get().getActionSpaceSize(), 0.1f));
            values.push_back(0.0f);
        }
    }
    
    std::future<std::pair<std::vector<float>, float>> predictAsync(
        const core::IGameState& state
    ) override {
        std::promise<std::pair<std::vector<float>, float>> promise;
        std::vector<float> policy(state.getActionSpaceSize(), 0.1f);
        promise.set_value({policy, 0.0f});
        return promise.get_future();
    }
    
    bool isGpuAvailable() const override { return false; }
    std::string getDeviceInfo() const override { return "Mock"; }
    float getInferenceTimeMs() const override { return 0.1f; }
    int getBatchSize() const override { return 8; }
    std::string getModelInfo() const override { return "Mock"; }
    size_t getModelSizeBytes() const override { return 0; }
    void benchmark(int numIterations = 100, int batchSize = 16) override {}
    void enableDebugMode(bool enable) override {}
    void printModelSummary() const override {}
};

class BatchQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a game state
        state = std::make_unique<gomoku::GomokuState>(9, false);
        
        // Create mock neural network
        network = std::make_unique<MockNeuralNetwork>();
        
        // Create a batch queue with batch size 4 and timeout 20ms
        BatchQueueConfig config;
        config.batchSize = 4;
        config.timeoutMs = 20;
        config.numWorkerThreads = 1;
        queue = std::make_unique<BatchQueue>(network.get(), config);
    }
    
    std::unique_ptr<gomoku::GomokuState> state;
    std::unique_ptr<NeuralNetwork> network;
    std::unique_ptr<BatchQueue> queue;
};

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

} // namespace nn
} // namespace alphazero