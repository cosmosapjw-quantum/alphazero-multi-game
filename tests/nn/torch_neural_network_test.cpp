#include <gtest/gtest.h>
#include "alphazero/nn/torch_neural_network.h"
#include "alphazero/games/gomoku/gomoku_state.h"

// Skip tests if LibTorch is not available
#ifndef LIBTORCH_OFF

namespace alphazero {
namespace nn {

class TorchNeuralNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a game state
        state = std::make_unique<gomoku::GomokuState>(15, false);
        
        // Skip setup if LibTorch is not available
        hasTorch = true;
        try {
            // Try to create a torch neural network with a random file that doesn't exist
            // (we don't have a real model file for testing)
            // This will load a fallback RandomPolicyNetwork
            network = std::make_unique<TorchNeuralNetwork>(
                "nonexistent_model_file.pt", 
                core::GameType::GOMOKU, 
                15, 
                false);  // Use CPU for testing
        } catch (const std::exception& e) {
            // If creation failed, mark as not having torch
            hasTorch = false;
            GTEST_SKIP() << "LibTorch not available or error loading model: " << e.what();
        }
    }
    
    std::unique_ptr<gomoku::GomokuState> state;
    std::unique_ptr<TorchNeuralNetwork> network;
    bool hasTorch = false;
};

TEST_F(TorchNeuralNetworkTest, Initialization) {
    if (!hasTorch) GTEST_SKIP();
    
    // Check if network was created
    EXPECT_NE(network, nullptr);
}

TEST_F(TorchNeuralNetworkTest, Predict) {
    if (!hasTorch) GTEST_SKIP();
    
    // Get prediction
    auto [policy, value] = network->predict(*state);
    
    // Policy should have correct size
    EXPECT_EQ(policy.size(), state->getActionSpaceSize());
    
    // Policy should sum to ~1.0
    float sum = 0.0f;
    for (float p : policy) {
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 0.1f);
    
    // Value should be in valid range
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
}

TEST_F(TorchNeuralNetworkTest, PredictBatch) {
    if (!hasTorch) GTEST_SKIP();
    
    // Create a batch of 2 states
    std::vector<std::reference_wrapper<const core::IGameState>> states;
    states.emplace_back(*state);
    
    // Make a move and add to batch
    auto state2 = state->clone();
    state2->makeMove(7 * 15 + 7);
    states.emplace_back(*state2);
    
    // Predict batch
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
    network->predictBatch(states, policies, values);
    
    // Should have 2 policies and 2 values
    EXPECT_EQ(policies.size(), 2);
    EXPECT_EQ(values.size(), 2);
    
    // Each policy should have correct size
    for (const auto& policy : policies) {
        EXPECT_EQ(policy.size(), state->getActionSpaceSize());
    }
    
    // All values should be in valid range
    for (float v : values) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST_F(TorchNeuralNetworkTest, PredictAsync) {
    if (!hasTorch) GTEST_SKIP();
    
    // Get async prediction
    auto future = network->predictAsync(*state);
    
    // Wait for result
    auto [policy, value] = future.get();
    
    // Policy should have correct size
    EXPECT_EQ(policy.size(), state->getActionSpaceSize());
    
    // Value should be in valid range
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
}

TEST_F(TorchNeuralNetworkTest, DeviceInfo) {
    if (!hasTorch) GTEST_SKIP();
    
    // Get device info
    std::string deviceInfo = network->getDeviceInfo();
    
    // Should have device info
    EXPECT_FALSE(deviceInfo.empty());
}

TEST_F(TorchNeuralNetworkTest, ModelInfo) {
    if (!hasTorch) GTEST_SKIP();
    
    // Get model info
    std::string modelInfo = network->getModelInfo();
    
    // Should have model info
    EXPECT_FALSE(modelInfo.empty());
}

TEST_F(TorchNeuralNetworkTest, Benchmark) {
    if (!hasTorch) GTEST_SKIP();
    
    // Run benchmark with few iterations
    EXPECT_NO_THROW(network->benchmark(2, 2));
}

TEST_F(TorchNeuralNetworkTest, MultipleInferences) {
    if (!hasTorch) GTEST_SKIP();
    
    // Run multiple inferences on the same state
    for (int i = 0; i < 5; ++i) {
        auto [policy, value] = network->predict(*state);
        
        // Results should be valid
        EXPECT_EQ(policy.size(), state->getActionSpaceSize());
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 1.0f);
    }
}

} // namespace nn
} // namespace alphazero

#endif // LIBTORCH_OFF