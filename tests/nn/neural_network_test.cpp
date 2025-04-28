#include <gtest/gtest.h>
#include "alphazero/nn/neural_network.h"
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace nn {

class NeuralNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a game state
        state = std::make_unique<gomoku::GomokuState>(15, false);
        
        // Create a random policy network
        network = NeuralNetwork::create("", core::GameType::GOMOKU, 15);
    }
    
    std::unique_ptr<gomoku::GomokuState> state;
    std::unique_ptr<NeuralNetwork> network;
};

TEST_F(NeuralNetworkTest, PredictInitialState) {
    // Get prediction for initial state
    auto [policy, value] = network->predict(*state);
    
    // Policy should have correct size
    EXPECT_EQ(policy.size(), state->getActionSpaceSize());
    
    // All policy values should be probabilities (sum to ~1.0)
    float sum = 0.0f;
    for (float p : policy) {
        // Each probability should be in [0, 1]
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 0.01f);
    
    // Value should be in [-1, 1]
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
}

TEST_F(NeuralNetworkTest, PredictBatch) {
    // Create a batch of 3 states
    std::vector<std::reference_wrapper<const core::IGameState>> states;
    states.emplace_back(*state);
    
    // Make a move and add to batch
    auto state2 = state->clone();
    state2->makeMove(7 * 15 + 7);
    states.emplace_back(*state2);
    
    // Make another move and add to batch
    auto state3 = state2->clone();
    state3->makeMove(8 * 15 + 8);
    states.emplace_back(*state3);
    
    // Predict batch
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
    network->predictBatch(states, policies, values);
    
    // Should have 3 policies and 3 values
    EXPECT_EQ(policies.size(), 3);
    EXPECT_EQ(values.size(), 3);
    
    // Each policy should have correct size
    for (const auto& policy : policies) {
        EXPECT_EQ(policy.size(), state->getActionSpaceSize());
    }
    
    // All values should be in [-1, 1]
    for (float v : values) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST_F(NeuralNetworkTest, PredictAsync) {
    // Get async prediction
    auto future = network->predictAsync(*state);
    
    // Wait for result
    auto [policy, value] = future.get();
    
    // Policy should have correct size
    EXPECT_EQ(policy.size(), state->getActionSpaceSize());
    
    // All policy values should be probabilities (sum to ~1.0)
    float sum = 0.0f;
    for (float p : policy) {
        // Each probability should be in [0, 1]
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 0.01f);
    
    // Value should be in [-1, 1]
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
}

} // namespace nn
} // namespace alphazero