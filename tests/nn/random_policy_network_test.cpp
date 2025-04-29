#include <gtest/gtest.h>
#include "alphazero/nn/random_policy_network.h"
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace nn {

class RandomPolicyNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a game state
        state = std::make_unique<gomoku::GomokuState>(15, false);
        
        // Create a random policy network
        network = std::make_unique<RandomPolicyNetwork>(core::GameType::GOMOKU, 15, 42);
    }
    
    std::unique_ptr<gomoku::GomokuState> state;
    std::unique_ptr<RandomPolicyNetwork> network;
};

TEST_F(RandomPolicyNetworkTest, Predict) {
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
    
    // Value should be in [-0.1, 0.1] for random policy
    EXPECT_GE(value, -0.1f);
    EXPECT_LE(value, 0.1f);
}

TEST_F(RandomPolicyNetworkTest, PredictBatch) {
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
    
    // All values should be in [-0.1, 0.1]
    for (float v : values) {
        EXPECT_GE(v, -0.1f);
        EXPECT_LE(v, 0.1f);
    }
}

TEST_F(RandomPolicyNetworkTest, PredictAsync) {
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
    
    // Value should be in [-0.1, 0.1]
    EXPECT_GE(value, -0.1f);
    EXPECT_LE(value, 0.1f);
}

TEST_F(RandomPolicyNetworkTest, Determinism) {
    // Using fixed seed should give deterministic results
    RandomPolicyNetwork network1(core::GameType::GOMOKU, 15, 42);
    RandomPolicyNetwork network2(core::GameType::GOMOKU, 15, 42);
    
    // Get predictions
    auto [policy1, value1] = network1.predict(*state);
    auto [policy2, value2] = network2.predict(*state);
    
    // Results should be identical
    EXPECT_EQ(policy1, policy2);
    EXPECT_FLOAT_EQ(value1, value2);
    
    // Different seeds should give different results
    RandomPolicyNetwork network3(core::GameType::GOMOKU, 15, 43);
    auto [policy3, value3] = network3.predict(*state);
    
    // At least one value should be different
    bool policiesDifferent = false;
    for (size_t i = 0; i < policy1.size(); ++i) {
        if (std::abs(policy1[i] - policy3[i]) > 1e-6) {
            policiesDifferent = true;
            break;
        }
    }
    
    EXPECT_TRUE(policiesDifferent || std::abs(value1 - value3) > 1e-6);
}

TEST_F(RandomPolicyNetworkTest, DifferentGameTypes) {
    // Create networks for different games
    RandomPolicyNetwork gomokuNet(core::GameType::GOMOKU, 15, 42);
    RandomPolicyNetwork chessNet(core::GameType::CHESS, 8, 42);
    RandomPolicyNetwork goNet(core::GameType::GO, 19, 42);
    
    // Get predictions from each
    auto [gomokuPolicy, gomokuValue] = gomokuNet.predict(*state);
    auto [chessPolicy, chessValue] = chessNet.predict(*state);
    auto [goPolicy, goValue] = goNet.predict(*state);
    
    // Policies should have appropriate sizes for each game
    EXPECT_EQ(gomokuPolicy.size(), 15 * 15);         // 15x15 board
    EXPECT_EQ(chessPolicy.size(), 64 * 64 * 5);      // From * To * Promotion options
    EXPECT_EQ(goPolicy.size(), 19 * 19 + 1);         // 19x19 board + pass
}

TEST_F(RandomPolicyNetworkTest, IllegalMoves) {
    // Make a move
    state->makeMove(7 * 15 + 7);
    
    // Get prediction
    auto [policy, value] = network->predict(*state);
    
    // Check that illegal moves have lower probability
    float occupiedProbability = policy[7 * 15 + 7];
    
    // Calculate average probability for legal moves
    float sumProbability = 0.0f;
    int legalCount = 0;
    
    for (int action = 0; action < state->getActionSpaceSize(); ++action) {
        if (state->isLegalMove(action)) {
            sumProbability += policy[action];
            legalCount++;
        }
    }
    
    float avgLegalProbability = sumProbability / legalCount;
    
    // Illegal move should have lower probability than average legal move
    EXPECT_LT(occupiedProbability, avgLegalProbability);
}

} // namespace nn
} // namespace alphazero