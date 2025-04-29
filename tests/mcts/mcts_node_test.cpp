#include <gtest/gtest.h>
#include "alphazero/mcts/mcts_node.h"
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace mcts {

class MCTSNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a game state
        state = std::make_unique<gomoku::GomokuState>(15, false);
        
        // Create root node
        rootNode = std::make_unique<MCTSNode>(state.get());
    }
    
    std::unique_ptr<gomoku::GomokuState> state;
    std::unique_ptr<MCTSNode> rootNode;
};

TEST_F(MCTSNodeTest, InitialState) {
    // Initial node should have 0 visits and value
    EXPECT_EQ(rootNode->visitCount.load(), 0);
    EXPECT_FLOAT_EQ(rootNode->valueSum.load(), 0.0f);
    EXPECT_FLOAT_EQ(rootNode->getValue(), 0.0f);
    
    // Should not be terminal
    EXPECT_FALSE(rootNode->isTerminal);
    
    // Should match the state hash
    EXPECT_EQ(rootNode->stateHash, state->getHash());
    
    // Should not be expanded
    EXPECT_FALSE(rootNode->isExpanded);
    
    // Should have no children
    EXPECT_FALSE(rootNode->hasChildren());
}

TEST_F(MCTSNodeTest, AddChild) {
    // Add a child node
    int action = 7 * 15 + 7;
    float prior = 0.1f;
    
    // Clone the state and make a move
    auto childState = state->clone();
    childState->makeMove(action);
    
    // Create child node
    auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
    
    // Add child to root
    rootNode->addChild(action, prior, std::move(childNode));
    
    // Root should now have one child
    EXPECT_TRUE(rootNode->hasChildren());
    EXPECT_EQ(rootNode->children.size(), 1);
    EXPECT_EQ(rootNode->actions.size(), 1);
    EXPECT_EQ(rootNode->actions[0], action);
    
    // Child should have correct parent and prior
    EXPECT_EQ(rootNode->children[0]->parent, rootNode.get());
    EXPECT_FLOAT_EQ(rootNode->children[0]->prior, prior);
}

TEST_F(MCTSNodeTest, BackPropagation) {
    // Test value backpropagation
    
    // First, create two levels of nodes
    int action1 = 7 * 15 + 7;
    float prior1 = 0.1f;
    
    // Clone the state and make a move
    auto childState = state->clone();
    childState->makeMove(action1);
    
    // Create child node
    auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior1);
    MCTSNode* childPtr = childNode.get();
    
    // Add child to root
    rootNode->addChild(action1, prior1, std::move(childNode));
    
    // Add a second level child
    int action2 = 8 * 15 + 8;
    float prior2 = 0.2f;
    
    // Clone the state and make another move
    auto grandchildState = childState->clone();
    grandchildState->makeMove(action2);
    
    // Create grandchild node
    auto grandchildNode = std::make_unique<MCTSNode>(grandchildState.get(), childPtr, prior2);
    MCTSNode* grandchildPtr = grandchildNode.get();
    
    // Add child to first child
    childPtr->addChild(action2, prior2, std::move(grandchildNode));
    
    // Simulate a visit to the grandchild with value 1.0
    grandchildPtr->visitCount.fetch_add(1);
    
    // For atomic<float>, we need to use exchange with the updated value
    float oldValue = grandchildPtr->valueSum.load();
    float newValue = oldValue + 1.0f;
    while (!grandchildPtr->valueSum.compare_exchange_weak(oldValue, newValue)) {
        newValue = oldValue + 1.0f;
    }
    
    // The visit and value should propagate up the tree
    EXPECT_EQ(grandchildPtr->visitCount.load(), 1);
    EXPECT_FLOAT_EQ(grandchildPtr->valueSum.load(), 1.0f);
    EXPECT_FLOAT_EQ(grandchildPtr->getValue(), 1.0f);
    
    // Update the parent
    childPtr->visitCount.fetch_add(1);
    
    // For atomic<float>, update parent value
    oldValue = childPtr->valueSum.load();
    newValue = oldValue - 1.0f; // Value flips sign for parent
    while (!childPtr->valueSum.compare_exchange_weak(oldValue, newValue)) {
        newValue = oldValue - 1.0f;
    }
    
    EXPECT_EQ(childPtr->visitCount.load(), 1);
    EXPECT_FLOAT_EQ(childPtr->valueSum.load(), -1.0f);
    EXPECT_FLOAT_EQ(childPtr->getValue(), -1.0f);
    
    // Update the root
    rootNode->visitCount.fetch_add(1);
    
    // For atomic<float>, update root value
    oldValue = rootNode->valueSum.load();
    newValue = oldValue + 1.0f; // Value flips sign again
    while (!rootNode->valueSum.compare_exchange_weak(oldValue, newValue)) {
        newValue = oldValue + 1.0f;
    }
    
    EXPECT_EQ(rootNode->visitCount.load(), 1);
    EXPECT_FLOAT_EQ(rootNode->valueSum.load(), 1.0f);
    EXPECT_FLOAT_EQ(rootNode->getValue(), 1.0f);
}

TEST_F(MCTSNodeTest, GetUcbScore) {
    // Add a child node
    int action = 7 * 15 + 7;
    float prior = 0.5f;
    
    // Clone the state and make a move
    auto childState = state->clone();
    childState->makeMove(action);
    
    // Create child node
    auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
    MCTSNode* childPtr = childNode.get();
    
    // Add child to root
    rootNode->addChild(action, prior, std::move(childNode));
    
    // Update root visit count to make parent visits non-zero
    rootNode->visitCount.store(1);
    
    // With zero visits to child, UCB should be influenced only by prior and exploration constant
    float cPuct = 1.5f;
    float ucbScore = childPtr->getUcbScore(cPuct, 1);
    
    // For unvisited node, score should be high
    EXPECT_GT(ucbScore, 1000.0f); // Should be close to infinity
    
    // After a visit, UCB should depend on value and prior
    childPtr->visitCount.store(1);
    childPtr->valueSum.store(0.5f);
    
    ucbScore = childPtr->getUcbScore(cPuct, 1);
    
    // Now the score should be more reasonable
    float expectedScore = 0.5f + cPuct * prior * sqrtf(1.0f) / (1.0f + 1.0f);
    EXPECT_NEAR(ucbScore, expectedScore, 0.001f);
}

TEST_F(MCTSNodeTest, VirtualLoss) {
    // Test virtual loss functionality
    
    // Add a child node
    int action = 7 * 15 + 7;
    float prior = 0.5f;
    
    // Clone the state and make a move
    auto childState = state->clone();
    childState->makeMove(action);
    
    // Create child node
    auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
    MCTSNode* childPtr = childNode.get();
    
    // Add child to root
    rootNode->addChild(action, prior, std::move(childNode));
    
    // Add virtual loss
    int virtualLoss = 3;
    childPtr->addVirtualLoss(virtualLoss);
    
    // Visit count should increase
    EXPECT_EQ(childPtr->visitCount.load(), virtualLoss);
    
    // Value sum should decrease
    EXPECT_FLOAT_EQ(childPtr->valueSum.load(), -static_cast<float>(virtualLoss));
    
    // Remove virtual loss
    childPtr->removeVirtualLoss(virtualLoss);
    
    // Should be back to initial values
    EXPECT_EQ(childPtr->visitCount.load(), 0);
    EXPECT_FLOAT_EQ(childPtr->valueSum.load(), 0.0f);
}

TEST_F(MCTSNodeTest, GetBestAction) {
    // Add multiple children with different visit counts
    for (int i = 0; i < 3; i++) {
        int action = i;
        float prior = 0.1f;
        
        // Clone the state
        auto childState = state->clone();
        childState->makeMove(action);
        
        // Create child node
        auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
        
        // Set different visit counts
        childNode->visitCount.store(i * 10);
        
        // Add child to root
        rootNode->addChild(action, prior, std::move(childNode));
    }
    
    // Best action should be the one with most visits (action=2)
    int bestAction = rootNode->getBestAction();
    EXPECT_EQ(bestAction, 2);
    
    // Best child should be the one with most visits
    MCTSNode* bestChild = rootNode->getBestChild();
    EXPECT_EQ(bestChild, rootNode->children[2].get());
}

TEST_F(MCTSNodeTest, GetVisitCountDistribution) {
    // Add multiple children with different visit counts
    for (int i = 0; i < 3; i++) {
        int action = i;
        float prior = 0.1f;
        
        // Clone the state
        auto childState = state->clone();
        childState->makeMove(action);
        
        // Create child node
        auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
        
        // Set different visit counts
        childNode->visitCount.store(i * 10 + 1); // +1 to avoid zero visits
        
        // Add child to root
        rootNode->addChild(action, prior, std::move(childNode));
    }
    
    // Get distribution with temperature=1.0
    std::vector<float> dist1 = rootNode->getVisitCountDistribution(1.0f);
    
    // Should sum to 1.0
    float sum = 0.0f;
    for (float p : dist1) {
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // Higher visit count should have higher probability
    EXPECT_GT(dist1[2], dist1[1]);
    EXPECT_GT(dist1[1], dist1[0]);
    
    // Get distribution with temperature=0.1 (more deterministic)
    std::vector<float> dist2 = rootNode->getVisitCountDistribution(0.1f);
    
    // Should sum to 1.0
    sum = 0.0f;
    for (float p : dist2) {
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // With lower temperature, probability should be more concentrated
    EXPECT_GT(dist2[2] / dist2[1], dist1[2] / dist1[1]);
}

TEST_F(MCTSNodeTest, GetActionIndex) {
    // Add a few children
    for (int i = 0; i < 3; i++) {
        int action = i;
        float prior = 0.1f;
        
        // Clone the state
        auto childState = state->clone();
        childState->makeMove(action);
        
        // Create child node
        auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
        
        // Add child to root
        rootNode->addChild(action, prior, std::move(childNode));
    }
    
    // Get index of action=1
    int idx = rootNode->getActionIndex(1);
    EXPECT_EQ(idx, 1);
    
    // Non-existent action should return -1
    EXPECT_EQ(rootNode->getActionIndex(99), -1);
}

TEST_F(MCTSNodeTest, GetChild) {
    // Add a few children
    for (int i = 0; i < 3; i++) {
        int action = i;
        float prior = 0.1f;
        
        // Clone the state
        auto childState = state->clone();
        childState->makeMove(action);
        
        // Create child node
        auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
        
        // Add child to root
        rootNode->addChild(action, prior, std::move(childNode));
    }
    
    // Get child at index 1
    MCTSNode* child = rootNode->getChild(1);
    EXPECT_NE(child, nullptr);
    EXPECT_EQ(child, rootNode->children[1].get());
    
    // Invalid index should return nullptr
    EXPECT_EQ(rootNode->getChild(99), nullptr);
}

TEST_F(MCTSNodeTest, GetBestActions) {
    // Add children with different visit counts
    // Actions 0 and 2 have the same (highest) visit count
    for (int i = 0; i < 3; i++) {
        int action = i;
        float prior = 0.1f;
        
        // Clone the state
        auto childState = state->clone();
        childState->makeMove(action);
        
        // Create child node
        auto childNode = std::make_unique<MCTSNode>(childState.get(), rootNode.get(), prior);
        
        // Set visit counts - make 0 and 2 have the same count
        childNode->visitCount.store(i == 1 ? 10 : 20);
        
        // Add child to root
        rootNode->addChild(action, prior, std::move(childNode));
    }
    
    // Get best actions (should be 0 and 2)
    std::vector<int> bestActions = rootNode->getBestActions();
    EXPECT_EQ(bestActions.size(), 2);
    
    // Sort for deterministic comparison
    std::sort(bestActions.begin(), bestActions.end());
    
    EXPECT_EQ(bestActions[0], 0);
    EXPECT_EQ(bestActions[1], 2);
}

} // namespace mcts
} // namespace alphazero