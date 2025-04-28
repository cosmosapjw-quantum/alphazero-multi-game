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
    grandchildPtr->valueSum.fetch_add(1.0f);
    
    // The visit and value should propagate up the tree
    EXPECT_EQ(grandchildPtr->visitCount.load(), 1);
    EXPECT_FLOAT_EQ(grandchildPtr->valueSum.load(), 1.0f);
    EXPECT_FLOAT_EQ(grandchildPtr->getValue(), 1.0f);
    
    // Update the parent
    childPtr->visitCount.fetch_add(1);
    childPtr->valueSum.fetch_add(-1.0f); // Value flips sign for parent
    
    EXPECT_EQ(childPtr->visitCount.load(), 1);
    EXPECT_FLOAT_EQ(childPtr->valueSum.load(), -1.0f);
    EXPECT_FLOAT_EQ(childPtr->getValue(), -1.0f);
    
    // Update the root
    rootNode->visitCount.fetch_add(1);
    rootNode->valueSum.fetch_add(1.0f); // Value flips sign again
    
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

} // namespace mcts
} // namespace alphazero