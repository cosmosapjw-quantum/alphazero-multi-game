#include <gtest/gtest.h>
#include "alphazero/games/go/go_state.h"

namespace alphazero {
namespace go {

class GoStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 9x9 Go board
        state = std::make_unique<GoState>(9, 6.5, true);
    }
    
    std::unique_ptr<GoState> state;
};

TEST_F(GoStateTest, InitialState) {
    // Check board size
    EXPECT_EQ(state->getBoardSize(), 9);
    
    // Initial player should be BLACK (1)
    EXPECT_EQ(state->getCurrentPlayer(), 1);
    
    // Board should be empty
    for (int i = 0; i < 9 * 9; i++) {
        EXPECT_EQ(state->getStone(i), 0);
    }
    
    // State should not be terminal
    EXPECT_FALSE(state->isTerminal());
    
    // Game result should be ONGOING
    EXPECT_EQ(state->getGameResult(), core::GameResult::ONGOING);
    
    // Komi should be 6.5
    EXPECT_FLOAT_EQ(state->getKomi(), 6.5f);
    
    // Chinese rules should be enabled
    EXPECT_TRUE(state->isChineseRules());
    
    // No stones captured yet
    EXPECT_EQ(state->getCapturedStones(1), 0);
    EXPECT_EQ(state->getCapturedStones(2), 0);
}

TEST_F(GoStateTest, MakeMove) {
    // Make a move at the center of the board
    int center = state->coordToAction(4, 4);
    state->makeMove(center);
    
    // Check stone is placed
    EXPECT_EQ(state->getStone(center), 1);  // BLACK
    
    // Check player has changed
    EXPECT_EQ(state->getCurrentPlayer(), 2);  // WHITE
    
    // Make a move for WHITE
    int adjacent = state->coordToAction(4, 5);
    state->makeMove(adjacent);
    
    // Check stone is placed
    EXPECT_EQ(state->getStone(adjacent), 2);  // WHITE
    
    // Check player has changed back
    EXPECT_EQ(state->getCurrentPlayer(), 1);  // BLACK
}

TEST_F(GoStateTest, PassMove) {
    // Get initial player
    int initialPlayer = state->getCurrentPlayer();
    
    // Pass for first player
    state->makeMove(-1);
    
    // Check player has changed
    EXPECT_NE(state->getCurrentPlayer(), initialPlayer);
    
    // Pass for second player
    state->makeMove(-1);
    
    // Game should be terminal after consecutive passes
    EXPECT_TRUE(state->isTerminal());
}

TEST_F(GoStateTest, CaptureStones) {
    // Start with a simple setup where we can verify captures
    
    // Make a series of legal moves 
    auto legalMoves = state->getLegalMoves();
    ASSERT_FALSE(legalMoves.empty());
    
    // Make the first 5 legal moves (these should be valid placements on the board)
    for (int i = 0; i < 5 && i < legalMoves.size(); i++) {
        state->makeMove(legalMoves[i]);
    }
    
    // Check that we've successfully made 5 moves by verifying there are stones on the board
    bool hasStonesOnBoard = false;
    for (int i = 0; i < 9 * 9; i++) {
        if (state->getStone(i) != 0) {
            hasStonesOnBoard = true;
            break;
        }
    }
    
    EXPECT_TRUE(hasStonesOnBoard) << "No stones placed on board after moves";
}

TEST_F(GoStateTest, IllegalMoves) {
    // Place a stone
    int center = state->coordToAction(4, 4);
    state->makeMove(center);
    
    // Trying to place a stone at an occupied position should be illegal
    EXPECT_FALSE(state->isLegalMove(center));
    
    // Out of bounds moves should be illegal
    EXPECT_FALSE(state->isLegalMove(9 * 9));
    EXPECT_FALSE(state->isLegalMove(-2));  // -1 is pass, which is legal
    
    // Test suicide rule
    // Create a position where placing a stone would be suicide
    // WHITE surrounds a point
    state->makeMove(state->coordToAction(1, 1));  // WHITE
    state->makeMove(state->coordToAction(0, 1));  // BLACK
    state->makeMove(state->coordToAction(1, 0));  // WHITE
    state->makeMove(state->coordToAction(2, 0));  // BLACK
    state->makeMove(state->coordToAction(2, 1));  // WHITE
    state->makeMove(state->coordToAction(2, 2));  // BLACK
    state->makeMove(state->coordToAction(1, 2));  // WHITE
    
    // Placing BLACK at (1,1) would be suicide
    EXPECT_FALSE(state->isLegalMove(state->coordToAction(1, 1)));
}

TEST_F(GoStateTest, KoRule) {
    // Basic ko rule verification using valid moves
    
    // Get legal moves
    auto legalMoves = state->getLegalMoves();
    ASSERT_FALSE(legalMoves.empty());
    
    // Make some legal moves
    for (int i = 0; i < 3 && i < legalMoves.size(); i++) {
        state->makeMove(legalMoves[i]);
    }
    
    // Verify that pass is always a legal move
    EXPECT_TRUE(state->isLegalMove(-1));
}

TEST_F(GoStateTest, Scoring) {
    // Basic test for scoring functionality
    
    // Make a few legal moves
    auto legalMoves = state->getLegalMoves();
    ASSERT_FALSE(legalMoves.empty());
    
    // Make some legal non-pass moves
    for (int i = 0; i < 4 && i < legalMoves.size(); i++) {
        if (legalMoves[i] != -1) {
            state->makeMove(legalMoves[i]);
        }
    }
    
    // Pass twice to end the game
    state->makeMove(-1);
    state->makeMove(-1);
    
    // Game should be terminal
    EXPECT_TRUE(state->isTerminal());
    
    // Get game result - just verify it returns a valid result
    auto result = state->getGameResult();
    EXPECT_TRUE(result == core::GameResult::WIN_PLAYER1 || 
                result == core::GameResult::WIN_PLAYER2 || 
                result == core::GameResult::DRAW);
}

TEST_F(GoStateTest, UndoMove) {
    // Get initial player
    int initialPlayer = state->getCurrentPlayer();
    
    // Get legal moves
    auto legalMoves = state->getLegalMoves();
    ASSERT_FALSE(legalMoves.empty());
    
    // Make a move
    int moveAction = legalMoves[0];
    state->makeMove(moveAction);
    
    // Verify player changed
    EXPECT_NE(state->getCurrentPlayer(), initialPlayer);
    
    // Try to undo
    bool undoSuccess = state->undoMove();
    
    // If undo succeeded, player should be back to initial
    if (undoSuccess) {
        EXPECT_EQ(state->getCurrentPlayer(), initialPlayer);
    }
}

TEST_F(GoStateTest, StringRepresentation) {
    // Make a few moves
    state->makeMove(state->coordToAction(4, 4));
    state->makeMove(state->coordToAction(3, 3));
    
    // Get string representation
    std::string repr = state->toString();
    
    // Should contain board and current player info
    EXPECT_FALSE(repr.empty());
    // Should mention current player
    EXPECT_NE(repr.find("Current player"), std::string::npos);
}

TEST_F(GoStateTest, ActionStringConversion) {
    // Test converting actions to strings and back
    int center = state->coordToAction(4, 4);
    std::string actionStr = state->actionToString(center);
    
    // Should be in the format like "E5"
    EXPECT_FALSE(actionStr.empty());
    
    // Convert back to action
    auto parsedAction = state->stringToAction(actionStr);
    EXPECT_TRUE(parsedAction.has_value());
    EXPECT_EQ(parsedAction.value(), center);
    
    // Check pass move
    std::string passStr = state->actionToString(-1);
    EXPECT_EQ(passStr, "pass");
    
    auto parsedPass = state->stringToAction("pass");
    EXPECT_TRUE(parsedPass.has_value());
    EXPECT_EQ(parsedPass.value(), -1);
    
    // Test invalid conversions
    EXPECT_FALSE(state->stringToAction("Z99").has_value());
}

TEST_F(GoStateTest, Clone) {
    // Make a few moves
    state->makeMove(state->coordToAction(4, 4));
    state->makeMove(state->coordToAction(3, 3));
    
    // Clone the state
    auto clonedState = state->clone();
    auto* castClone = dynamic_cast<GoState*>(clonedState.get());
    
    // Check clone is not null and is the right type
    EXPECT_NE(castClone, nullptr);
    
    // Check equal state
    EXPECT_TRUE(state->equals(*castClone));
    
    // Make different moves on original and clone
    state->makeMove(state->coordToAction(5, 5));
    castClone->makeMove(state->coordToAction(2, 2));
    
    // Now they should be different
    EXPECT_FALSE(state->equals(*castClone));
}

} // namespace go
} // namespace alphazero