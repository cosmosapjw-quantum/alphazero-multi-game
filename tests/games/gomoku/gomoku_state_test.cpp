#include <gtest/gtest.h>
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace gomoku {

class GomokuStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default setup
        state = std::make_unique<GomokuState>(15, false);
    }
    
    std::unique_ptr<GomokuState> state;
};

TEST_F(GomokuStateTest, InitialState) {
    // Initial state should have no stones on the board
    EXPECT_EQ(state->count_total_stones(), 0);
    
    // First player should be BLACK (1)
    EXPECT_EQ(state->getCurrentPlayer(), BLACK);
    
    // Board size should be 15
    EXPECT_EQ(state->getBoardSize(), 15);
    
    // Action space should be 15*15 = 225
    EXPECT_EQ(state->getActionSpaceSize(), 225);
    
    // State should not be terminal
    EXPECT_FALSE(state->isTerminal());
    
    // Result should be ONGOING
    EXPECT_EQ(state->getGameResult(), core::GameResult::ONGOING);
}

TEST_F(GomokuStateTest, MakeMove) {
    // Make a move at the center
    int centerAction = 7 * 15 + 7;
    state->makeMove(centerAction);
    
    // Board should have 1 stone
    EXPECT_EQ(state->count_total_stones(), 1);
    
    // Player should be WHITE (2)
    EXPECT_EQ(state->getCurrentPlayer(), WHITE);
    
    // Center position should be occupied
    EXPECT_TRUE(state->is_occupied(centerAction));
    
    // Position should have BLACK's stone
    EXPECT_TRUE(state->is_bit_set(0, centerAction));  // BLACK index is 0
    EXPECT_FALSE(state->is_bit_set(1, centerAction)); // WHITE index is 1
}

TEST_F(GomokuStateTest, UndoMove) {
    // Make a move
    int action = 7 * 15 + 7;
    state->makeMove(action);
    
    // Undo the move
    bool undoResult = state->undoMove();
    
    // Undo should succeed
    EXPECT_TRUE(undoResult);
    
    // Board should be back to initial state
    EXPECT_EQ(state->count_total_stones(), 0);
    EXPECT_EQ(state->getCurrentPlayer(), BLACK);
    EXPECT_FALSE(state->is_occupied(action));
}

TEST_F(GomokuStateTest, CheckWin) {
    // Create a winning position for BLACK
    // ○ ○ ○ ○ ○ . . . . . . . . . .
    // . . . . . . . . . . . . . . .
    // ... (rest of the board empty)
    
    // Place five black stones in a row
    for (int i = 0; i < 5; i++) {
        int action = 0 * 15 + i; // First row, columns 0-4
        state->makeMove(action);
        
        if (i < 4) {
            // Place white stones in second row to avoid wins
            int whiteAction = 1 * 15 + i;
            state->makeMove(whiteAction);
        }
    }
    
    // After 9 moves (5 black, 4 white), BLACK should have won
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER1);
}

TEST_F(GomokuStateTest, IllegalMoves) {
    // Make a move
    int action = 7 * 15 + 7;
    state->makeMove(action);
    
    // Trying to place a stone at an occupied position should be illegal
    EXPECT_FALSE(state->isLegalMove(action));
    
    // Out of bounds moves should be illegal
    EXPECT_FALSE(state->isLegalMove(-1));
    EXPECT_FALSE(state->isLegalMove(15 * 15));
}

TEST_F(GomokuStateTest, GetLegalMoves) {
    // Initial state should have all positions as legal moves
    auto legalMoves = state->getLegalMoves();
    EXPECT_EQ(legalMoves.size(), 15 * 15);
    
    // After making a move, there should be one less legal move
    int action = 7 * 15 + 7;
    state->makeMove(action);
    
    legalMoves = state->getLegalMoves();
    EXPECT_EQ(legalMoves.size(), 15 * 15 - 1);
    
    // Check that the occupied position is not in legal moves
    auto it = std::find(legalMoves.begin(), legalMoves.end(), action);
    EXPECT_EQ(it, legalMoves.end());
}

TEST_F(GomokuStateTest, RenjuRules) {
    // Create a state with Renju rules
    std::unique_ptr<GomokuState> renjuState = std::make_unique<GomokuState>(15, true);
    
    // Basic test for Renju rule implementation
    
    // Check that initial state is properly configured
    EXPECT_TRUE(renjuState->isUsingRenjuRules());
    EXPECT_FALSE(state->isUsingRenjuRules());
    
    // Verify both rule sets allow valid moves
    int centerAction = 7 * 15 + 7;
    EXPECT_TRUE(renjuState->isLegalMove(centerAction));
    EXPECT_TRUE(state->isLegalMove(centerAction));
}

TEST_F(GomokuStateTest, Clone) {
    // Make a few moves
    state->makeMove(7 * 15 + 7);  // Center
    state->makeMove(8 * 15 + 8);  // Adjacent
    
    // Clone the state
    auto clonedState = state->clone();
    auto* castClone = dynamic_cast<GomokuState*>(clonedState.get());
    
    // Check clone is not null and is the right type
    EXPECT_NE(castClone, nullptr);
    
    // Check equal board state
    EXPECT_TRUE(state->board_equal(*castClone));
    
    // Check same player turn
    EXPECT_EQ(state->getCurrentPlayer(), castClone->getCurrentPlayer());
    
    // Make different moves on original and clone
    state->makeMove(9 * 15 + 9);
    castClone->makeMove(6 * 15 + 6);
    
    // Now they should be different
    EXPECT_FALSE(state->board_equal(*castClone));
}

TEST_F(GomokuStateTest, TestHash) {
    // Two identical boards should have the same hash
    auto state2 = std::make_unique<GomokuState>(15, false);
    EXPECT_EQ(state->getHash(), state2->getHash());
    
    // After making the same moves, hashes should still match
    state->makeMove(7 * 15 + 7);
    state2->makeMove(7 * 15 + 7);
    EXPECT_EQ(state->getHash(), state2->getHash());
    
    // Different moves should yield different hashes
    state->makeMove(8 * 15 + 8);
    state2->makeMove(8 * 15 + 7);
    EXPECT_NE(state->getHash(), state2->getHash());
}

TEST_F(GomokuStateTest, ActionStringConversion) {
    // Test converting actions to strings and back
    int action = 7 * 15 + 7;  // Center
    std::string actionStr = state->actionToString(action);
    
    // Should be in the format like "H8" (column, row)
    EXPECT_FALSE(actionStr.empty());
    
    // Convert back to action
    auto parsedAction = state->stringToAction(actionStr);
    EXPECT_TRUE(parsedAction.has_value());
    EXPECT_EQ(parsedAction.value(), action);
    
    // Test invalid conversions
    EXPECT_FALSE(state->stringToAction("Z99").has_value());
    EXPECT_EQ(state->actionToString(-1), "invalid");
}

TEST_F(GomokuStateTest, Equals) {
    auto state2 = std::make_unique<GomokuState>(15, false);
    
    // Initial states should be equal
    EXPECT_TRUE(state->equals(*state2));
    
    // After same moves, still equal
    state->makeMove(7 * 15 + 7);
    state2->makeMove(7 * 15 + 7);
    EXPECT_TRUE(state->equals(*state2));
    
    // Different moves -> not equal
    state->makeMove(8 * 15 + 8);
    state2->makeMove(8 * 15 + 7);
    EXPECT_FALSE(state->equals(*state2));
}

TEST_F(GomokuStateTest, CustomBoardSize) {
    // Create a smaller board
    auto smallState = std::make_unique<GomokuState>(9, false);
    
    // Check correct size
    EXPECT_EQ(smallState->getBoardSize(), 9);
    EXPECT_EQ(smallState->getActionSpaceSize(), 9 * 9);
    
    // Make sure center is calculated correctly
    int centerAction = 4 * 9 + 4;
    smallState->makeMove(centerAction);
    EXPECT_TRUE(smallState->is_occupied(centerAction));
}

} // namespace gomoku
} // namespace alphazero