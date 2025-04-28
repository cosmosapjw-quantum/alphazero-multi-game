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

} // namespace gomoku
} // namespace alphazero