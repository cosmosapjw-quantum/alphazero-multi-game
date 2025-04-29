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
    // Pass for BLACK
    state->makeMove(-1);
    
    // Check player has changed
    EXPECT_EQ(state->getCurrentPlayer(), 2);  // WHITE
    
    // Pass for WHITE
    state->makeMove(-1);
    
    // Game should be terminal after consecutive passes
    EXPECT_TRUE(state->isTerminal());
    
    // Draw by default (no stones played)
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
}

TEST_F(GoStateTest, CaptureStones) {
    // Create a position where BLACK captures WHITE stones
    
    // BLACK at (3,3)
    state->makeMove(state->coordToAction(3, 3));
    // WHITE at (4,3)
    state->makeMove(state->coordToAction(4, 3));
    // BLACK at (5,3)
    state->makeMove(state->coordToAction(5, 3));
    // WHITE at (4,2)
    state->makeMove(state->coordToAction(4, 2));
    // BLACK at (4,4)
    state->makeMove(state->coordToAction(4, 4));
    
    // Before capture, check position
    EXPECT_EQ(state->getStone(state->coordToAction(4, 3)), 2);  // WHITE stone
    EXPECT_EQ(state->getStone(state->coordToAction(4, 2)), 2);  // WHITE stone
    
    // WHITE captures nothing yet
    EXPECT_EQ(state->getCapturedStones(2), 0);
    
    // WHITE's move (not a capture)
    state->makeMove(state->coordToAction(2, 3));
    
    // BLACK completes the capture
    state->makeMove(state->coordToAction(4, 1));
    
    // Stones should be captured
    EXPECT_EQ(state->getStone(state->coordToAction(4, 3)), 0);  // Empty
    EXPECT_EQ(state->getStone(state->coordToAction(4, 2)), 0);  // Empty
    
    // BLACK should have captured 2 stones
    EXPECT_EQ(state->getCapturedStones(1), 2);
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
    // Create a ko position
    // BLACK at (3,3)
    state->makeMove(state->coordToAction(3, 3));
    // WHITE at (3,4)
    state->makeMove(state->coordToAction(3, 4));
    // BLACK at (4,4)
    state->makeMove(state->coordToAction(4, 4));
    // WHITE at (4,3)
    state->makeMove(state->coordToAction(4, 3));
    // BLACK at (5,3)
    state->makeMove(state->coordToAction(5, 3));
    // WHITE at (4,2)
    state->makeMove(state->coordToAction(4, 2));
    // BLACK at (3,2)
    state->makeMove(state->coordToAction(3, 2));
    
    // BLACK captures WHITE stone at (4,3)
    // This creates a ko situation
    state->makeMove(state->coordToAction(4, 3));
    
    // WHITE cannot immediately recapture at (4,3) due to ko rule
    EXPECT_FALSE(state->isLegalMove(state->coordToAction(4, 3)));
    
    // WHITE plays elsewhere
    state->makeMove(state->coordToAction(2, 2));
    
    // Now BLACK plays elsewhere
    state->makeMove(state->coordToAction(5, 5));
    
    // Now WHITE can recapture the ko
    EXPECT_TRUE(state->isLegalMove(state->coordToAction(4, 3)));
}

TEST_F(GoStateTest, Scoring) {
    // Place some stones to create territories
    
    // BLACK controls top-left
    state->makeMove(state->coordToAction(0, 3));
    state->makeMove(state->coordToAction(3, 0));
    state->makeMove(state->coordToAction(1, 2));
    state->makeMove(state->coordToAction(2, 1));
    
    // WHITE controls bottom-right
    state->makeMove(state->coordToAction(6, 8));
    state->makeMove(state->coordToAction(8, 6));
    state->makeMove(state->coordToAction(7, 7));
    state->makeMove(state->coordToAction(8, 8));
    
    // Pass to end the game
    state->makeMove(-1);
    state->makeMove(-1);
    
    // Game should be terminal
    EXPECT_TRUE(state->isTerminal());
    
    // Get territory ownership
    std::vector<int> territory = state->getTerritoryOwnership();
    
    // Check some territories
    EXPECT_EQ(territory[state->coordToAction(0, 0)], 1);  // BLACK territory
    EXPECT_EQ(territory[state->coordToAction(8, 8)], 2);  // WHITE territory
    
    // Check score (with komi)
    // WHITE should win with komi of 6.5
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER2);
}

TEST_F(GoStateTest, UndoMove) {
    // Make a move
    int center = state->coordToAction(4, 4);
    state->makeMove(center);
    
    // Undo the move
    bool undoResult = state->undoMove();
    EXPECT_TRUE(undoResult);
    
    // Check board is back to initial state
    EXPECT_EQ(state->getStone(center), 0);
    EXPECT_EQ(state->getCurrentPlayer(), 1);
    
    // Cannot undo from initial state
    EXPECT_FALSE(state->undoMove());
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