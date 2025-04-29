#include <gtest/gtest.h>
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace core {

class IGameStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create game states for testing
        gomokuState = createGameState(GameType::GOMOKU);
        chessState = createGameState(GameType::CHESS);
        goState = createGameState(GameType::GO);
    }
    
    std::unique_ptr<IGameState> gomokuState;
    std::unique_ptr<IGameState> chessState;
    std::unique_ptr<IGameState> goState;
};

TEST_F(IGameStateTest, CreateGameState) {
    // Test that all game states were created successfully
    EXPECT_NE(gomokuState, nullptr);
    EXPECT_NE(chessState, nullptr);
    EXPECT_NE(goState, nullptr);
    
    // Check correct game types
    EXPECT_EQ(gomokuState->getGameType(), GameType::GOMOKU);
    EXPECT_EQ(chessState->getGameType(), GameType::CHESS);
    EXPECT_EQ(goState->getGameType(), GameType::GO);
}

TEST_F(IGameStateTest, InitialState) {
    // Test initial game state properties
    
    // All games should start in non-terminal state
    EXPECT_FALSE(gomokuState->isTerminal());
    EXPECT_FALSE(chessState->isTerminal());
    EXPECT_FALSE(goState->isTerminal());
    
    // All games should have ONGOING result at start
    EXPECT_EQ(gomokuState->getGameResult(), GameResult::ONGOING);
    EXPECT_EQ(chessState->getGameResult(), GameResult::ONGOING);
    EXPECT_EQ(goState->getGameResult(), GameResult::ONGOING);
    
    // Check current player (all should start with player 1)
    EXPECT_EQ(gomokuState->getCurrentPlayer(), 1);
    EXPECT_EQ(chessState->getCurrentPlayer(), 1);
    EXPECT_EQ(goState->getCurrentPlayer(), 1);
}

TEST_F(IGameStateTest, GetLegalMoves) {
    // Test that legal moves are returned for each game
    
    auto gomokuMoves = gomokuState->getLegalMoves();
    auto chessMoves = chessState->getLegalMoves();
    auto goMoves = goState->getLegalMoves();
    
    // Initial position should have legal moves
    EXPECT_FALSE(gomokuMoves.empty());
    EXPECT_FALSE(chessMoves.empty());
    EXPECT_FALSE(goMoves.empty());
    
    // Check specific move counts
    // Gomoku should have board_size^2 moves
    EXPECT_EQ(gomokuMoves.size(), gomokuState->getBoardSize() * gomokuState->getBoardSize());
    
    // Chess should have 20 initial moves (8 pawns with 2 options each + 2 knights with 2 options each)
    EXPECT_EQ(chessMoves.size(), 20);
    
    // Go should have board_size^2 + 1 moves (including pass)
    EXPECT_EQ(goMoves.size(), goState->getBoardSize() * goState->getBoardSize() + 1);
}

TEST_F(IGameStateTest, MakeMove) {
    // Test making a move changes the state
    
    // Get initial state properties
    int initialGomokuPlayer = gomokuState->getCurrentPlayer();
    int initialChessPlayer = chessState->getCurrentPlayer();
    int initialGoPlayer = goState->getCurrentPlayer();
    
    // Make a move for each game
    auto gomokuMoves = gomokuState->getLegalMoves();
    auto chessMoves = chessState->getLegalMoves();
    auto goMoves = goState->getLegalMoves();
    
    if (!gomokuMoves.empty()) {
        gomokuState->makeMove(gomokuMoves[0]);
    }
    
    if (!chessMoves.empty()) {
        chessState->makeMove(chessMoves[0]);
    }
    
    if (!goMoves.empty()) {
        goState->makeMove(goMoves[0]);
    }
    
    // Check player has changed
    EXPECT_NE(gomokuState->getCurrentPlayer(), initialGomokuPlayer);
    EXPECT_NE(chessState->getCurrentPlayer(), initialChessPlayer);
    EXPECT_NE(goState->getCurrentPlayer(), initialGoPlayer);
    
    // Check move history contains the move
    EXPECT_EQ(gomokuState->getMoveHistory().size(), 1);
    EXPECT_EQ(chessState->getMoveHistory().size(), 1);
    EXPECT_EQ(goState->getMoveHistory().size(), 1);
}

TEST_F(IGameStateTest, UndoMove) {
    // Test undoing a move reverts the state
    
    // Make a move for each game
    auto gomokuMoves = gomokuState->getLegalMoves();
    auto chessMoves = chessState->getLegalMoves();
    auto goMoves = goState->getLegalMoves();
    
    if (!gomokuMoves.empty()) {
        gomokuState->makeMove(gomokuMoves[0]);
    }
    
    if (!chessMoves.empty()) {
        chessState->makeMove(chessMoves[0]);
    }
    
    if (!goMoves.empty()) {
        goState->makeMove(goMoves[0]);
    }
    
    // Save state after move
    int gomokuPlayerAfterMove = gomokuState->getCurrentPlayer();
    int chessPlayerAfterMove = chessState->getCurrentPlayer();
    int goPlayerAfterMove = goState->getCurrentPlayer();
    
    // Undo move
    bool gomokuUndone = gomokuState->undoMove();
    bool chessUndone = chessState->undoMove();
    bool goUndone = goState->undoMove();
    
    // Check undo was successful
    EXPECT_TRUE(gomokuUndone);
    EXPECT_TRUE(chessUndone);
    EXPECT_TRUE(goUndone);
    
    // Check player has been reverted
    EXPECT_NE(gomokuState->getCurrentPlayer(), gomokuPlayerAfterMove);
    EXPECT_NE(chessState->getCurrentPlayer(), chessPlayerAfterMove);
    EXPECT_NE(goState->getCurrentPlayer(), goPlayerAfterMove);
    
    // Check move history is empty
    EXPECT_TRUE(gomokuState->getMoveHistory().empty());
    EXPECT_TRUE(chessState->getMoveHistory().empty());
    EXPECT_TRUE(goState->getMoveHistory().empty());
}

TEST_F(IGameStateTest, CloneState) {
    // Test cloning creates an equal but separate state
    
    auto gomokuClone = gomokuState->clone();
    auto chessClone = chessState->clone();
    auto goClone = goState->clone();
    
    // Check clones are not null
    EXPECT_NE(gomokuClone, nullptr);
    EXPECT_NE(chessClone, nullptr);
    EXPECT_NE(goClone, nullptr);
    
    // Check equality
    EXPECT_TRUE(gomokuState->equals(*gomokuClone));
    EXPECT_TRUE(chessState->equals(*chessClone));
    EXPECT_TRUE(goState->equals(*goClone));
    
    // Modify clone and check they are different
    auto gomokuMoves = gomokuClone->getLegalMoves();
    auto chessMoves = chessClone->getLegalMoves();
    auto goMoves = goClone->getLegalMoves();
    
    if (!gomokuMoves.empty()) {
        gomokuClone->makeMove(gomokuMoves[0]);
    }
    
    if (!chessMoves.empty()) {
        chessClone->makeMove(chessMoves[0]);
    }
    
    if (!goMoves.empty()) {
        goClone->makeMove(goMoves[0]);
    }
    
    // Check clones are now different from originals
    EXPECT_FALSE(gomokuState->equals(*gomokuClone));
    EXPECT_FALSE(chessState->equals(*chessClone));
    EXPECT_FALSE(goState->equals(*goClone));
}

TEST_F(IGameStateTest, TensorRepresentation) {
    // Test tensor representation for neural networks
    
    auto gomokuTensor = gomokuState->getTensorRepresentation();
    auto chessTensor = chessState->getTensorRepresentation();
    auto goTensor = goState->getTensorRepresentation();
    
    // Check dimensions
    EXPECT_GT(gomokuTensor.size(), 0);
    EXPECT_GT(gomokuTensor[0].size(), 0);
    EXPECT_GT(gomokuTensor[0][0].size(), 0);
    
    EXPECT_GT(chessTensor.size(), 0);
    EXPECT_GT(chessTensor[0].size(), 0);
    EXPECT_GT(chessTensor[0][0].size(), 0);
    
    EXPECT_GT(goTensor.size(), 0);
    EXPECT_GT(goTensor[0].size(), 0);
    EXPECT_GT(goTensor[0][0].size(), 0);
    
    // Enhanced representation should have more planes
    auto gomokuEnhanced = gomokuState->getEnhancedTensorRepresentation();
    auto chessEnhanced = chessState->getEnhancedTensorRepresentation();
    auto goEnhanced = goState->getEnhancedTensorRepresentation();
    
    EXPECT_GT(gomokuEnhanced.size(), gomokuTensor.size());
    EXPECT_GT(chessEnhanced.size(), chessTensor.size());
    EXPECT_GT(goEnhanced.size(), goTensor.size());
}

} // namespace core
} // namespace alphazero