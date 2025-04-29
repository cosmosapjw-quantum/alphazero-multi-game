#include <gtest/gtest.h>
#include "alphazero/games/chess/chess_state.h"
#include "alphazero/games/chess/chess960.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"

namespace alphazero {
namespace integration {

class ChessIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a chess state
        chessState = std::make_unique<chess::ChessState>();
        
        // Create a neural network
        network = nn::NeuralNetwork::create("", core::GameType::CHESS);
        
        // Create a transposition table
        tt = std::make_unique<mcts::TranspositionTable>(1024, 16);
        
        // Create MCTS with 2 threads and 50 simulations
        mcts = std::make_unique<mcts::ParallelMCTS>(
            *chessState, network.get(), tt.get(), 2, 50);
    }
    
    std::unique_ptr<chess::ChessState> chessState;
    std::unique_ptr<nn::NeuralNetwork> network;
    std::unique_ptr<mcts::TranspositionTable> tt;
    std::unique_ptr<mcts::ParallelMCTS> mcts;
};

TEST_F(ChessIntegrationTest, OpeningMoves) {
    // Run search and make a few opening moves
    mcts->search();
    
    // Get action
    int action = mcts->selectAction();
    
    // Should be a valid action
    EXPECT_GE(action, 0);
    
    // Get the move
    chess::ChessMove move = chessState->actionToChessMove(action);
    
    // Make the move
    chessState->makeMove(move);
    
    // Update MCTS
    mcts->updateWithMove(action);
    
    // Should be black's turn
    EXPECT_EQ(chessState->getCurrentPlayer(), static_cast<int>(chess::PieceColor::BLACK));
    
    // Run search for black
    mcts->search();
    
    // Get action for black
    int blackAction = mcts->selectAction();
    
    // Should be valid
    EXPECT_GE(blackAction, 0);
    
    // Make the move
    chess::ChessMove blackMove = chessState->actionToChessMove(blackAction);
    chessState->makeMove(blackMove);
    
    // Update MCTS
    mcts->updateWithMove(blackAction);
    
    // Should be white's turn again
    EXPECT_EQ(chessState->getCurrentPlayer(), static_cast<int>(chess::PieceColor::WHITE));
}

TEST_F(ChessIntegrationTest, Check) {
    // Setup a position where white can check black
    chessState->setFromFEN("rnbqkbnr/ppp2ppp/8/3pp3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1");
    
    // Create new MCTS for this position
    mcts = std::make_unique<mcts::ParallelMCTS>(
        *chessState, network.get(), tt.get(), 2, 50);
    
    // Run search
    mcts->search();
    
    // Get action
    int action = mcts->selectAction();
    
    // Make the move
    chess::ChessMove move = chessState->actionToChessMove(action);
    chessState->makeMove(move);
    
    // Check if black is in check
    EXPECT_TRUE(chessState->isInCheck(chess::PieceColor::BLACK));
}

TEST_F(ChessIntegrationTest, Checkmate) {
    // Setup a position where white can checkmate black in one move (scholar's mate)
    chessState->setFromFEN("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1");
    
    // Create new MCTS for this position
    mcts = std::make_unique<mcts::ParallelMCTS>(
        *chessState, network.get(), tt.get(), 2, 100);
    
    // Run search with more simulations to find checkmate
    mcts->search();
    
    // Get action
    int action = mcts->selectAction();
    
    // Make the move
    chess::ChessMove move = chessState->actionToChessMove(action);
    chessState->makeMove(move);
    
    // Check if game is over
    EXPECT_TRUE(chessState->isTerminal());
    
    // Check result is white win
    EXPECT_EQ(chessState->getGameResult(), core::GameResult::WIN_PLAYER1);
}

TEST_F(ChessIntegrationTest, Chess960) {
    // Create a Chess960 position
    int positionNumber = 518;  // Standard chess is position 518
    std::string fen = chess::Chess960::getStartingFEN(positionNumber);
    
    // Create a state with this position
    auto chess960State = std::make_unique<chess::ChessState>(true, fen);
    
    // Create MCTS for this position
    auto chess960Mcts = std::make_unique<mcts::ParallelMCTS>(
        *chess960State, network.get(), tt.get(), 2, 50);
    
    // Run search
    chess960Mcts->search();
    
    // Get and make a move
    int action = chess960Mcts->selectAction();
    chess::ChessMove move = chess960State->actionToChessMove(action);
    chess960State->makeMove(move);
    
    // Update MCTS
    chess960Mcts->updateWithMove(action);
    
    // Run search again
    chess960Mcts->search();
    
    // Get and make another move
    int action2 = chess960Mcts->selectAction();
    chess::ChessMove move2 = chess960State->actionToChessMove(action2);
    chess960State->makeMove(move2);
    
    // Should not crash
    EXPECT_TRUE(true);
}

TEST_F(ChessIntegrationTest, CastlingMoves) {
    // Setup a position where castling is possible
    chessState->setFromFEN("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3");
    
    // Create new MCTS for this position
    mcts = std::make_unique<mcts::ParallelMCTS>(
        *chessState, network.get(), tt.get(), 2, 100);
    
    // Run search
    mcts->search();
    
    // Get action probabilities
    auto probs = mcts->getActionProbabilities();
    
    // Find castling move (E1 to G1)
    chess::ChessMove castlingMove{chess::E1, chess::G1};
    int castlingAction = chessState->chessMoveToAction(castlingMove);
    
    // Castling should have non-zero probability
    EXPECT_GT(probs[castlingAction], 0.0f);
}

TEST_F(ChessIntegrationTest, PawnPromotion) {
    // Setup a position where pawn promotion is possible
    chessState->setFromFEN("8/P7/8/8/8/8/8/k6K w - - 0 1");
    
    // Create new MCTS for this position
    mcts = std::make_unique<mcts::ParallelMCTS>(
        *chessState, network.get(), tt.get(), 2, 50);
    
    // Run search
    mcts->search();
    
    // Get action
    int action = mcts->selectAction();
    
    // Make the move
    chess::ChessMove move = chessState->actionToChessMove(action);
    chessState->makeMove(move);
    
    // Check if promoted piece is at A8
    chess::Piece piece = chessState->getPiece(chess::A8);
    
    // Should be a white piece
    EXPECT_EQ(piece.color, chess::PieceColor::WHITE);
    
    // Should be promoted (not a pawn)
    EXPECT_NE(piece.type, chess::PieceType::PAWN);
}

} // namespace integration
} // namespace alphazero