#include <gtest/gtest.h>
#include "alphazero/games/chess/chess_state.h"

namespace alphazero {
namespace chess {

class ChessStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create default chess state (standard starting position)
        state = std::make_unique<ChessState>();
    }
    
    std::unique_ptr<ChessState> state;
};

TEST_F(ChessStateTest, InitialState) {
    // Check board size
    EXPECT_EQ(state->getBoardSize(), 8);
    
    // Check initial player is WHITE
    EXPECT_EQ(state->getCurrentPlayer(), static_cast<int>(PieceColor::WHITE));
    
    // Check that the board has the correct number of pieces
    int pieceCount = 0;
    for (int square = 0; square < 64; square++) {
        if (!state->getPiece(square).is_empty()) {
            pieceCount++;
        }
    }
    EXPECT_EQ(pieceCount, 32);  // 16 pieces per player
    
    // Check that kings are in the right positions
    Piece whiteKing = state->getPiece(E1);
    EXPECT_EQ(whiteKing.type, PieceType::KING);
    EXPECT_EQ(whiteKing.color, PieceColor::WHITE);
    
    Piece blackKing = state->getPiece(E8);
    EXPECT_EQ(blackKing.type, PieceType::KING);
    EXPECT_EQ(blackKing.color, PieceColor::BLACK);
    
    // Check initial castling rights
    CastlingRights rights = state->getCastlingRights();
    EXPECT_TRUE(rights.white_kingside);
    EXPECT_TRUE(rights.white_queenside);
    EXPECT_TRUE(rights.black_kingside);
    EXPECT_TRUE(rights.black_queenside);
    
    // Check initial legal moves (20 in standard chess)
    auto legalMoves = state->getLegalMoves();
    EXPECT_EQ(legalMoves.size(), 20);
}

TEST_F(ChessStateTest, MakeMove) {
    // Test making a simple pawn move
    auto moves = state->generateLegalMoves();
    
    // Find e2-e4 move
    ChessMove e4Move{E2, E4};
    bool foundMove = false;
    for (const auto& move : moves) {
        if (move.from_square == E2 && move.to_square == E4) {
            e4Move = move;
            foundMove = true;
            break;
        }
    }
    EXPECT_TRUE(foundMove);
    
    // Make the move
    state->makeMove(e4Move);
    
    // Check position updates
    EXPECT_EQ(state->getCurrentPlayer(), static_cast<int>(PieceColor::BLACK));
    EXPECT_TRUE(state->getPiece(E2).is_empty());
    EXPECT_EQ(state->getPiece(E4).type, PieceType::PAWN);
    
    // Check en passant square is set
    EXPECT_EQ(state->getEnPassantSquare(), E3);
    
    // Check FEN is updated
    std::string fen = state->toFEN();
    EXPECT_EQ(fen, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
}

TEST_F(ChessStateTest, CastlingMoves) {
    // Setup a position where white can castle
    state->setFromFEN("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3");
    
    // Check castling is legal
    auto moves = state->generateLegalMoves();
    bool canCastleKingside = false;
    
    for (const auto& move : moves) {
        if (move.from_square == E1 && move.to_square == G1) {
            canCastleKingside = true;
            break;
        }
    }
    
    EXPECT_TRUE(canCastleKingside);
    
    // Perform castling
    state->makeMove(ChessMove{E1, G1});
    
    // Check king and rook have moved correctly
    EXPECT_EQ(state->getPiece(G1).type, PieceType::KING);
    EXPECT_EQ(state->getPiece(F1).type, PieceType::ROOK);
    EXPECT_TRUE(state->getPiece(E1).is_empty());
    EXPECT_TRUE(state->getPiece(H1).is_empty());
    
    // Check castling rights updated
    CastlingRights rights = state->getCastlingRights();
    EXPECT_FALSE(rights.white_kingside);
    EXPECT_FALSE(rights.white_queenside);
    EXPECT_TRUE(rights.black_kingside);
    EXPECT_TRUE(rights.black_queenside);
}

TEST_F(ChessStateTest, EnPassantCapture) {
    // Setup a position where en passant is possible
    state->setFromFEN("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    
    // Check en passant capture is legal
    auto moves = state->generateLegalMoves();
    bool canCaptureEnPassant = false;
    ChessMove enPassantMove{E5, F6};
    
    for (const auto& move : moves) {
        if (move.from_square == E5 && move.to_square == F6) {
            canCaptureEnPassant = true;
            enPassantMove = move;
            break;
        }
    }
    
    EXPECT_TRUE(canCaptureEnPassant);
    
    // Perform en passant capture
    state->makeMove(enPassantMove);
    
    // Check positions updated correctly
    EXPECT_EQ(state->getPiece(F6).type, PieceType::PAWN);
    EXPECT_EQ(state->getPiece(F6).color, PieceColor::WHITE);
    EXPECT_TRUE(state->getPiece(E5).is_empty());
    EXPECT_TRUE(state->getPiece(F5).is_empty());  // Captured pawn should be removed
}

TEST_F(ChessStateTest, PawnPromotion) {
    // Setup a position with a pawn about to promote
    state->setFromFEN("rnbqkbnr/pppppPpp/8/8/8/8/PPPPPP1P/RNBQKBNR w KQkq - 0 1");
    
    // Check promotion is legal and different options exist
    auto moves = state->generateLegalMoves();
    std::vector<ChessMove> promotionMoves;
    
    for (const auto& move : moves) {
        if (move.from_square == F7 && move.to_square == F8 && move.promotion_piece != PieceType::NONE) {
            promotionMoves.push_back(move);
        }
    }
    
    // Should have 4 promotion options (Q, R, B, N)
    EXPECT_EQ(promotionMoves.size(), 4);
    
    // Perform promotion to queen
    ChessMove queenPromotion;
    for (const auto& move : promotionMoves) {
        if (move.promotion_piece == PieceType::QUEEN) {
            queenPromotion = move;
            break;
        }
    }
    
    state->makeMove(queenPromotion);
    
    // Check pawn was replaced with queen
    EXPECT_EQ(state->getPiece(F8).type, PieceType::QUEEN);
    EXPECT_EQ(state->getPiece(F8).color, PieceColor::WHITE);
    EXPECT_TRUE(state->getPiece(F7).is_empty());
}

TEST_F(ChessStateTest, CheckAndCheckmate) {
    // Setup a checkmate position (Scholar's mate)
    state->setFromFEN("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3");
    
    // Check that black has no legal moves
    auto moves = state->generateLegalMoves();
    EXPECT_TRUE(moves.empty());
    
    // Check game is terminal
    EXPECT_TRUE(state->isTerminal());
    
    // Check result is white win
    EXPECT_EQ(state->getGameResult(), core::GameResult::WIN_PLAYER1);
    
    // Check that king is in check
    EXPECT_TRUE(state->isInCheck(PieceColor::BLACK));
}

TEST_F(ChessStateTest, DrawConditions) {
    // Test stalemate
    state->setFromFEN("8/8/8/8/8/6k1/5q2/7K w - - 0 1");
    
    // No legal moves in stalemate
    auto moves = state->generateLegalMoves();
    EXPECT_TRUE(moves.empty());
    
    // Game should be terminal
    EXPECT_TRUE(state->isTerminal());
    
    // Result should be draw
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
    
    // King is not in check in stalemate
    EXPECT_FALSE(state->isInCheck(PieceColor::WHITE));
    
    // Test insufficient material (K vs K)
    state->setFromFEN("8/8/8/8/8/6k1/8/7K w - - 0 1");
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
    
    // Test 50-move rule
    state->setFromFEN("8/8/8/8/8/6k1/8/7K w - - 100 1");
    EXPECT_TRUE(state->isTerminal());
    EXPECT_EQ(state->getGameResult(), core::GameResult::DRAW);
}

TEST_F(ChessStateTest, FENConversion) {
    // Test converting to and from FEN
    std::string startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    // Create state from FEN
    state->setFromFEN(startFEN);
    
    // Convert back to FEN
    std::string outputFEN = state->toFEN();
    
    // Should match original FEN
    EXPECT_EQ(outputFEN, startFEN);
    
    // Test custom position
    std::string customFEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
    state->setFromFEN(customFEN);
    EXPECT_EQ(state->toFEN(), customFEN);
}

TEST_F(ChessStateTest, MoveStringConversion) {
    // Test converting moves to and from algebraic notation
    
    // e2-e4 move
    ChessMove e4Move{E2, E4};
    std::string moveStr = state->moveToString(e4Move);
    
    // Should be in the format like "e2e4"
    EXPECT_EQ(moveStr, "e2e4");
    
    // Convert back to move
    auto parsedMove = state->stringToMove(moveStr);
    EXPECT_TRUE(parsedMove.has_value());
    EXPECT_EQ(parsedMove->from_square, E2);
    EXPECT_EQ(parsedMove->to_square, E4);
    
    // Test invalid conversions
    EXPECT_FALSE(state->stringToMove("z9z9").has_value());
}

TEST_F(ChessStateTest, Chess960Mode) {
    // Create a Chess960 position
    auto chess960 = std::make_unique<ChessState>(true);
    
    // Set a specific Chess960 position
    // This is position 518 (RNBQKBNR)
    chess960->setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    // Verify castling works in Chess960
    bool canCastleKingside = false;
    auto moves = chess960->generateLegalMoves();
    
    // Make some opening moves to enable castling
    chess960->makeMove(ChessMove{G1, F3});  // Nf3
    chess960->makeMove(ChessMove{G8, F6});  // Nf6
    chess960->makeMove(ChessMove{F1, E2});  // Be2
    chess960->makeMove(ChessMove{F8, E7});  // Be7
    chess960->makeMove(ChessMove{E1, G1});  // O-O
    
    // Check king and rook moved correctly for castling
    EXPECT_EQ(chess960->getPiece(G1).type, PieceType::KING);
    EXPECT_EQ(chess960->getPiece(F1).type, PieceType::ROOK);
}

TEST_F(ChessStateTest, UndoMove) {
    // Make a move
    state->makeMove(ChessMove{E2, E4});
    
    // Save state
    std::string fenAfterMove = state->toFEN();
    
    // Undo the move
    bool undoSuccess = state->undoMove();
    EXPECT_TRUE(undoSuccess);
    
    // Check state is back to initial position
    EXPECT_EQ(state->toFEN(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    // Make the move again and check it's the same as before
    state->makeMove(ChessMove{E2, E4});
    EXPECT_EQ(state->toFEN(), fenAfterMove);
}

TEST_F(ChessStateTest, CloneAndEquals) {
    // Make a move
    state->makeMove(ChessMove{E2, E4});
    
    // Clone the state
    auto clonedState = state->clone();
    auto* castClone = dynamic_cast<ChessState*>(clonedState.get());
    EXPECT_NE(castClone, nullptr);
    
    // Check states are equal
    EXPECT_TRUE(state->equals(*castClone));
    EXPECT_EQ(state->toFEN(), castClone->toFEN());
    
    // Make different moves on original and clone
    state->makeMove(ChessMove{G1, F3});
    castClone->makeMove(ChessMove{D2, D4});
    
    // Now they should be different
    EXPECT_FALSE(state->equals(*castClone));
    EXPECT_NE(state->toFEN(), castClone->toFEN());
}

} // namespace chess
} // namespace alphazero