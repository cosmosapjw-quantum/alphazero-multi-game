// include/alphazero/games/chess/chess_rules.h
#ifndef CHESS_RULES_H
#define CHESS_RULES_H

#include <vector>
#include <functional>
#include <optional>
#include <unordered_map>

namespace alphazero {
namespace chess {

// Forward declarations
struct Piece;
struct CastlingRights;
struct ChessMove;
enum class PieceType;
enum class PieceColor;

/**
 * @brief Rules implementation for Chess
 */
class ChessRules {
public:
    /**
     * @brief Constructor
     * 
     * @param chess960 Whether to use Chess960 rules
     */
    ChessRules(bool chess960 = false);
    
    /**
     * @brief Set board accessor functions
     * 
     * @param get_piece Function to get piece at square
     * @param is_valid_square Function to check if square is valid
     * @param get_king_square Function to find king's square
     */
    void setBoardAccessor(
        std::function<Piece(int)> get_piece,
        std::function<bool(int)> is_valid_square,
        std::function<int(PieceColor)> get_king_square);
        
    /**
     * @brief Generate all legal moves
     * 
     * @param current_player Current player
     * @param castling_rights Current castling rights
     * @param en_passant_square Current en passant square
     * @return Vector of legal ChessMove objects
     */
    std::vector<ChessMove> generateLegalMoves(
        PieceColor current_player,
        const CastlingRights& castling_rights,
        int en_passant_square) const;
    
    /**
     * @brief Generate pseudo-legal moves (not checking for check)
     * 
     * @param current_player Current player
     * @param castling_rights Current castling rights
     * @param en_passant_square Current en passant square
     * @return Vector of pseudo-legal ChessMove objects
     */
    std::vector<ChessMove> generatePseudoLegalMoves(
        PieceColor current_player,
        const CastlingRights& castling_rights,
        int en_passant_square) const;
    
    /**
     * @brief Check if a move is legal
     * 
     * @param move Move to check
     * @param current_player Current player
     * @param castling_rights Current castling rights
     * @param en_passant_square Current en passant square
     * @return true if legal, false otherwise
     */
    bool isLegalMove(
        const ChessMove& move,
        PieceColor current_player,
        const CastlingRights& castling_rights,
        int en_passant_square) const;
    
    /**
     * @brief Check if a position is in check
     * 
     * @param color Color to check for
     * @return true if in check, false otherwise
     */
    bool isInCheck(PieceColor color) const;
    
    /**
     * @brief Check if a square is attacked by a player
     * 
     * @param square Square index
     * @param by_color Color of the attacker
     * @return true if attacked, false otherwise
     */
    bool isSquareAttacked(int square, PieceColor by_color) const;
    
    /**
     * @brief Check for insufficient material (draw condition)
     * 
     * @return true if position has insufficient material for checkmate
     */
    bool hasInsufficientMaterial() const;
    
    /**
     * @brief Check for threefold repetition (requires position history)
     * 
     * @param position_history Vector of position hashes
     * @return true if threefold repetition occurred
     */
    bool isThreefoldRepetition(const std::vector<uint64_t>& position_history) const;
    
    /**
     * @brief Check for 50-move rule
     * 
     * @param halfmove_clock Current halfmove clock value
     * @return true if 50-move rule applies
     */
    bool isFiftyMoveRule(int halfmove_clock) const;
    
    /**
     * @brief Get updated castling rights after a move
     * 
     * @param move The executed move
     * @param piece The piece that moved
     * @param captured Captured piece (if any)
     * @param current_rights Current castling rights
     * @return Updated castling rights
     */
    CastlingRights getUpdatedCastlingRights(
        const ChessMove& move,
        const Piece& piece,
        const Piece& captured,
        const CastlingRights& current_rights) const;
        
private:
    bool chess960_;
    
    // Board accessor functions
    std::function<Piece(int)> get_piece_;
    std::function<bool(int)> is_valid_square_;
    std::function<int(PieceColor)> get_king_square_;
    
    // Move generation helpers
    void addPawnMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player, int en_passant_square) const;
    void addKnightMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addBishopMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addRookMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addQueenMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addKingMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const;
    void addCastlingMoves(std::vector<ChessMove>& moves, PieceColor current_player, const CastlingRights& castling_rights) const;
    
    // Sliding piece movement
    void addSlidingMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player, 
                         const std::vector<std::pair<int, int>>& directions) const;
    
    // Utility functions
    static int getRank(int square) { return square / 8; }
    static int getFile(int square) { return square % 8; }
    static int getSquare(int rank, int file) { return rank * 8 + file; }
    static PieceColor oppositeColor(PieceColor color);
};

} // namespace chess
} // namespace alphazero

#endif // CHESS_RULES_H