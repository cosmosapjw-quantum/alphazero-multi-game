// src/games/chess/chess_rules.cpp
#include "alphazero/games/chess/chess_rules.h"
#include "alphazero/games/chess/chess_state.h"
#include <algorithm>
#include <unordered_map>

namespace alphazero {
namespace chess {

// Helper constant arrays for knight and king moves
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}
};

const std::vector<std::pair<int, int>> KING_MOVES = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}
};

// Sliding piece directions (bishop, rook, queen)
const std::vector<std::pair<int, int>> BISHOP_DIRECTIONS = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
};

const std::vector<std::pair<int, int>> ROOK_DIRECTIONS = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

const std::vector<std::pair<int, int>> QUEEN_DIRECTIONS = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}
};

// Constants for board representation
const int A1 = 56;
const int H1 = 63;
const int A8 = 0;
const int H8 = 7;
const int E1 = 60;
const int E8 = 4;

ChessRules::ChessRules(bool chess960) 
    : chess960_(chess960) {
    
    // Default implementations (will be replaced by setBoardAccessor)
    get_piece_ = [](int) { return Piece(); };
    is_valid_square_ = [](int) { return false; };
    get_king_square_ = [](PieceColor) { return -1; };
}

void ChessRules::setBoardAccessor(
    std::function<Piece(int)> get_piece,
    std::function<bool(int)> is_valid_square,
    std::function<int(PieceColor)> get_king_square) {
    
    get_piece_ = get_piece;
    is_valid_square_ = is_valid_square;
    get_king_square_ = get_king_square;
}

std::vector<ChessMove> ChessRules::generateLegalMoves(
    PieceColor current_player,
    const CastlingRights& castling_rights,
    int en_passant_square) const {
    
    std::vector<ChessMove> pseudoLegalMoves = generatePseudoLegalMoves(
        current_player, castling_rights, en_passant_square);
    std::vector<ChessMove> legalMoves;
    
    // Filter out moves that leave the king in check
    for (const ChessMove& move : pseudoLegalMoves) {
        // Create a temporary copy of the board to simulate the move
        // This is a simplified check - in a real implementation, we would need
        // to actually apply the move to a copy of the ChessState
        
        // Create a map for efficient lookup of pieces
        std::unordered_map<int, Piece> board_copy;
        for (int square = 0; square < 64; ++square) {
            Piece piece = get_piece_(square);
            if (!piece.is_empty()) {
                board_copy[square] = piece;
            }
        }
        
        // Apply the move to our temporary board
        Piece moving_piece = board_copy[move.from_square];
        board_copy.erase(move.from_square);
        
        // Handle captures
        board_copy.erase(move.to_square);
        
        // Handle en passant captures
        if (moving_piece.type == PieceType::PAWN && move.to_square == en_passant_square) {
            int captured_pawn_square = getSquare(getRank(move.from_square), getFile(move.to_square));
            board_copy.erase(captured_pawn_square);
        }
        
        // Handle promotion
        if (move.promotion_piece != PieceType::NONE) {
            moving_piece.type = move.promotion_piece;
        }
        
        // Place the piece at the destination
        board_copy[move.to_square] = moving_piece;
        
        // Handle castling (move the rook too)
        if (moving_piece.type == PieceType::KING && 
            std::abs(getFile(move.from_square) - getFile(move.to_square)) == 2) {
            
            int rank = getRank(move.from_square);
            bool isKingside = getFile(move.to_square) > getFile(move.from_square);
            
            if (isKingside) {
                // Kingside castling
                int rookFrom = getSquare(rank, 7);
                int rookTo = getSquare(rank, 5);
                Piece rook = board_copy[rookFrom];
                board_copy.erase(rookFrom);
                board_copy[rookTo] = rook;
            } else {
                // Queenside castling
                int rookFrom = getSquare(rank, 0);
                int rookTo = getSquare(rank, 3);
                Piece rook = board_copy[rookFrom];
                board_copy.erase(rookFrom);
                board_copy[rookTo] = rook;
            }
        }
        
        // Override our accessor to use the temporary board
        auto temp_get_piece = [&board_copy](int square) -> Piece {
            auto it = board_copy.find(square);
            if (it != board_copy.end()) {
                return it->second;
            }
            return Piece();
        };
        
        auto temp_get_king_square = [&board_copy, current_player]() -> int {
            for (const auto& [square, piece] : board_copy) {
                if (piece.type == PieceType::KING && piece.color == current_player) {
                    return square;
                }
            }
            return -1;  // Should not happen
        };
        
        // Store original accessors
        auto original_get_piece = get_piece_;
        auto original_get_king_square = get_king_square_;
        
        // Replace with temporary accessors
        const_cast<ChessRules*>(this)->get_piece_ = temp_get_piece;
        const_cast<ChessRules*>(this)->get_king_square_ = 
            [temp_get_king_square, current_player](PieceColor color) -> int {
                if (color == current_player) {
                    return temp_get_king_square();
                }
                return -1;  // Not needed for this check
            };
        
        // Check if the king is in check after the move
        bool king_in_check = isInCheck(current_player);
        
        // Restore original accessors
        const_cast<ChessRules*>(this)->get_piece_ = original_get_piece;
        const_cast<ChessRules*>(this)->get_king_square_ = original_get_king_square;
        
        // If the king is not in check after the move, it's legal
        if (!king_in_check) {
            legalMoves.push_back(move);
        }
    }
    
    return legalMoves;
}

std::vector<ChessMove> ChessRules::generatePseudoLegalMoves(
    PieceColor current_player,
    const CastlingRights& castling_rights,
    int en_passant_square) const {
    
    std::vector<ChessMove> moves;
    
    // Generate moves for each piece of the current player
    for (int square = 0; square < 64; ++square) {
        if (!is_valid_square_(square)) continue;
        
        Piece piece = get_piece_(square);
        
        if (piece.color == current_player) {
            switch (piece.type) {
                case PieceType::PAWN:
                    addPawnMoves(moves, square, current_player, en_passant_square);
                    break;
                case PieceType::KNIGHT:
                    addKnightMoves(moves, square, current_player);
                    break;
                case PieceType::BISHOP:
                    addBishopMoves(moves, square, current_player);
                    break;
                case PieceType::ROOK:
                    addRookMoves(moves, square, current_player);
                    break;
                case PieceType::QUEEN:
                    addQueenMoves(moves, square, current_player);
                    break;
                case PieceType::KING:
                    addKingMoves(moves, square, current_player);
                    break;
                default:
                    break;
            }
        }
    }
    
    // Add castling moves
    addCastlingMoves(moves, current_player, castling_rights);
    
    return moves;
}

bool ChessRules::isLegalMove(
    const ChessMove& move,
    PieceColor current_player,
    const CastlingRights& castling_rights,
    int en_passant_square) const {
    
    // Check if the move is in the list of legal moves
    const std::vector<ChessMove>& legalMoves = generateLegalMoves(
        current_player, castling_rights, en_passant_square);
    
    return std::find(legalMoves.begin(), legalMoves.end(), move) != legalMoves.end();
}

bool ChessRules::isInCheck(PieceColor color) const {
    // Find the king
    int kingSquare = get_king_square_(color);
    if (kingSquare == -1) {
        return false;  // No king found
    }
    
    // Check if the king is attacked
    return isSquareAttacked(kingSquare, oppositeColor(color));
}

bool ChessRules::isSquareAttacked(int square, PieceColor by_color) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    // Check for pawn attacks
    int pawnDir = (by_color == PieceColor::WHITE) ? -1 : 1;
    for (int fileOffset : {-1, 1}) {
        int attackRank = rank + pawnDir;
        int attackFile = file + fileOffset;
        
        if (attackRank >= 0 && attackRank < 8 && attackFile >= 0 && attackFile < 8) {
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = get_piece_(attackSquare);
            
            if (attacker.type == PieceType::PAWN && attacker.color == by_color) {
                return true;
            }
        }
    }
    
    // Check for knight attacks
    for (const auto& [rankOffset, fileOffset] : KNIGHT_MOVES) {
        int attackRank = rank + rankOffset;
        int attackFile = file + fileOffset;
        
        if (attackRank >= 0 && attackRank < 8 && attackFile >= 0 && attackFile < 8) {
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = get_piece_(attackSquare);
            
            if (attacker.type == PieceType::KNIGHT && attacker.color == by_color) {
                return true;
            }
        }
    }
    
    // Check for king attacks
    for (const auto& [rankOffset, fileOffset] : KING_MOVES) {
        int attackRank = rank + rankOffset;
        int attackFile = file + fileOffset;
        
        if (attackRank >= 0 && attackRank < 8 && attackFile >= 0 && attackFile < 8) {
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = get_piece_(attackSquare);
            
            if (attacker.type == PieceType::KING && attacker.color == by_color) {
                return true;
            }
        }
    }
    
    // Check for sliding piece attacks (bishop, rook, queen)
    
    // Bishop/Queen: diagonal directions
    for (const auto& [rankDir, fileDir] : BISHOP_DIRECTIONS) {
        for (int step = 1; ; ++step) {
            int attackRank = rank + rankDir * step;
            int attackFile = file + fileDir * step;
            
            if (attackRank < 0 || attackRank >= 8 || attackFile < 0 || attackFile >= 8) {
                break;  // Off the board
            }
            
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = get_piece_(attackSquare);
            
            if (!attacker.is_empty()) {
                if (attacker.color == by_color && 
                    (attacker.type == PieceType::BISHOP || attacker.type == PieceType::QUEEN)) {
                    return true;
                }
                break;  // Piece blocks further attacks in this direction
            }
        }
    }
    
    // Rook/Queen: straight directions
    for (const auto& [rankDir, fileDir] : ROOK_DIRECTIONS) {
        for (int step = 1; ; ++step) {
            int attackRank = rank + rankDir * step;
            int attackFile = file + fileDir * step;
            
            if (attackRank < 0 || attackRank >= 8 || attackFile < 0 || attackFile >= 8) {
                break;  // Off the board
            }
            
            int attackSquare = getSquare(attackRank, attackFile);
            Piece attacker = get_piece_(attackSquare);
            
            if (!attacker.is_empty()) {
                if (attacker.color == by_color && 
                    (attacker.type == PieceType::ROOK || attacker.type == PieceType::QUEEN)) {
                    return true;
                }
                break;  // Piece blocks further attacks in this direction
            }
        }
    }
    
    return false;
}

bool ChessRules::hasInsufficientMaterial() const {
    // Count material on the board
    int numWhitePawns = 0;
    int numBlackPawns = 0;
    int numWhiteKnights = 0;
    int numBlackKnights = 0;
    int numWhiteBishops = 0;
    int numBlackBishops = 0;
    int numWhiteRooks = 0;
    int numBlackRooks = 0;
    int numWhiteQueens = 0;
    int numBlackQueens = 0;
    
    for (int square = 0; square < 64; ++square) {
        if (!is_valid_square_(square)) continue;
        
        Piece piece = get_piece_(square);
        
        if (piece.is_empty()) continue;
        
        if (piece.color == PieceColor::WHITE) {
            switch (piece.type) {
                case PieceType::PAWN:   numWhitePawns++; break;
                case PieceType::KNIGHT: numWhiteKnights++; break;
                case PieceType::BISHOP: numWhiteBishops++; break;
                case PieceType::ROOK:   numWhiteRooks++; break;
                case PieceType::QUEEN:  numWhiteQueens++; break;
                default: break;
            }
        } else {
            switch (piece.type) {
                case PieceType::PAWN:   numBlackPawns++; break;
                case PieceType::KNIGHT: numBlackKnights++; break;
                case PieceType::BISHOP: numBlackBishops++; break;
                case PieceType::ROOK:   numBlackRooks++; break;
                case PieceType::QUEEN:  numBlackQueens++; break;
                default: break;
            }
        }
    }
    
    // Check for insufficient material scenarios
    
    // King vs King
    if (numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteKnights == 0 && numBlackKnights == 0 &&
        numWhiteBishops == 0 && numBlackBishops == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King and Knight vs King
    if ((numWhiteKnights == 1 && numBlackKnights == 0 ||
         numWhiteKnights == 0 && numBlackKnights == 1) &&
        numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteBishops == 0 && numBlackBishops == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King and Bishop vs King
    if ((numWhiteBishops == 1 && numBlackBishops == 0 ||
         numWhiteBishops == 0 && numBlackBishops == 1) &&
        numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteKnights == 0 && numBlackKnights == 0 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        return true;
    }
    
    // King and Bishop vs King and Bishop of the same color
    if (numWhitePawns == 0 && numBlackPawns == 0 &&
        numWhiteKnights == 0 && numBlackKnights == 0 &&
        numWhiteBishops == 1 && numBlackBishops == 1 &&
        numWhiteRooks == 0 && numBlackRooks == 0 &&
        numWhiteQueens == 0 && numBlackQueens == 0) {
        
        // Check if bishops are on the same color
        bool whiteBishopOnWhite = false;
        bool blackBishopOnWhite = false;
        
        for (int square = 0; square < 64; ++square) {
            if (!is_valid_square_(square)) continue;
            
            Piece piece = get_piece_(square);
            
            if (piece.type == PieceType::BISHOP) {
                int rank = getRank(square);
                int file = getFile(square);
                bool squareIsWhite = (rank + file) % 2 == 0;
                
                if (piece.color == PieceColor::WHITE) {
                    whiteBishopOnWhite = squareIsWhite;
                } else {
                    blackBishopOnWhite = squareIsWhite;
                }
            }
        }
        
        if (whiteBishopOnWhite == blackBishopOnWhite) {
            return true;
        }
    }
    
    return false;
}

bool ChessRules::isThreefoldRepetition(const std::vector<uint64_t>& position_history) const {
    if (position_history.empty()) {
        return false;
    }
    
    // Count current position
    uint64_t current_hash = position_history.back();
    int count = 0;
    
    for (uint64_t hash : position_history) {
        if (hash == current_hash) {
            count++;
        }
    }
    
    return count >= 3;
}

bool ChessRules::isFiftyMoveRule(int halfmove_clock) const {
    return halfmove_clock >= 100;  // 50 moves = 100 half-moves
}

CastlingRights ChessRules::getUpdatedCastlingRights(
    const ChessMove& move,
    const Piece& piece,
    const Piece& captured,
    const CastlingRights& current_rights) const {
    
    CastlingRights updated_rights = current_rights;
    
    // Update based on king movement
    if (piece.type == PieceType::KING) {
        if (piece.color == PieceColor::WHITE) {
            updated_rights.white_kingside = false;
            updated_rights.white_queenside = false;
        } else {
            updated_rights.black_kingside = false;
            updated_rights.black_queenside = false;
        }
    }
    
    // Update based on rook movement
    if (piece.type == PieceType::ROOK) {
        int file = getFile(move.from_square);
        int rank = getRank(move.from_square);
        
        if (piece.color == PieceColor::WHITE && rank == 7) {
            if (file == 0) {
                updated_rights.white_queenside = false;
            } else if (file == 7) {
                updated_rights.white_kingside = false;
            }
        } else if (piece.color == PieceColor::BLACK && rank == 0) {
            if (file == 0) {
                updated_rights.black_queenside = false;
            } else if (file == 7) {
                updated_rights.black_kingside = false;
            }
        }
    }
    
    // Update based on rook capture
    if (!captured.is_empty() && captured.type == PieceType::ROOK) {
        int file = getFile(move.to_square);
        int rank = getRank(move.to_square);
        
        if (captured.color == PieceColor::WHITE && rank == 7) {
            if (file == 0) {
                updated_rights.white_queenside = false;
            } else if (file == 7) {
                updated_rights.white_kingside = false;
            }
        } else if (captured.color == PieceColor::BLACK && rank == 0) {
            if (file == 0) {
                updated_rights.black_queenside = false;
            } else if (file == 7) {
                updated_rights.black_kingside = false;
            }
        }
    }
    
    return updated_rights;
}

void ChessRules::addPawnMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player, int en_passant_square) const {
    int rank = getRank(square);
    int file = getFile(square);
    int direction = (current_player == PieceColor::WHITE) ? -1 : 1;
    
    // Regular move forward
    int newRank = rank + direction;
    if (newRank >= 0 && newRank < 8) {
        int newSquare = getSquare(newRank, file);
        if (get_piece_(newSquare).is_empty()) {
            // Check if pawn is on the last rank (promotion)
            if (newRank == 0 || newRank == 7) {
                // Add all promotion options
                moves.push_back({square, newSquare, PieceType::QUEEN});
                moves.push_back({square, newSquare, PieceType::ROOK});
                moves.push_back({square, newSquare, PieceType::BISHOP});
                moves.push_back({square, newSquare, PieceType::KNIGHT});
            } else {
                moves.push_back({square, newSquare});
            }
            
            // Initial two-square move
            if ((current_player == PieceColor::WHITE && rank == 6) ||
                (current_player == PieceColor::BLACK && rank == 1)) {
                int twoSquaresForward = newRank + direction;
                int twoSquareNewSquare = getSquare(twoSquaresForward, file);
                if (get_piece_(twoSquareNewSquare).is_empty()) {
                    moves.push_back({square, twoSquareNewSquare});
                }
            }
        }
    }
    
    // Captures (including en passant)
    for (int fileOffset : {-1, 1}) {
        int newFile = file + fileOffset;
        if (newFile >= 0 && newFile < 8) {
            int newSquare = getSquare(newRank, newFile);
            
            // Regular capture
            if (is_valid_square_(newSquare)) {
                Piece targetPiece = get_piece_(newSquare);
                if (!targetPiece.is_empty() && targetPiece.color != current_player) {
                    // Check for promotion
                    if (newRank == 0 || newRank == 7) {
                        moves.push_back({square, newSquare, PieceType::QUEEN});
                        moves.push_back({square, newSquare, PieceType::ROOK});
                        moves.push_back({square, newSquare, PieceType::BISHOP});
                        moves.push_back({square, newSquare, PieceType::KNIGHT});
                    } else {
                        moves.push_back({square, newSquare});
                    }
                }
            }
            
            // En passant capture
            if (en_passant_square == newSquare) {
                moves.push_back({square, newSquare});
            }
        }
    }
}

void ChessRules::addKnightMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    for (const auto& [rankOffset, fileOffset] : KNIGHT_MOVES) {
        int newRank = rank + rankOffset;
        int newFile = file + fileOffset;
        
        if (newRank >= 0 && newRank < 8 && newFile >= 0 && newFile < 8) {
            int newSquare = getSquare(newRank, newFile);
            Piece targetPiece = get_piece_(newSquare);
            
            if (targetPiece.is_empty() || targetPiece.color != current_player) {
                moves.push_back({square, newSquare});
            }
        }
    }
}

void ChessRules::addBishopMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    addSlidingMoves(moves, square, current_player, BISHOP_DIRECTIONS);
}

void ChessRules::addRookMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    addSlidingMoves(moves, square, current_player, ROOK_DIRECTIONS);
}

void ChessRules::addQueenMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    addSlidingMoves(moves, square, current_player, QUEEN_DIRECTIONS);
}

void ChessRules::addSlidingMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player, 
                               const std::vector<std::pair<int, int>>& directions) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    for (const auto& [rankDir, fileDir] : directions) {
        for (int step = 1; ; ++step) {
            int newRank = rank + rankDir * step;
            int newFile = file + fileDir * step;
            
            if (newRank < 0 || newRank >= 8 || newFile < 0 || newFile >= 8) {
                break;  // Off the board
            }
            
            int newSquare = getSquare(newRank, newFile);
            Piece targetPiece = get_piece_(newSquare);
            
            if (targetPiece.is_empty()) {
                // Empty square, can move here
                moves.push_back({square, newSquare});
            } else if (targetPiece.color != current_player) {
                // Capture opponent's piece
                moves.push_back({square, newSquare});
                break;  // Can't move beyond this
            } else {
                // Own piece, can't move here or beyond
                break;
            }
        }
    }
}

void ChessRules::addKingMoves(std::vector<ChessMove>& moves, int square, PieceColor current_player) const {
    int rank = getRank(square);
    int file = getFile(square);
    
    for (const auto& [rankOffset, fileOffset] : KING_MOVES) {
        int newRank = rank + rankOffset;
        int newFile = file + fileOffset;
        
        if (newRank >= 0 && newRank < 8 && newFile >= 0 && newFile < 8) {
            int newSquare = getSquare(newRank, newFile);
            Piece targetPiece = get_piece_(newSquare);
            
            if (targetPiece.is_empty() || targetPiece.color != current_player) {
                moves.push_back({square, newSquare});
            }
        }
    }
}

void ChessRules::addCastlingMoves(std::vector<ChessMove>& moves, PieceColor current_player, const CastlingRights& castling_rights) const {
    // Check if king is in check
    if (isInCheck(current_player)) {
        return;  // Cannot castle when in check
    }
    
    // Find the king
    int kingSquare = get_king_square_(current_player);
    if (kingSquare == -1) {
        return;  // No king found
    }
    
    int rank = getRank(kingSquare);
    int file = getFile(kingSquare);
    
    // Kingside castling
    if ((current_player == PieceColor::WHITE && castling_rights.white_kingside) ||
        (current_player == PieceColor::BLACK && castling_rights.black_kingside)) {
        
        bool pathClear = true;
        for (int f = file + 1; f < 7; ++f) {
            int square = getSquare(rank, f);
            if (!get_piece_(square).is_empty()) {
                pathClear = false;
                break;
            }
        }
        
        if (pathClear) {
            // Check if king passes through or ends up in check
            bool safeToPass = true;
            for (int f = file + 1; f <= file + 2; ++f) {
                int square = getSquare(rank, f);
                if (isSquareAttacked(square, oppositeColor(current_player))) {
                    safeToPass = false;
                    break;
                }
            }
            
            if (safeToPass) {
                int kingTarget = getSquare(rank, file + 2);
                moves.push_back({kingSquare, kingTarget});
            }
        }
    }
    
    // Queenside castling
    if ((current_player == PieceColor::WHITE && castling_rights.white_queenside) ||
        (current_player == PieceColor::BLACK && castling_rights.black_queenside)) {
        
        bool pathClear = true;
        for (int f = file - 1; f > 0; --f) {
            int square = getSquare(rank, f);
            if (!get_piece_(square).is_empty()) {
                pathClear = false;
                break;
            }
        }
        
        if (pathClear) {
            // Check if king passes through or ends up in check
            bool safeToPass = true;
            for (int f = file - 1; f >= file - 2; --f) {
                int square = getSquare(rank, f);
                if (isSquareAttacked(square, oppositeColor(current_player))) {
                    safeToPass = false;
                    break;
                }
            }
            
            if (safeToPass) {
                int kingTarget = getSquare(rank, file - 2);
                moves.push_back({kingSquare, kingTarget});
            }
        }
    }
}

PieceColor ChessRules::oppositeColor(PieceColor color) {
    return color == PieceColor::WHITE ? PieceColor::BLACK : PieceColor::WHITE;
}

} // namespace chess
} // namespace alphazero