// src/games/chess/chess960.cpp
#include "alphazero/games/chess/chess960.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace alphazero {
namespace chess {

int Chess960::generateRandomPosition(unsigned seed) {
    // Initialize random number generator
    if (seed == 0) {
        seed = static_cast<unsigned>(
            std::chrono::system_clock::now().time_since_epoch().count());
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 959);
    
    return dist(rng);
}

std::array<PieceType, 8> Chess960::generatePosition(int positionNumber) {
    if (positionNumber < 0 || positionNumber >= 960) {
        throw std::invalid_argument("Position number must be between 0 and 959");
    }
    
    // Initialize the back rank with empty spaces
    std::array<PieceType, 8> position;
    position.fill(PieceType::NONE);
    
    // Convert position number to a valid Chess960 arrangement using the permutation algorithm
    std::array<int, 8> arrangement = getPermutation(positionNumber);
    
    // Map the arrangement to actual pieces
    for (int i = 0; i < 8; ++i) {
        switch (arrangement[i]) {
            case 0: // First bishop (must be on odd square)
                position[i] = PieceType::BISHOP;
                break;
            case 1: // Second bishop (must be on even square)
                position[i] = PieceType::BISHOP;
                break;
            case 2: // Queen
                position[i] = PieceType::QUEEN;
                break;
            case 3: // First knight
                position[i] = PieceType::KNIGHT;
                break;
            case 4: // Second knight
                position[i] = PieceType::KNIGHT;
                break;
            case 5: // First rook
                position[i] = PieceType::ROOK;
                break;
            case 6: // King (must be between rooks)
                position[i] = PieceType::KING;
                break;
            case 7: // Second rook
                position[i] = PieceType::ROOK;
                break;
            default:
                throw std::runtime_error("Invalid piece index in Chess960 generation");
        }
    }
    
    // Verify the generated position is valid
    assert(isValidPosition(position));
    
    return position;
}

int Chess960::getPositionNumber(const std::array<PieceType, 8>& position) {
    if (!isValidPosition(position)) {
        return -1;  // Invalid position
    }
    
    // This is a complex reverse mapping that requires understanding the specific
    // algorithm used to generate positions. For simplicity, we'll use a brute-force
    // approach here, which is inefficient but correct.
    for (int i = 0; i < 960; ++i) {
        if (generatePosition(i) == position) {
            return i;
        }
    }
    
    return -1;  // Should not reach here if position is valid
}

bool Chess960::isValidPosition(const std::array<PieceType, 8>& position) {
    // Check that we have exactly the right pieces
    int bishops = 0;
    int knights = 0;
    int rooks = 0;
    int queens = 0;
    int kings = 0;
    
    for (PieceType piece : position) {
        switch (piece) {
            case PieceType::BISHOP: bishops++; break;
            case PieceType::KNIGHT: knights++; break;
            case PieceType::ROOK: rooks++; break;
            case PieceType::QUEEN: queens++; break;
            case PieceType::KING: kings++; break;
            default: return false;  // No empty spaces or other pieces allowed
        }
    }
    
    if (bishops != 2 || knights != 2 || rooks != 2 || queens != 1 || kings != 1) {
        return false;
    }
    
    // Ensure bishops are on opposite colored squares
    if (!hasValidBishopPlacement(position)) {
        return false;
    }
    
    // Ensure king is between the two rooks
    if (!hasKingBetweenRooks(position)) {
        return false;
    }
    
    return true;
}

void Chess960::setupPosition(int positionNumber, ChessState& state) {
    // Clear the board first
    for (int square = 0; square < 64; ++square) {
        state.setPiece(square, Piece());
    }
    
    // Generate the position arrangement
    std::array<PieceType, 8> arrangement = generatePosition(positionNumber);
    
    // Set up white pieces (back rank)
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(7, file), {arrangement[file], PieceColor::WHITE});
    }
    
    // Set up white pawns
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(6, file), {PieceType::PAWN, PieceColor::WHITE});
    }
    
    // Set up black pieces (back rank)
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(0, file), {arrangement[file], PieceColor::BLACK});
    }
    
    // Set up black pawns
    for (int file = 0; file < 8; ++file) {
        state.setPiece(ChessState::getSquare(1, file), {PieceType::PAWN, PieceColor::BLACK});
    }
    
    // Reset game state (player, castling rights, etc.)
    std::string fen = getStartingFEN(positionNumber);
    state.setFromFEN(fen);
}

std::string Chess960::getStartingFEN(int positionNumber) {
    std::array<PieceType, 8> arrangement = generatePosition(positionNumber);
    
    // Construct the FEN string
    std::stringstream ss;
    
    // First rank (black pieces)
    for (int file = 0; file < 8; ++file) {
        char pieceChar;
        switch (arrangement[file]) {
            case PieceType::PAWN: pieceChar = 'p'; break;
            case PieceType::KNIGHT: pieceChar = 'n'; break;
            case PieceType::BISHOP: pieceChar = 'b'; break;
            case PieceType::ROOK: pieceChar = 'r'; break;
            case PieceType::QUEEN: pieceChar = 'q'; break;
            case PieceType::KING: pieceChar = 'k'; break;
            default: pieceChar = '?'; break;
        }
        ss << pieceChar;
    }
    
    // Remaining ranks
    ss << "/pppppppp/8/8/8/8/PPPPPPPP/";
    
    // Last rank (white pieces)
    for (int file = 0; file < 8; ++file) {
        char pieceChar;
        switch (arrangement[file]) {
            case PieceType::PAWN: pieceChar = 'P'; break;
            case PieceType::KNIGHT: pieceChar = 'N'; break;
            case PieceType::BISHOP: pieceChar = 'B'; break;
            case PieceType::ROOK: pieceChar = 'R'; break;
            case PieceType::QUEEN: pieceChar = 'Q'; break;
            case PieceType::KING: pieceChar = 'K'; break;
            default: pieceChar = '?'; break;
        }
        ss << pieceChar;
    }
    
    // White to move, full castling rights, no en passant, no halfmove clock, fullmove number 1
    ss << " w KQkq - 0 1";
    
    return ss.str();
}

std::string Chess960::convertToChess960FEN(const std::string& standardFEN) {
    // For Chess960, castling rights use the file letter of the rook
    // instead of the standard KQkq notation
    
    // Parse FEN
    std::istringstream iss(standardFEN);
    std::string position, activeColor, castlingRights, enPassant, halfmoves, fullmoves;
    
    if (!(iss >> position >> activeColor >> castlingRights >> enPassant >> halfmoves >> fullmoves)) {
        throw std::invalid_argument("Invalid FEN string");
    }
    
    // If no castling rights or already using Chess960 notation, return as is
    if (castlingRights == "-" || (castlingRights.find_first_of("KQkq") == std::string::npos)) {
        return standardFEN;
    }
    
    // Parse the board position to find initial rook positions
    std::vector<std::string> ranks;
    size_t pos = 0;
    std::string token;
    while ((pos = position.find('/')) != std::string::npos) {
        ranks.push_back(position.substr(0, pos));
        position.erase(0, pos + 1);
    }
    ranks.push_back(position);  // Add the last rank
    
    if (ranks.size() != 8) {
        throw std::invalid_argument("Invalid FEN: wrong number of ranks");
    }
    
    // Find white rooks (in the 8th rank)
    std::string whiteRank = ranks[7];
    std::vector<int> whiteRookFiles;
    for (int file = 0; file < static_cast<int>(whiteRank.length()); ++file) {
        if (whiteRank[file] == 'R') {
            whiteRookFiles.push_back(file);
        } else if (std::isdigit(whiteRank[file])) {
            file += whiteRank[file] - '1';  // Skip empty squares
        }
    }
    
    // Find black rooks (in the 1st rank)
    std::string blackRank = ranks[0];
    std::vector<int> blackRookFiles;
    for (int file = 0; file < static_cast<int>(blackRank.length()); ++file) {
        if (blackRank[file] == 'r') {
            blackRookFiles.push_back(file);
        } else if (std::isdigit(blackRank[file])) {
            file += blackRank[file] - '1';  // Skip empty squares
        }
    }
    
    // Convert castling rights to Chess960 notation
    std::string newCastlingRights;
    if (castlingRights.find('K') != std::string::npos && whiteRookFiles.size() >= 2) {
        newCastlingRights += static_cast<char>('A' + whiteRookFiles.back());
    }
    if (castlingRights.find('Q') != std::string::npos && whiteRookFiles.size() >= 1) {
        newCastlingRights += static_cast<char>('A' + whiteRookFiles.front());
    }
    if (castlingRights.find('k') != std::string::npos && blackRookFiles.size() >= 2) {
        newCastlingRights += static_cast<char>('a' + blackRookFiles.back());
    }
    if (castlingRights.find('q') != std::string::npos && blackRookFiles.size() >= 1) {
        newCastlingRights += static_cast<char>('a' + blackRookFiles.front());
    }
    
    if (newCastlingRights.empty()) {
        newCastlingRights = "-";
    }
    
    // Reconstruct FEN with new castling rights
    std::stringstream result;
    for (size_t i = 0; i < ranks.size(); ++i) {
        result << ranks[i];
        if (i < ranks.size() - 1) {
            result << '/';
        }
    }
    result << " " << activeColor << " " << newCastlingRights << " " 
           << enPassant << " " << halfmoves << " " << fullmoves;
    
    return result.str();
}

// Helper methods

bool Chess960::hasValidBishopPlacement(const std::array<PieceType, 8>& position) {
    // Find indices of bishops
    std::vector<int> bishopIndices;
    for (int i = 0; i < 8; ++i) {
        if (position[i] == PieceType::BISHOP) {
            bishopIndices.push_back(i);
        }
    }
    
    if (bishopIndices.size() != 2) {
        return false;
    }
    
    // Check if bishops are on opposite colored squares
    return (bishopIndices[0] % 2) != (bishopIndices[1] % 2);
}

bool Chess960::hasKingBetweenRooks(const std::array<PieceType, 8>& position) {
    // Find indices of king and rooks
    int kingIndex = -1;
    std::vector<int> rookIndices;
    
    for (int i = 0; i < 8; ++i) {
        if (position[i] == PieceType::KING) {
            kingIndex = i;
        } else if (position[i] == PieceType::ROOK) {
            rookIndices.push_back(i);
        }
    }
    
    if (kingIndex == -1 || rookIndices.size() != 2) {
        return false;
    }
    
    // King must be between two rooks
    return (kingIndex > rookIndices[0] && kingIndex < rookIndices[1]) || 
           (kingIndex > rookIndices[1] && kingIndex < rookIndices[0]);
}

std::array<int, 8> Chess960::getPermutation(int n) {
    // This algorithm generates a valid Chess960 arrangement from a position number
    // The algorithm ensures:
    // 1. Bishops are on opposite colored squares
    // 2. King is between the two rooks
    
    if (n < 0 || n >= 960) {
        throw std::invalid_argument("Position number must be between 0 and 959");
    }
    
    std::array<int, 8> result;
    result.fill(-1);  // Initialize with -1 (empty)
    
    // Place bishops on opposite colored squares
    // There are 4 odd squares and 4 even squares
    // So there are 4 * 4 = 16 ways to place two bishops
    int bishopConfig = n % 16;
    int firstBishop = 2 * (bishopConfig / 4) + 1;  // Odd square (1, 3, 5, 7)
    int secondBishop = 2 * (bishopConfig % 4);     // Even square (0, 2, 4, 6)
    
    result[firstBishop] = 0;   // First bishop
    result[secondBishop] = 1;  // Second bishop
    
    // Place the queen (6 remaining squares)
    int queenConfig = (n / 16) % 6;
    int queenPos = 0;
    for (int i = 0; i < 8; ++i) {
        if (result[i] == -1) {  // Empty square
            if (queenConfig == 0) {
                result[i] = 2;  // Queen
                queenPos = i;
                break;
            }
            queenConfig--;
        }
    }
    
    // Place knights (5 * 4 / 2 = 10 configurations for 2 knights in 5 remaining squares)
    int knightConfig = (n / (16 * 6)) % 10;
    
    // Convert knightConfig to two positions
    int firstKnight = knightConfig / 4;
    int secondKnight = knightConfig % 4 + (firstKnight < (knightConfig % 4) ? 1 : 0);
    
    // Map these to actual positions
    int knightCount = 0;
    for (int i = 0; i < 8; ++i) {
        if (result[i] == -1) {  // Empty square
            if (knightCount == firstKnight || knightCount == secondKnight) {
                result[i] = 3 + (knightCount == firstKnight ? 0 : 1);  // Knights
            }
            knightCount++;
        }
    }
    
    // Place king and rooks in the remaining 3 squares
    // King must be between the rooks
    int kingRookConfig = (n / (16 * 6 * 10)) % 6;
    
    // Find the 3 remaining empty squares
    std::vector<int> emptySquares;
    for (int i = 0; i < 8; ++i) {
        if (result[i] == -1) {
            emptySquares.push_back(i);
        }
    }
    
    // 6 possible arrangements for rook-king-rook
    switch (kingRookConfig) {
        case 0:  // R K R
            result[emptySquares[0]] = 5;  // First rook
            result[emptySquares[1]] = 6;  // King
            result[emptySquares[2]] = 7;  // Second rook
            break;
        case 1:  // R R K
            result[emptySquares[0]] = 5;  // First rook
            result[emptySquares[1]] = 7;  // Second rook
            result[emptySquares[2]] = 6;  // King
            break;
        case 2:  // K R R
            result[emptySquares[0]] = 6;  // King
            result[emptySquares[1]] = 5;  // First rook
            result[emptySquares[2]] = 7;  // Second rook
            break;
        case 3:  // R K R (reversed)
            result[emptySquares[0]] = 7;  // Second rook
            result[emptySquares[1]] = 6;  // King
            result[emptySquares[2]] = 5;  // First rook
            break;
        case 4:  // K R R (reversed)
            result[emptySquares[0]] = 6;  // King
            result[emptySquares[1]] = 7;  // Second rook
            result[emptySquares[2]] = 5;  // First rook
            break;
        case 5:  // R R K (reversed)
            result[emptySquares[0]] = 7;  // Second rook
            result[emptySquares[1]] = 5;  // First rook
            result[emptySquares[2]] = 6;  // King
            break;
        default:
            throw std::runtime_error("Invalid king-rook configuration");
    }
    
    // Ensure all values are set
    for (int i = 0; i < 8; ++i) {
        assert(result[i] >= 0 && result[i] <= 7);
    }
    
    return result;
}

} // namespace chess
} // namespace alphazero