// include/alphazero/games/chess/chess960.h
#ifndef CHESS960_H
#define CHESS960_H

#include <vector>
#include <array>
#include <random>
#include <cstdint>
#include "alphazero/games/chess/chess_state.h"

namespace alphazero {
namespace chess {

/**
 * @brief Utility class for Chess960 (Fischer Random Chess)
 * 
 * This class provides utilities for generating and working with
 * Chess960 starting positions.
 */
class Chess960 {
public:
    /**
     * @brief Generate a random Chess960 starting position
     * 
     * @param seed Optional seed for the random number generator
     * @return Integer representing the position (0-959)
     */
    static int generateRandomPosition(unsigned seed = 0);
    
    /**
     * @brief Generate a Chess960 position from a specific position number
     * 
     * @param positionNumber Position number (0-959)
     * @return Array representing the pieces on the back rank (a-h)
     */
    static std::array<PieceType, 8> generatePosition(int positionNumber);
    
    /**
     * @brief Get position number from a specific arrangement
     * 
     * @param position Array representing the pieces on the back rank (a-h)
     * @return Position number (0-959), or -1 if invalid
     */
    static int getPositionNumber(const std::array<PieceType, 8>& position);
    
    /**
     * @brief Check if an arrangement is a valid Chess960 position
     * 
     * @param position Array representing the pieces on the back rank (a-h)
     * @return true if valid, false otherwise
     */
    static bool isValidPosition(const std::array<PieceType, 8>& position);
    
    /**
     * @brief Set up a Chess960 position on a chess board
     * 
     * @param positionNumber Position number (0-959)
     * @param state Chess state to update
     */
    static void setupPosition(int positionNumber, ChessState& state);
    
    /**
     * @brief Get FEN string for a Chess960 position
     * 
     * @param positionNumber Position number (0-959)
     * @return FEN string representing the starting position
     */
    static std::string getStartingFEN(int positionNumber);
    
    /**
     * @brief Convert a standard FEN position to Chess960 format
     * 
     * This is needed for proper castling notation in Chess960
     * 
     * @param standardFEN FEN in standard notation
     * @return FEN in Chess960 notation
     */
    static std::string convertToChess960FEN(const std::string& standardFEN);
    
    /**
     * @brief Get the total number of Chess960 positions
     * 
     * @return Number of positions (960)
     */
    static constexpr int getNumberOfPositions() { return 960; }
    
private:
    // Helper methods for position generation and validation
    static bool hasValidBishopPlacement(const std::array<PieceType, 8>& position);
    static bool hasKingBetweenRooks(const std::array<PieceType, 8>& position);
    static std::array<int, 8> getPermutation(int n);
};

} // namespace chess
} // namespace alphazero

#endif // CHESS960_H