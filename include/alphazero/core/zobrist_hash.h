// zobrist_hash.h
#ifndef ZOBRIST_HASH_H
#define ZOBRIST_HASH_H

#include <vector>
#include <cstdint>
#include <random>
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace core {

/**
 * @brief Zobrist hashing for game positions
 * 
 * Implementation of Zobrist hashing to generate unique hash values
 * for board positions, supporting efficient transposition table lookups.
 */
class ZobristHash {
public:
    /**
     * @brief Constructor for board games
     * 
     * @param gameType Type of game
     * @param boardSize Size of the game board
     * @param numPieces Number of different piece types
     * @param seed Random seed for deterministic initialization
     */
    ZobristHash(GameType gameType, int boardSize, int numPieces, unsigned seed = 0);
    
    /**
     * @brief Get hash value for a piece at a position
     * 
     * @param piece The piece type (usually player 1 or 2)
     * @param position The board position
     * @return 64-bit hash value
     */
    uint64_t getPieceHash(int piece, int position) const;
    
    /**
     * @brief Get player turn hash
     * 
     * @param player Current player
     * @return 64-bit hash value for the player
     */
    uint64_t getPlayerHash(int player) const;
    
    /**
     * @brief Get game-specific feature hash
     * 
     * @param featureIndex Index of the feature
     * @param value Value of the feature
     * @return 64-bit hash value
     */
    uint64_t getFeatureHash(int featureIndex, int value) const;
    
    /**
     * @brief Get board size
     * 
     * @return Board size
     */
    int getBoardSize() const { return boardSize_; }
    
    /**
     * @brief Get game type
     * 
     * @return Game type
     */
    GameType getGameType() const { return gameType_; }
    
private:
    GameType gameType_;                   // Type of game
    int boardSize_;                       // Board size
    int numPieces_;                       // Number of piece types
    int numFeatures_;                     // Number of game-specific features
    
    std::vector<std::vector<uint64_t>> pieceHashes_;   // Hash values for each piece at each position
    std::vector<uint64_t> playerHashes_;               // Hash values for player turns
    std::vector<std::vector<uint64_t>> featureHashes_; // Hash values for game-specific features
    
    /**
     * @brief Generate random 64-bit hash value
     * 
     * @param rng Random number generator
     * @return Random 64-bit hash
     */
    static uint64_t generateRandomHash(std::mt19937_64& rng);
    
    /**
     * @brief Initialize the hash table for specific game features
     * 
     * @param rng Random number generator
     */
    void initializeGameSpecificFeatures(std::mt19937_64& rng);
};

} // namespace core
} // namespace alphazero

#endif // ZOBRIST_HASH_H