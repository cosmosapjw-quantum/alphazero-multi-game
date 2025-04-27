// zobrist_hash.cpp
#include "alphazero/core/zobrist_hash.h"
#include <stdexcept>
#include <chrono>

namespace alphazero {
namespace core {

ZobristHash::ZobristHash(GameType gameType, int boardSize, int numPieces, unsigned seed)
    : gameType_(gameType), boardSize_(boardSize), numPieces_(numPieces), numFeatures_(16) {
    
    // Initialize random number generator with seed or time
    unsigned actualSeed = seed != 0 ? seed : 
        static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(actualSeed);
    
    // Initialize hash values for pieces at positions
    int totalPositions = boardSize * boardSize;
    pieceHashes_.resize(numPieces);
    for (int p = 0; p < numPieces; ++p) {
        pieceHashes_[p].resize(totalPositions);
        for (int pos = 0; pos < totalPositions; ++pos) {
            pieceHashes_[p][pos] = generateRandomHash(rng);
        }
    }
    
    // Initialize hash values for player turns
    playerHashes_.resize(numPieces);
    for (int p = 0; p < numPieces; ++p) {
        playerHashes_[p] = generateRandomHash(rng);
    }
    
    // Initialize game-specific features
    initializeGameSpecificFeatures(rng);
}

uint64_t ZobristHash::getPieceHash(int piece, int position) const {
    if (piece < 0 || piece >= numPieces_ || 
        position < 0 || position >= boardSize_ * boardSize_) {
        throw std::out_of_range("Piece or position index out of range");
    }
    return pieceHashes_[piece][position];
}

uint64_t ZobristHash::getPlayerHash(int player) const {
    if (player < 0 || player >= numPieces_) {
        throw std::out_of_range("Player index out of range");
    }
    return playerHashes_[player];
}

uint64_t ZobristHash::getFeatureHash(int featureIndex, int value) const {
    if (featureIndex < 0 || featureIndex >= numFeatures_) {
        throw std::out_of_range("Feature index out of range");
    }
    
    // Ensure value is within bounds (use modulo for safety)
    int safeValue = value % featureHashes_[featureIndex].size();
    return featureHashes_[featureIndex][safeValue];
}

uint64_t ZobristHash::generateRandomHash(std::mt19937_64& rng) {
    return rng();
}

void ZobristHash::initializeGameSpecificFeatures(std::mt19937_64& rng) {
    // Initialize with default size of features
    featureHashes_.resize(numFeatures_);
    
    // Different games have different special features that need hashing
    switch (gameType_) {
        case GameType::GOMOKU:
            // Gomoku doesn't have many special features
            // Can add ko position if using ko rule variant
            featureHashes_[0].resize(boardSize_ * boardSize_ + 1);
            for (size_t i = 0; i < featureHashes_[0].size(); ++i) {
                featureHashes_[0][i] = generateRandomHash(rng);
            }
            break;
            
        case GameType::CHESS:
            // Chess has castling rights, en passant, etc.
            // Castling rights (4 possibilities: KQ for white, KQ for black)
            featureHashes_[0].resize(16);  // 2^4 = 16 combinations
            // En passant files (8 files + none)
            featureHashes_[1].resize(9);
            // Halfmove clock for 50-move rule
            featureHashes_[2].resize(100);
            
            // Initialize all values
            for (auto& feature : featureHashes_) {
                for (size_t i = 0; i < feature.size(); ++i) {
                    feature[i] = generateRandomHash(rng);
                }
            }
            break;
            
        case GameType::GO:
            // Go has ko positions, prisoner count
            // Ko position
            featureHashes_[0].resize(boardSize_ * boardSize_ + 1);
            // Komi and rules variations
            featureHashes_[1].resize(20);  // Different komi values
            
            // Initialize all values
            for (auto& feature : featureHashes_) {
                for (size_t i = 0; i < feature.size(); ++i) {
                    feature[i] = generateRandomHash(rng);
                }
            }
            break;
            
        default:
            // Default initialization for unknown game types
            for (auto& feature : featureHashes_) {
                feature.resize(1);
                feature[0] = generateRandomHash(rng);
            }
            break;
    }
}

} // namespace core
} // namespace alphazero