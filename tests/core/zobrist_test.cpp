#include <gtest/gtest.h>
#include "alphazero/core/zobrist_hash.h"
#include <unordered_set>

namespace alphazero {
namespace core {

class ZobristHashTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create Zobrist hashers for different games
        gomokuHash = std::make_unique<ZobristHash>(GameType::GOMOKU, 15, 2);
        chessHash = std::make_unique<ZobristHash>(GameType::CHESS, 8, 12);
        goHash = std::make_unique<ZobristHash>(GameType::GO, 19, 2);
    }
    
    std::unique_ptr<ZobristHash> gomokuHash;
    std::unique_ptr<ZobristHash> chessHash;
    std::unique_ptr<ZobristHash> goHash;
};

TEST_F(ZobristHashTest, Initialization) {
    // Test initialization with different parameters
    
    ZobristHash hash1(GameType::GOMOKU, 9, 2);
    EXPECT_EQ(hash1.getGameType(), GameType::GOMOKU);
    EXPECT_EQ(hash1.getBoardSize(), 9);
    
    ZobristHash hash2(GameType::CHESS, 8, 12);
    EXPECT_EQ(hash2.getGameType(), GameType::CHESS);
    EXPECT_EQ(hash2.getBoardSize(), 8);
}

TEST_F(ZobristHashTest, PieceHashing) {
    // Test hashing of pieces at positions
    
    // Same piece at same position should have same hash
    uint64_t hash1 = gomokuHash->getPieceHash(0, 0);
    uint64_t hash2 = gomokuHash->getPieceHash(0, 0);
    EXPECT_EQ(hash1, hash2);
    
    // Different pieces should have different hashes
    uint64_t hash3 = gomokuHash->getPieceHash(0, 0);
    uint64_t hash4 = gomokuHash->getPieceHash(1, 0);
    EXPECT_NE(hash3, hash4);
    
    // Same piece at different positions should have different hashes
    uint64_t hash5 = gomokuHash->getPieceHash(0, 0);
    uint64_t hash6 = gomokuHash->getPieceHash(0, 1);
    EXPECT_NE(hash5, hash6);
}

TEST_F(ZobristHashTest, PlayerHashing) {
    // Test hashing of players
    
    // Same player should have same hash
    uint64_t hash1 = gomokuHash->getPlayerHash(0);
    uint64_t hash2 = gomokuHash->getPlayerHash(0);
    EXPECT_EQ(hash1, hash2);
    
    // Different players should have different hashes
    uint64_t hash3 = gomokuHash->getPlayerHash(0);
    uint64_t hash4 = gomokuHash->getPlayerHash(1);
    EXPECT_NE(hash3, hash4);
}

TEST_F(ZobristHashTest, FeatureHashing) {
    // Test hashing of game-specific features
    
    // Same feature/value should have same hash
    uint64_t hash1 = gomokuHash->getFeatureHash(0, 0);
    uint64_t hash2 = gomokuHash->getFeatureHash(0, 0);
    EXPECT_EQ(hash1, hash2);
    
    // Different features should have different hashes
    uint64_t hash3 = gomokuHash->getFeatureHash(0, 0);
    uint64_t hash4 = gomokuHash->getFeatureHash(1, 0);
    EXPECT_NE(hash3, hash4);
    
    // Same feature with different values should have different hashes
    uint64_t hash5 = gomokuHash->getFeatureHash(0, 0);
    uint64_t hash6 = gomokuHash->getFeatureHash(0, 1);
    EXPECT_NE(hash5, hash6);
}

TEST_F(ZobristHashTest, HashDistribution) {
    // Test that hashes are well-distributed
    
    std::unordered_set<uint64_t> hashes;
    
    // Generate piece hashes for all positions (limited sample)
    for (int piece = 0; piece < 2; piece++) {
        for (int pos = 0; pos < 20; pos++) {
            try {
                uint64_t hash = gomokuHash->getPieceHash(piece, pos);
                hashes.insert(hash);
            } catch (const std::exception&) {
                // Ignore out of bounds
            }
        }
    }
    
    // Generate player hashes
    for (int player = 0; player < 2; player++) {
        try {
            uint64_t hash = gomokuHash->getPlayerHash(player);
            hashes.insert(hash);
        } catch (const std::exception&) {
            // Ignore out of bounds
        }
    }
    
    // Generate feature hashes
    for (int feature = 0; feature < 3; feature++) {
        for (int value = 0; value < 3; value++) {
            try {
                uint64_t hash = gomokuHash->getFeatureHash(feature, value);
                hashes.insert(hash);
            } catch (const std::exception&) {
                // Ignore out of bounds
            }
        }
    }
    
    // Check that all hashes are unique (good distribution)
    size_t expectedCount = 2 * 20 + 2 + 3 * 3;  // Maximum possible unique hashes
    EXPECT_EQ(hashes.size(), expectedCount);
}

TEST_F(ZobristHashTest, ErrorHandling) {
    // Test error handling for invalid inputs
    
    // Invalid piece index
    EXPECT_THROW(gomokuHash->getPieceHash(-1, 0), std::out_of_range);
    EXPECT_THROW(gomokuHash->getPieceHash(2, 0), std::out_of_range);
    
    // Invalid position
    EXPECT_THROW(gomokuHash->getPieceHash(0, -1), std::out_of_range);
    EXPECT_THROW(gomokuHash->getPieceHash(0, 15*15), std::out_of_range);
    
    // Invalid player
    EXPECT_THROW(gomokuHash->getPlayerHash(-1), std::out_of_range);
    EXPECT_THROW(gomokuHash->getPlayerHash(2), std::out_of_range);
    
    // Invalid feature
    EXPECT_THROW(gomokuHash->getFeatureHash(-1, 0), std::out_of_range);
}

TEST_F(ZobristHashTest, DeterministicHashing) {
    // Test that hashing is deterministic with same seed
    
    // Create two hashers with same parameters and seed
    ZobristHash hash1(GameType::GOMOKU, 9, 2, 42);
    ZobristHash hash2(GameType::GOMOKU, 9, 2, 42);
    
    // Same inputs should produce same hashes
    EXPECT_EQ(hash1.getPieceHash(0, 0), hash2.getPieceHash(0, 0));
    EXPECT_EQ(hash1.getPlayerHash(0), hash2.getPlayerHash(0));
    EXPECT_EQ(hash1.getFeatureHash(0, 0), hash2.getFeatureHash(0, 0));
}

} // namespace core
} // namespace alphazero