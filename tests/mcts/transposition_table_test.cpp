#include <gtest/gtest.h>
#include "alphazero/mcts/transposition_table.h"

namespace alphazero {
namespace mcts {

class TranspositionTableTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a transposition table with 1024 entries and 16 shards
        tt = std::make_unique<TranspositionTable>(1024, 16);
    }
    
    std::unique_ptr<TranspositionTable> tt;
};

TEST_F(TranspositionTableTest, StoreAndLookup) {
    uint64_t hash1 = 0x123456789ABCDEF0;
    core::GameType gameType = core::GameType::GOMOKU;
    std::vector<float> policy1 = {0.1f, 0.2f, 0.3f, 0.4f};
    float value1 = 0.5f;
    
    // Store entry
    tt->store(hash1, gameType, policy1, value1);
    
    // Lookup entry
    TranspositionTable::Entry result;
    bool found = tt->lookup(hash1, gameType, result);
    
    // Should find the entry
    EXPECT_TRUE(found);
    
    // Entry should match what we stored
    EXPECT_EQ(result.hash, hash1);
    EXPECT_EQ(result.gameType, gameType);
    EXPECT_EQ(result.policy, policy1);
    EXPECT_FLOAT_EQ(result.value, value1);
    EXPECT_EQ(result.visitCount.load(), 1);
}

TEST_F(TranspositionTableTest, EntryCount) {
    // Initially empty
    EXPECT_EQ(tt->getEntryCount(), 0);
    
    // Add some entries
    for (uint64_t i = 0; i < 10; ++i) {
        std::vector<float> policy = {static_cast<float>(i) / 10.0f};
        float value = static_cast<float>(i) / 10.0f;
        tt->store(i, core::GameType::GOMOKU, policy, value);
    }
    
    // Should have 10 entries
    EXPECT_EQ(tt->getEntryCount(), 10);
}

TEST_F(TranspositionTableTest, HitRate) {
    // Initially zero lookups and hits
    EXPECT_EQ(tt->getLookups(), 0);
    EXPECT_EQ(tt->getHits(), 0);
    EXPECT_FLOAT_EQ(tt->getHitRate(), 0.0f);
    
    // Add an entry
    uint64_t hash = 0x123456789ABCDEF0;
    std::vector<float> policy = {0.1f, 0.2f, 0.3f, 0.4f};
    float value = 0.5f;
    tt->store(hash, core::GameType::GOMOKU, policy, value);
    
    // Look it up twice - should be 2 hits
    TranspositionTable::Entry result;
    tt->lookup(hash, core::GameType::GOMOKU, result);
    tt->lookup(hash, core::GameType::GOMOKU, result);
    
    // Look up a non-existent entry - should be a miss
    tt->lookup(hash + 1, core::GameType::GOMOKU, result);
    
    // Should have 3 lookups, 2 hits
    EXPECT_EQ(tt->getLookups(), 3);
    EXPECT_EQ(tt->getHits(), 2);
    EXPECT_FLOAT_EQ(tt->getHitRate(), 2.0f / 3.0f);
}

TEST_F(TranspositionTableTest, Clear) {
    // Add some entries
    for (uint64_t i = 0; i < 10; ++i) {
        std::vector<float> policy = {static_cast<float>(i) / 10.0f};
        float value = static_cast<float>(i) / 10.0f;
        tt->store(i, core::GameType::GOMOKU, policy, value);
    }
    
    // Should have 10 entries
    EXPECT_EQ(tt->getEntryCount(), 10);
    
    // Clear the table
    tt->clear();
    
    // Should be empty again
    EXPECT_EQ(tt->getEntryCount(), 0);
}

TEST_F(TranspositionTableTest, Resize) {
    // Add some entries
    for (uint64_t i = 0; i < 10; ++i) {
        std::vector<float> policy = {static_cast<float>(i) / 10.0f};
        float value = static_cast<float>(i) / 10.0f;
        tt->store(i, core::GameType::GOMOKU, policy, value);
    }
    
    // Resize to 512 entries
    tt->resize(512);
    
    // Size should now be 512
    EXPECT_EQ(tt->getSize(), 512);
    
    // Should still have the entries
    TranspositionTable::Entry result;
    for (uint64_t i = 0; i < 10; ++i) {
        bool found = tt->lookup(i, core::GameType::GOMOKU, result);
        EXPECT_TRUE(found);
    }
}

} // namespace mcts
} // namespace alphazero