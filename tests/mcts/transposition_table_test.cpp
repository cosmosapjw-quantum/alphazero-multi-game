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

TEST_F(TranspositionTableTest, ReplacementPolicy) {
    // Set replacement policy
    tt->setReplacementPolicy(1000, 2);  // Replace entries older than 1s or with < 2 visits
    
    // Add an entry
    uint64_t hash = 0x123456789ABCDEF0;
    std::vector<float> policy = {0.1f, 0.2f, 0.3f, 0.4f};
    float value = 0.5f;
    tt->store(hash, core::GameType::GOMOKU, policy, value);
    
    // Look it up to increase visit count
    TranspositionTable::Entry result;
    tt->lookup(hash, core::GameType::GOMOKU, result);
    tt->lookup(hash, core::GameType::GOMOKU, result);
    
    // Visit count should now be 3
    tt->lookup(hash, core::GameType::GOMOKU, result);
    EXPECT_EQ(result.visitCount.load(), 3);
    
    // Store with same hash but different policy - should be collision and keep old entry
    std::vector<float> policy2 = {0.9f, 0.8f, 0.7f, 0.6f};
    float value2 = 0.1f;
    tt->store(hash, core::GameType::GOMOKU, policy2, value2);
    
    // Look up again - should still have original policy
    tt->lookup(hash, core::GameType::GOMOKU, result);
    EXPECT_EQ(result.policy, policy);
    EXPECT_FLOAT_EQ(result.value, value);
}

TEST_F(TranspositionTableTest, MemoryUsage) {
    // Add some entries
    for (uint64_t i = 0; i < 10; ++i) {
        std::vector<float> policy(100, 0.1f);  // Larger policy to use more memory
        float value = 0.5f;
        tt->store(i, core::GameType::GOMOKU, policy, value);
    }
    
    // Get memory usage
    size_t mem = tt->getMemoryUsageBytes();
    
    // Should be non-zero
    EXPECT_GT(mem, 0);
    
    // Should include base table size + entries
    size_t expectedBaseSize = sizeof(TranspositionTable) + 16 * sizeof(std::mutex);
    EXPECT_GT(mem, expectedBaseSize);
}

TEST_F(TranspositionTableTest, ThreadSafety) {
    // Test concurrent access (simple test, not exhaustive)
    
    // Create threads to store and lookup concurrently
    const int numThreads = 8;
    const int entriesPerThread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t, entriesPerThread]() {
            uint64_t baseHash = static_cast<uint64_t>(t) << 32;
            
            // Each thread stores and looks up its own entries
            for (int i = 0; i < entriesPerThread; ++i) {
                uint64_t hash = baseHash | static_cast<uint64_t>(i);
                std::vector<float> policy(10, 0.1f * i);
                float value = 0.1f * i;
                
                // Store
                tt->store(hash, core::GameType::GOMOKU, policy, value);
                
                // Lookup
                TranspositionTable::Entry result;
                bool found = tt->lookup(hash, core::GameType::GOMOKU, result);
                
                // Basic check - may fail occasionally due to race conditions
                if (found) {
                    EXPECT_EQ(result.hash, hash);
                }
            }
        });
    }
    
    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Table should have some entries (exact count may vary due to collisions)
    EXPECT_GT(tt->getEntryCount(), 0);
}

TEST_F(TranspositionTableTest, MultipleGameTypes) {
    // Store entries for different game types
    uint64_t hash = 0x123456789ABCDEF0;
    
    std::vector<float> gomokuPolicy = {0.1f, 0.2f, 0.3f, 0.4f};
    float gomokuValue = 0.5f;
    tt->store(hash, core::GameType::GOMOKU, gomokuPolicy, gomokuValue);
    
    std::vector<float> chessPolicy = {0.5f, 0.6f, 0.7f, 0.8f};
    float chessValue = -0.5f;
    tt->store(hash, core::GameType::CHESS, chessPolicy, chessValue);
    
    // Lookup entries
    TranspositionTable::Entry gomokuResult;
    bool foundGomoku = tt->lookup(hash, core::GameType::GOMOKU, gomokuResult);
    
    TranspositionTable::Entry chessResult;
    bool foundChess = tt->lookup(hash, core::GameType::CHESS, chessResult);
    
    // Both should be found
    EXPECT_TRUE(foundGomoku);
    EXPECT_TRUE(foundChess);
    
    // Should have correct values
    EXPECT_EQ(gomokuResult.policy, gomokuPolicy);
    EXPECT_FLOAT_EQ(gomokuResult.value, gomokuValue);
    
    EXPECT_EQ(chessResult.policy, chessPolicy);
    EXPECT_FLOAT_EQ(chessResult.value, chessValue);
}

} // namespace mcts
} // namespace alphazero