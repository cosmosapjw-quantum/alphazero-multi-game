// transposition_table.h
#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <memory>
#include <chrono>
#include <unordered_map>
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace mcts {

/**
 * @brief Transposition table for caching evaluated positions
 * 
 * This class implements a thread-safe transposition table that
 * caches evaluation results to avoid redundant calculations.
 */
class TranspositionTable {
public:
    /**
     * @brief Entry in the transposition table
     */
    struct Entry {
        uint64_t hash;                     // Zobrist hash of the position
        core::GameType gameType;           // Game type this entry belongs to
        std::vector<float> policy;         // Policy vector from neural network
        float value;                       // Value estimate from neural network
        std::atomic<int> visitCount;       // Number of times this entry was used
        std::atomic<uint64_t> lastAccessTime;  // For aging/replacement policy
        std::atomic<bool> isValid;         // Whether this entry contains valid data
        
        Entry() : hash(0), gameType(core::GameType::GOMOKU), value(0.0f), 
                 visitCount(0), lastAccessTime(0), isValid(false) {}
    };
    
    /**
     * @brief Constructor
     * 
     * @param size Number of entries in the table
     * @param numShards Number of shards for lock reduction
     */
    explicit TranspositionTable(size_t size = 1048576, size_t numShards = 1024);
    
    /**
     * @brief Destructor
     */
    ~TranspositionTable() = default;
    
    // Non-copyable but movable
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;
    TranspositionTable(TranspositionTable&&) noexcept = default;
    TranspositionTable& operator=(TranspositionTable&&) noexcept = default;
    
    /**
     * @brief Look up an entry in the table
     * 
     * @param hash The position hash
     * @param gameType The game type
     * @param result Reference to store the result
     * @return true if found, false otherwise
     */
    bool lookup(uint64_t hash, core::GameType gameType, Entry& result) const;
    
    /**
     * @brief Store an entry in the table
     * 
     * @param hash The position hash
     * @param gameType The game type
     * @param entry The entry to store
     */
    void store(uint64_t hash, core::GameType gameType, const Entry& entry);
    
    /**
     * @brief Store policy and value in the table
     * 
     * @param hash The position hash
     * @param gameType The game type
     * @param policy The policy vector
     * @param value The value estimate
     */
    void store(uint64_t hash, core::GameType gameType, const std::vector<float>& policy, float value);
    
    /**
     * @brief Clear the table
     */
    void clear();
    
    /**
     * @brief Set the replacement policy parameters
     * 
     * @param maxAgeMs Maximum age in milliseconds
     * @param minVisitsThreshold Minimum visits before replacement
     */
    void setReplacementPolicy(uint64_t maxAgeMs, int minVisitsThreshold);
    
    /**
     * @brief Get the size of the table
     * 
     * @return Size of the table
     */
    size_t getSize() const { return size_; }
    
    /**
     * @brief Get the hit rate
     * 
     * @return Cache hit rate
     */
    float getHitRate() const { return lookups_ > 0 ? static_cast<float>(hits_) / lookups_ : 0.0f; }
    
    /**
     * @brief Get the number of lookups
     * 
     * @return Number of lookups
     */
    size_t getLookups() const { return lookups_; }
    
    /**
     * @brief Get the number of hits
     * 
     * @return Number of hits
     */
    size_t getHits() const { return hits_; }
    
    /**
     * @brief Get the number of valid entries
     * 
     * @return Number of valid entries
     */
    size_t getEntryCount() const;
    
    /**
     * @brief Get the memory usage in bytes
     * 
     * @return Memory usage in bytes
     */
    size_t getMemoryUsageBytes() const;
    
    /**
     * @brief Resize the table
     * 
     * @param newSize New size of the table
     */
    void resize(size_t newSize);
    
private:
    // Table data
    std::vector<Entry> table_;            // The actual table storage
    size_t size_;                          // Size of the table
    size_t sizeMask_;                      // For fast modulo with power of 2
    
    // Sharding for reduced lock contention
    size_t numShards_;                     // Number of shards
    mutable std::vector<std::mutex> mutexShards_; // Shard mutexes
    
    // Replacement policy
    uint64_t maxAge_ = 60000;              // Maximum age in milliseconds (1 minute)
    int minVisits_ = 5;                    // Minimum visits before replacement
    
    // Stats
    mutable std::atomic<size_t> lookups_{0};      // Number of lookups
    mutable std::atomic<size_t> hits_{0};         // Number of hits
    mutable std::atomic<size_t> collisions_{0};   // Number of hash collisions
    mutable std::atomic<size_t> replacements_{0}; // Number of entry replacements
    
    /**
     * @brief Get current time in milliseconds
     * 
     * @return Current time in milliseconds
     */
    uint64_t getCurrentTime() const;
    
    /**
     * @brief Calculate hash index in table
     * 
     * @param hash The position hash
     * @return Table index
     */
    size_t getHashIndex(uint64_t hash) const { return hash & sizeMask_; }
    
    /**
     * @brief Calculate shard index for mutex
     * 
     * @param hash The position hash
     * @return Shard index
     */
    size_t getShardIndex(uint64_t hash) const { return hash % numShards_; }
};

} // namespace mcts
} // namespace alphazero

#endif // TRANSPOSITION_TABLE_H