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
        std::atomic<int> depth;            // Depth of search that led to this evaluation (higher is better)
        
        Entry() : hash(0), gameType(core::GameType::GOMOKU), value(0.0f), 
                 visitCount(0), lastAccessTime(0), isValid(false), depth(0) {}
        
        // Add move constructor and move assignment
        Entry(Entry&& other) noexcept
            : hash(other.hash), 
              gameType(other.gameType),
              policy(std::move(other.policy)),
              value(other.value),
              visitCount(other.visitCount.load()),
              lastAccessTime(other.lastAccessTime.load()),
              isValid(other.isValid.load()),
              depth(other.depth.load()) {}
        
        Entry& operator=(Entry&& other) noexcept {
            if (this != &other) {
                hash = other.hash;
                gameType = other.gameType;
                policy = std::move(other.policy);
                value = other.value;
                visitCount.store(other.visitCount.load());
                lastAccessTime.store(other.lastAccessTime.load());
                isValid.store(other.isValid.load());
                depth.store(other.depth.load());
            }
            return *this;
        }
        
        // Delete copy constructor and copy assignment
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
    };
    
    /**
     * @brief Replacement policies for the transposition table
     */
    enum class ReplacementPolicy {
        ALWAYS,       // Always replace
        DEPTH,        // Replace if depth is higher
        VISITS,       // Replace if visits is lower
        VISITS_AGE,   // Replace based on visits and age
        LRU           // Least Recently Used
    };
    
    /**
     * @brief Constructor
     * 
     * @param size Number of entries in the table
     * @param numShards Number of shards for lock reduction
     * @param replacementPolicy Replacement policy
     */
    explicit TranspositionTable(size_t size = 1048576, size_t numShards = 1024, 
                             ReplacementPolicy replacementPolicy = ReplacementPolicy::VISITS_AGE);
    
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
     * @brief Look up entries that match a partial hash
     * 
     * Useful for retrieving similar positions for RAVE algorithm.
     * 
     * @param partialHash Hash pattern to match
     * @param hashMask Bitmask for the hash pattern
     * @param gameType The game type
     * @param maxEntries Maximum number of entries to return
     * @return Vector of matching entries
     */
    std::vector<Entry> lookupPartial(uint64_t partialHash, uint64_t hashMask, 
                                   core::GameType gameType, size_t maxEntries = 10) const;
    
    /**
     * @brief Store an entry in the table
     * 
     * @param hash The position hash
     * @param gameType The game type
     * @param entry The entry to store
     * @param depth Search depth (higher values have priority)
     */
    void store(uint64_t hash, core::GameType gameType, const Entry& entry, int depth = 0);
    
    /**
     * @brief Store policy and value in the table
     * 
     * @param hash The position hash
     * @param gameType The game type
     * @param policy The policy vector
     * @param value The value estimate
     * @param depth Search depth (higher values have priority)
     */
    void store(uint64_t hash, core::GameType gameType, const std::vector<float>& policy, 
              float value, int depth = 0);
    
    /**
     * @brief Clear the table
     */
    void clear();
    
    /**
     * @brief Set the replacement policy parameters
     * 
     * @param maxAgeMs Maximum age in milliseconds
     * @param minVisitsThreshold Minimum visits before replacement
     * @param policy Replacement policy
     */
    void setReplacementPolicy(uint64_t maxAgeMs, int minVisitsThreshold, 
                             ReplacementPolicy policy = ReplacementPolicy::VISITS_AGE);
    
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
    float getHitRate() const { 
        return lookups_.load(std::memory_order_relaxed) > 0 ? 
               static_cast<float>(hits_.load(std::memory_order_relaxed)) / 
               lookups_.load(std::memory_order_relaxed) : 0.0f;
    }
    
    /**
     * @brief Get the number of lookups
     * 
     * @return Number of lookups
     */
    size_t getLookups() const { return lookups_.load(std::memory_order_relaxed); }
    
    /**
     * @brief Get the number of hits
     * 
     * @return Number of hits
     */
    size_t getHits() const { return hits_.load(std::memory_order_relaxed); }
    
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
    
    /**
     * @brief Age the cache to remove old entries
     * 
     * @param maxAgeMs Maximum age in milliseconds
     * @return Number of entries removed
     */
    size_t ageCache(uint64_t maxAgeMs);
    
    /**
     * @brief Prefetch entries if possible
     * 
     * @param hash Hash to prefetch
     */
    void prefetch(uint64_t hash) const;
    
    /**
     * @brief Get statistics about the table
     * 
     * @return String with statistics
     */
    std::string getStats() const;
    
private:
    // Table data
    std::vector<Entry> table_;            // The actual table storage
    size_t size_;                          // Size of the table
    size_t sizeMask_;                      // For fast modulo with power of 2
    
    // Sharding for reduced lock contention
    size_t numShards_;                     // Number of shards
    mutable std::vector<std::unique_ptr<std::mutex>> mutexShards_; // Shard mutexes
    
    // Replacement policy
    ReplacementPolicy replacementPolicy_;
    uint64_t maxAge_ = 60000;              // Maximum age in milliseconds (1 minute)
    int minVisits_ = 5;                    // Minimum visits before replacement
    
    // Stats
    mutable std::atomic<size_t> lookups_{0};      // Number of lookups
    mutable std::atomic<size_t> hits_{0};         // Number of hits
    mutable std::atomic<size_t> collisions_{0};   // Number of hash collisions
    mutable std::atomic<size_t> replacements_{0}; // Number of entry replacements
    mutable std::atomic<size_t> evictions_{0};    // Number of entry evictions
    
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
    
    /**
     * @brief Check if entry should be replaced
     * 
     * @param existing Existing entry
     * @param newDepth Depth of new entry
     * @return true if entry should be replaced
     */
    bool shouldReplace(const Entry& existing, int newDepth) const;
};

} // namespace mcts
} // namespace alphazero

#endif // TRANSPOSITION_TABLE_H