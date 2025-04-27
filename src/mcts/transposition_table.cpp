// transposition_table.cpp
#include "alphazero/mcts/transposition_table.h"
#include <cmath>
#include <algorithm>
#include <thread>

namespace alphazero {
namespace mcts {

TranspositionTable::TranspositionTable(size_t size, size_t numShards)
    : numShards_(numShards) {
    // Ensure size is a power of 2 for efficient modulo
    size_t powerOf2Size = 1;
    while (powerOf2Size < size) {
        powerOf2Size *= 2;
    }
    
    size_ = powerOf2Size;
    sizeMask_ = size_ - 1;
    
    // Initialize table
    table_.resize(size_);
    mutexShards_.resize(numShards_);
    
    // Reset all entries
    clear();
}

bool TranspositionTable::lookup(uint64_t hash, core::GameType gameType, Entry& result) const {
    lookups_.fetch_add(1, std::memory_order_relaxed);
    
    // Calculate table index
    size_t index = getHashIndex(hash);
    
    // Lock the appropriate shard
    size_t shardIndex = getShardIndex(hash);
    std::lock_guard<std::mutex> lock(mutexShards_[shardIndex]);
    
    // Check if entry is valid and matches
    const Entry& entry = table_[index];
    if (entry.isValid.load() && entry.hash == hash && entry.gameType == gameType) {
        // Copy entry data
        result.hash = entry.hash;
        result.gameType = entry.gameType;
        result.policy = entry.policy;
        result.value = entry.value;
        result.visitCount.store(entry.visitCount.load());
        result.lastAccessTime.store(getCurrentTime());
        result.isValid.store(true);
        
        // Update access time and visit count in the table
        entry.lastAccessTime.store(getCurrentTime());
        entry.visitCount.fetch_add(1, std::memory_order_relaxed);
        
        hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    
    return false;
}

void TranspositionTable::store(uint64_t hash, core::GameType gameType, const Entry& entry) {
    // Calculate table index
    size_t index = getHashIndex(hash);
    
    // Lock the appropriate shard
    size_t shardIndex = getShardIndex(hash);
    std::lock_guard<std::mutex> lock(mutexShards_[shardIndex]);
    
    // Check if we need to replace an existing entry
    Entry& tableEntry = table_[index];
    if (tableEntry.isValid.load()) {
        // Check if existing entry should be kept (based on visit count and age)
        uint64_t currentTime = getCurrentTime();
        uint64_t entryAge = currentTime - tableEntry.lastAccessTime.load();
        int visitCount = tableEntry.visitCount.load();
        
        if (visitCount >= minVisits_ && entryAge < maxAge_) {
            // Existing entry is still useful, don't replace
            collisions_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        
        // Replace existing entry
        replacements_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Store new entry
    tableEntry.hash = hash;
    tableEntry.gameType = gameType;
    tableEntry.policy = entry.policy;
    tableEntry.value = entry.value;
    tableEntry.visitCount.store(1);
    tableEntry.lastAccessTime.store(getCurrentTime());
    tableEntry.isValid.store(true);
}

void TranspositionTable::store(uint64_t hash, core::GameType gameType, 
                             const std::vector<float>& policy, float value) {
    Entry entry;
    entry.hash = hash;
    entry.gameType = gameType;
    entry.policy = policy;
    entry.value = value;
    entry.visitCount.store(1);
    entry.lastAccessTime.store(getCurrentTime());
    entry.isValid.store(true);
    
    store(hash, gameType, entry);
}

void TranspositionTable::clear() {
    // Lock all shards
    for (auto& mutex : mutexShards_) {
        mutex.lock();
    }
    
    // Clear all entries
    for (auto& entry : table_) {
        entry.isValid.store(false);
        entry.visitCount.store(0);
        entry.lastAccessTime.store(0);
        entry.policy.clear();
        entry.value = 0.0f;
    }
    
    // Reset statistics
    lookups_.store(0, std::memory_order_relaxed);
    hits_.store(0, std::memory_order_relaxed);
    collisions_.store(0, std::memory_order_relaxed);
    replacements_.store(0, std::memory_order_relaxed);
    
    // Unlock all shards
    for (auto& mutex : mutexShards_) {
        mutex.unlock();
    }
}

void TranspositionTable::setReplacementPolicy(uint64_t maxAgeMs, int minVisitsThreshold) {
    maxAge_ = maxAgeMs;
    minVisits_ = minVisitsThreshold;
}

size_t TranspositionTable::getEntryCount() const {
    size_t count = 0;
    
    // Lock all shards
    for (auto& mutex : mutexShards_) {
        mutex.lock();
    }
    
    // Count valid entries
    for (const auto& entry : table_) {
        if (entry.isValid.load()) {
            count++;
        }
    }
    
    // Unlock all shards
    for (auto& mutex : mutexShards_) {
        mutex.unlock();
    }
    
    return count;
}

size_t TranspositionTable::getMemoryUsageBytes() const {
    // Calculate base memory usage
    size_t baseUsage = sizeof(TranspositionTable) + 
                      sizeof(std::mutex) * numShards_;
    
    // Lock all shards
    for (auto& mutex : mutexShards_) {
        mutex.lock();
    }
    
    // Calculate memory for entries
    size_t entryMemory = sizeof(Entry) * table_.size();
    size_t policyMemory = 0;
    
    for (const auto& entry : table_) {
        if (entry.isValid.load()) {
            policyMemory += sizeof(float) * entry.policy.size();
        }
    }
    
    // Unlock all shards
    for (auto& mutex : mutexShards_) {
        mutex.unlock();
    }
    
    return baseUsage + entryMemory + policyMemory;
}

void TranspositionTable::resize(size_t newSize) {
    // Ensure newSize is a power of 2
    size_t powerOf2Size = 1;
    while (powerOf2Size < newSize) {
        powerOf2Size *= 2;
    }
    
    // Create new table
    TranspositionTable newTable(powerOf2Size, numShards_);
    
    // Copy entries from current table to new table
    for (size_t i = 0; i < size_; ++i) {
        if (table_[i].isValid.load()) {
            Entry entry = table_[i];
            newTable.store(entry.hash, entry.gameType, entry);
        }
    }
    
    // Swap contents
    std::swap(table_, newTable.table_);
    size_ = newTable.size_;
    sizeMask_ = newTable.sizeMask_;
}

uint64_t TranspositionTable::getCurrentTime() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return static_cast<uint64_t>(millis);
}

} // namespace mcts
} // namespace alphazero