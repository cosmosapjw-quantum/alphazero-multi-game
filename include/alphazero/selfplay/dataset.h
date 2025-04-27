// include/alphazero/selfplay/dataset.h
#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <utility>
#include "alphazero/selfplay/game_record.h"
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace selfplay {

/**
 * @brief Training example structure
 */
struct TrainingExample {
    std::vector<std::vector<std::vector<float>>> state;  // Board state tensor
    std::vector<float> policy;                          // Target policy
    float value;                                        // Target value
    
    // Serialization methods
    std::string toJson() const;
    static TrainingExample fromJson(const std::string& json);
};

/**
 * @brief Dataset for managing training examples
 */
class Dataset {
public:
    /**
     * @brief Constructor
     */
    Dataset();
    
    /**
     * @brief Add a game record to the dataset
     * 
     * @param record Game record to add
     * @param useEnhancedFeatures Whether to use enhanced features
     */
    void addGameRecord(const GameRecord& record, bool useEnhancedFeatures = true);
    
    /**
     * @brief Extract training examples from game records
     * 
     * @param includeAugmentations Whether to include data augmentations
     */
    void extractExamples(bool includeAugmentations = true);
    
    /**
     * @brief Get the number of examples
     * 
     * @return Number of examples
     */
    size_t size() const;
    
    /**
     * @brief Get a batch of examples
     * 
     * @param batchSize Batch size
     * @return Tuple of state tensors, policies, and values
     */
    std::tuple<std::vector<std::vector<std::vector<std::vector<float>>>>,
               std::vector<std::vector<float>>,
               std::vector<float>> getBatch(size_t batchSize) const;
    
    /**
     * @brief Shuffle the examples
     */
    void shuffle();
    
    /**
     * @brief Save dataset to file
     * 
     * @param filename Filename to save to
     * @return true if successful, false otherwise
     */
    bool saveToFile(const std::string& filename) const;
    
    /**
     * @brief Load dataset from file
     * 
     * @param filename Filename to load from
     * @return true if successful, false otherwise
     */
    bool loadFromFile(const std::string& filename);
    
    /**
     * @brief Get a random subset of examples
     * 
     * @param count Number of examples to get
     * @return Subset of examples
     */
    std::vector<TrainingExample> getRandomSubset(size_t count) const;
    
private:
    std::vector<GameRecord> gameRecords_;          // Original game records
    std::vector<TrainingExample> examples_;        // Extracted training examples
    mutable std::mt19937 rng_;                     // Random number generator
    
    /**
     * @brief Apply data augmentation to a training example
     * 
     * @param example Original example
     * @param gameType Type of game
     * @return Vector of augmented examples
     */
    std::vector<TrainingExample> augmentExample(
        const TrainingExample& example, core::GameType gameType) const;
};

} // namespace selfplay
} // namespace alphazero

#endif // DATASET_H