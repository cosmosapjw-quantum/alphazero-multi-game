// include/alphazero/selfplay/game_record.h
#ifndef GAME_RECORD_H
#define GAME_RECORD_H

// Include our types header first to prevent pthread conflicts
#include "alphazero/types.h"

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <utility>
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace selfplay {

/**
 * @brief Move data in a game record
 */
struct MoveData {
    int action;                         // The action taken
    std::vector<float> policy;          // Policy vector from MCTS
    float value;                        // Value prediction
    int64_t thinking_time_ms;           // Time spent on this move
    
    // Serialization methods
    std::string toJson() const;
    static MoveData fromJson(const std::string& json);
};

/**
 * @brief Game record for storing self-play data
 */
class GameRecord {
public:
    /**
     * @brief Constructor
     * 
     * @param gameType Game type
     * @param boardSize Board size
     * @param useVariantRules Whether variant rules were used
     */
    GameRecord(core::GameType gameType, int boardSize, bool useVariantRules = false);
    
    /**
     * @brief Add a move to the record
     * 
     * @param action The action taken
     * @param policy The policy vector from MCTS
     * @param value The value prediction
     * @param thinkingTimeMs Time spent on this move
     */
    void addMove(int action, const std::vector<float>& policy, float value, int64_t thinkingTimeMs);
    
    /**
     * @brief Set the final result of the game
     * 
     * @param result Game result
     */
    void setResult(core::GameResult result);
    
    /**
     * @brief Get the game metadata
     * 
     * @return Game type, board size, etc.
     */
    std::tuple<core::GameType, int, bool> getMetadata() const;
    
    /**
     * @brief Get the moves
     * 
     * @return Vector of move data
     */
    const std::vector<MoveData>& getMoves() const;
    
    /**
     * @brief Get the game result
     * 
     * @return Game result
     */
    core::GameResult getResult() const;
    
    /**
     * @brief Serialize to JSON
     * 
     * @return JSON string representation
     */
    std::string toJson() const;
    
    /**
     * @brief Deserialize from JSON
     * 
     * @param json JSON string
     * @return GameRecord object
     */
    static GameRecord fromJson(const std::string& json);
    
    /**
     * @brief Save to file
     * 
     * @param filename Filename to save to
     * @return true if successful, false otherwise
     */
    bool saveToFile(const std::string& filename) const;
    
    /**
     * @brief Load from file
     * 
     * @param filename Filename to load from
     * @return GameRecord object
     */
    static GameRecord loadFromFile(const std::string& filename);
    
private:
    core::GameType gameType_;           // Type of game
    int boardSize_;                     // Board size
    bool useVariantRules_;              // Whether variant rules were used
    std::vector<MoveData> moves_;       // List of moves
    core::GameResult result_;           // Final result
    std::chrono::system_clock::time_point timestamp_;  // When the game was played
};

} // namespace selfplay
} // namespace alphazero

#endif // GAME_RECORD_H