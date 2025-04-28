// include/alphazero/games/go/go_rules.h
#ifndef GO_RULES_H
#define GO_RULES_H

#include <vector>
#include <unordered_set>
#include <functional>

namespace alphazero {
namespace go {

/**
 * @brief Group of connected stones
 */
struct StoneGroup {
    std::unordered_set<int> stones;
    std::unordered_set<int> liberties;
};

/**
 * @brief Rules implementation for the game of Go
 */
class GoRules {
public:
    /**
     * @brief Constructor
     * 
     * @param board_size Board size
     * @param chinese_rules Whether to use Chinese rules
     */
    GoRules(int board_size, bool chinese_rules = true);
    
    /**
     * @brief Set board accessor functions
     * 
     * @param get_stone Function to get stone at position
     * @param is_in_bounds Function to check if position is in bounds
     * @param get_adjacent_positions Function to get adjacent positions
     */
    void setBoardAccessor(
        std::function<int(int)> get_stone,
        std::function<bool(int)> is_in_bounds,
        std::function<std::vector<int>(int)> get_adjacent_positions);
    
    /**
     * @brief Check if a move would be suicide
     * 
     * @param action Action index
     * @param player Current player
     * @return true if move would be suicide, false otherwise
     */
    bool isSuicidalMove(int action, int player) const;
    
    /**
     * @brief Check if a move would violate the ko rule
     * 
     * @param action Action index
     * @param ko_point Current ko point
     * @return true if move would violate ko, false otherwise
     */
    bool isKoViolation(int action, int ko_point) const;
    
    /**
     * @brief Find all stone groups for a player
     * 
     * @param player Player (1 for black, 2 for white)
     * @return Vector of stone groups
     */
    std::vector<StoneGroup> findGroups(int player) const;
    
    /**
     * @brief Find liberties for a group of stones
     * 
     * @param stones Set of stone positions
     * @param liberties Output set for liberties
     */
    void findLiberties(std::unordered_set<int>& stones, std::unordered_set<int>& liberties) const;
    
    /**
     * @brief Calculate territory ownership
     * 
     * @return Vector of territory ownership (0 for neutral, 1 for black, 2 for white)
     */
    std::vector<int> getTerritoryOwnership() const;
    
    /**
     * @brief Calculate scores based on the current board position
     * 
     * @param captured_stones Array of captured stones count
     * @param komi Komi value
     * @return Pair of (black_score, white_score)
     */
    std::pair<float, float> calculateScores(const std::vector<int>& captured_stones, float komi) const;
    
    /**
     * @brief Flood fill to find territory
     * 
     * @param territory Territory map to update
     * @param pos Start position
     * @param territory_color Output territory color
     */
    void floodFillTerritory(std::vector<int>& territory, int pos, int& territory_color) const;
    
private:
    int board_size_;
    bool chinese_rules_;
    
    // Board accessor functions
    std::function<int(int)> get_stone_;
    std::function<bool(int)> is_in_bounds_;
    std::function<std::vector<int>(int)> get_adjacent_positions_;
};

} // namespace go
} // namespace alphazero

#endif // GO_RULES_H