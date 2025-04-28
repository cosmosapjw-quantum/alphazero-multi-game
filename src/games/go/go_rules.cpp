// src/games/go/go_rules.cpp
#include "alphazero/games/go/go_rules.h"
#include <queue>

namespace alphazero {
namespace go {

GoRules::GoRules(int board_size, bool chinese_rules)
    : board_size_(board_size),
      chinese_rules_(chinese_rules) {
    
    // Default implementations (will be replaced by setBoardAccessor)
    get_stone_ = [](int) { return 0; };
    is_in_bounds_ = [](int) { return false; };
    get_adjacent_positions_ = [](int) { return std::vector<int>(); };
}

void GoRules::setBoardAccessor(
    std::function<int(int)> get_stone,
    std::function<bool(int)> is_in_bounds,
    std::function<std::vector<int>(int)> get_adjacent_positions) {
    
    get_stone_ = get_stone;
    is_in_bounds_ = is_in_bounds;
    get_adjacent_positions_ = get_adjacent_positions;
}

bool GoRules::isSuicidalMove(int action, int player) const {
    if (!is_in_bounds_(action) || get_stone_(action) != 0) {
        return true;  // Invalid move
    }
    
    // Create a temporary board state for simulation
    std::vector<int> temp_board(board_size_ * board_size_, 0);
    
    // Copy current board state
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (is_in_bounds_(pos)) {
            temp_board[pos] = get_stone_(pos);
        }
    }
    
    // Place the stone
    temp_board[action] = player;
    
    // Define temporary accessor functions for the simulated board
    auto temp_get_stone = [&temp_board](int pos) { return temp_board[pos]; };
    
    // Check if any opponent group would be captured
    int opponent = (player == 1) ? 2 : 1;
    bool captures_opponent = false;
    
    // Check each adjacent position for opponent groups that might be captured
    for (int adj_pos : get_adjacent_positions_(action)) {
        if (temp_get_stone(adj_pos) == opponent) {
            // Check if this group would have liberties after the move
            std::unordered_set<int> group_stones;
            std::queue<int> queue;
            
            // Find connected opponent stones
            queue.push(adj_pos);
            group_stones.insert(adj_pos);
            std::vector<bool> visited(board_size_ * board_size_, false);
            visited[adj_pos] = true;
            
            while (!queue.empty()) {
                int current = queue.front();
                queue.pop();
                
                for (int next_pos : get_adjacent_positions_(current)) {
                    if (temp_get_stone(next_pos) == opponent && !visited[next_pos]) {
                        queue.push(next_pos);
                        visited[next_pos] = true;
                        group_stones.insert(next_pos);
                    }
                }
            }
            
            // Check if this group has any liberties
            bool has_liberty = false;
            for (int stone : group_stones) {
                for (int lib_pos : get_adjacent_positions_(stone)) {
                    if (temp_get_stone(lib_pos) == 0) {
                        has_liberty = true;
                        break;
                    }
                }
                if (has_liberty) break;
            }
            
            if (!has_liberty) {
                captures_opponent = true;
                break;
            }
        }
    }
    
    // If capturing opponent groups, the move is not suicidal
    if (captures_opponent) {
        return false;
    }
    
    // Check if the placed stone's group would have liberties
    std::unordered_set<int> own_stones;
    std::queue<int> queue;
    
    // Find connected stones of the same color
    queue.push(action);
    own_stones.insert(action);
    std::vector<bool> visited(board_size_ * board_size_, false);
    visited[action] = true;
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        
        for (int next_pos : get_adjacent_positions_(current)) {
            if (temp_get_stone(next_pos) == player && !visited[next_pos]) {
                queue.push(next_pos);
                visited[next_pos] = true;
                own_stones.insert(next_pos);
            }
        }
    }
    
    // Check if this group has any liberties
    for (int stone : own_stones) {
        for (int lib_pos : get_adjacent_positions_(stone)) {
            if (temp_get_stone(lib_pos) == 0) {
                return false;  // Has liberty, not suicide
            }
        }
    }
    
    // No liberties, move is suicidal
    return true;
}

bool GoRules::isKoViolation(int action, int ko_point) const {
    return action == ko_point;
}

std::vector<StoneGroup> GoRules::findGroups(int player) const {
    std::vector<StoneGroup> groups;
    std::vector<bool> visited(board_size_ * board_size_, false);
    
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!is_in_bounds_(pos) || get_stone_(pos) != player || visited[pos]) {
            continue;
        }
        
        // Found an unvisited stone of the player
        StoneGroup group;
        std::queue<int> queue;
        
        queue.push(pos);
        visited[pos] = true;
        group.stones.insert(pos);
        
        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            
            for (int adj : get_adjacent_positions_(current)) {
                if (get_stone_(adj) == player && !visited[adj]) {
                    queue.push(adj);
                    visited[adj] = true;
                    group.stones.insert(adj);
                }
            }
        }
        
        // Find liberties for the group
        findLiberties(group.stones, group.liberties);
        
        groups.push_back(group);
    }
    
    return groups;
}

void GoRules::findLiberties(std::unordered_set<int>& stones, std::unordered_set<int>& liberties) const {
    liberties.clear();
    
    // Check adjacent positions of all stones in the group
    for (int pos : stones) {
        for (int adj : get_adjacent_positions_(pos)) {
            if (get_stone_(adj) == 0) {
                liberties.insert(adj);
            }
        }
    }
}

std::vector<int> GoRules::getTerritoryOwnership() const {
    std::vector<int> territory(board_size_ * board_size_, 0);
    
    // Mark existing stones as territory
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!is_in_bounds_(pos)) continue;
        
        int stone = get_stone_(pos);
        if (stone != 0) {
            territory[pos] = stone;  // Owned by the player with a stone here
        }
    }
    
    // Find empty regions and determine ownership
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!is_in_bounds_(pos)) continue;
        
        if (get_stone_(pos) == 0 && territory[pos] == 0) {
            // Unmarked empty intersection, flood fill to find territory
            int territory_color = 0;
            floodFillTerritory(territory, pos, territory_color);
        }
    }
    
    return territory;
}

void GoRules::floodFillTerritory(std::vector<int>& territory, int pos, int& territory_color) const {
    // Start with neutral territory
    territory_color = 0;
    
    // Use queue for flood fill
    std::queue<int> queue;
    std::unordered_set<int> visited;
    
    queue.push(pos);
    visited.insert(pos);
    
    // Keep track of stones this territory touches
    bool touches_black = false;
    bool touches_white = false;
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        
        // Mark as part of this territory
        territory[current] = territory_color;
        
        for (int adj : get_adjacent_positions_(current)) {
            int stone = get_stone_(adj);
            
            if (stone == 0) {
                // Empty intersection, add to flood fill if not visited
                if (visited.find(adj) == visited.end()) {
                    queue.push(adj);
                    visited.insert(adj);
                }
            } else if (stone == 1) {
                // Touches black
                touches_black = true;
            } else if (stone == 2) {
                // Touches white
                touches_white = true;
            }
        }
    }
    
    // Determine territory color based on what stones it touches
    if (touches_black && !touches_white) {
        territory_color = 1;  // Black territory
    } else if (touches_white && !touches_black) {
        territory_color = 2;  // White territory
    } else {
        territory_color = 0;  // Neutral or contested
    }
    
    // Update territory with the determined color
    for (int visited_pos : visited) {
        territory[visited_pos] = territory_color;
    }
}

std::pair<float, float> GoRules::calculateScores(const std::vector<int>& captured_stones, float komi) const {
    float black_score = 0.0f;
    float white_score = 0.0f;
    
    // Count stones for Chinese rules
    if (chinese_rules_) {
        for (int pos = 0; pos < board_size_ * board_size_; pos++) {
            if (!is_in_bounds_(pos)) continue;
            
            int stone = get_stone_(pos);
            if (stone == 1) {
                black_score += 1.0f;
            } else if (stone == 2) {
                white_score += 1.0f;
            }
        }
    }
    
    // Count territory
    std::vector<int> territory = getTerritoryOwnership();
    
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!is_in_bounds_(pos)) continue;
        
        int owner = territory[pos];
        if (owner == 1) {
            black_score += 1.0f;
        } else if (owner == 2) {
            white_score += 1.0f;
        }
    }
    
    // Add captures for Japanese rules
    if (!chinese_rules_) {
        black_score += static_cast<float>(captured_stones[1]);
        white_score += static_cast<float>(captured_stones[2]);
    }
    
    // Add komi
    white_score += komi;
    
    return {black_score, white_score};
}

} // namespace go
} // namespace alphazero