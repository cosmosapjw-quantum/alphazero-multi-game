// gomoku_rules.h
#ifndef GOMOKU_RULES_H
#define GOMOKU_RULES_H

#include <vector>
#include <set>
#include <utility>
#include <functional>
#include "alphazero/core/utils/hash_specializations.h"

class GomokuRules {
public:
    GomokuRules(int board_size);
    
    // Set board accessor function
    void setBoardAccessor(std::function<bool(int, int)> is_bit_set_func,
                          std::function<int(int, int)> coords_to_action_func,
                          std::function<std::pair<int,int>(int)> action_to_coords_pair_func,
                          std::function<bool(int, int)> in_bounds_func);
    
    // Line and pattern detection
    bool is_five_in_a_row(int action, int player) const;
    bool check_line_for_five(int cell, int player, int p_idx) const;
    int count_direction(int x0, int y0, int dx, int dy, int p_idx) const;
    
    // Renju rule checks
    bool is_black_renju_forbidden(int action);
    bool renju_is_overline(int action) const;
    bool renju_double_four_or_more(int action) const;
    bool renju_double_three_or_more(int action) const;
    int renju_count_all_fours() const;
    int renju_count_all_threes(int action) const;
    
    // Omok rule checks
    bool is_black_omok_forbidden(int action);
    bool omok_is_overline(int action) const;
    bool omok_check_double_three_strict(int action) const;
    int count_open_threes_globally() const;
    
    // Pattern recognition (for neural network enhancement)
    std::vector<std::set<int>> get_three_patterns_for_action(int action) const;
    std::vector<std::set<int>> get_open_three_patterns_globally() const;
    
private:
    int board_size;
    
    // Board accessor functions (set by the GomokuState)
    std::function<bool(int, int)> is_bit_set;
    std::function<int(int, int)> coords_to_action;
    std::function<std::pair<int,int>(int)> action_to_coords_pair;
    std::function<bool(int, int)> in_bounds;
    
    // Pattern recognition helpers
    bool renju_is_three_shape(const std::vector<std::pair<int, int>>& segment) const;
    bool renju_is_four_shape(const std::vector<std::pair<int, int>>& segment) const;
    std::pair<bool, bool> ends_are_open(const std::vector<std::pair<int, int>>& segment) const;
    bool check_broken_four(const std::vector<std::pair<int, int>>& segment, bool front_open, bool back_open) const;
    bool simple_is_4_contiguous(const std::vector<std::pair<int, int>>& segment) const;
    std::set<int> positions_of_black(const std::vector<std::pair<int, int>>& segment) const;
    bool try_unify_four_shape(std::set<std::pair<std::set<int>, int>>& found_fours, 
                             const std::set<int>& new_fs, int size) const;
    bool try_unify_three_shape(std::set<std::set<int>>& found_threes, 
                              const std::set<int>& new_fs, int action) const;
    std::set<int> check_open_three_5slice(const std::vector<std::pair<int, int>>& cells_5) const;
    bool are_patterns_connected(const std::set<int>& pattern1, const std::set<int>& pattern2) const;
    
    // Enhanced double-three detection
    bool is_allowed_double_three(int action) const;
    bool can_make_straight_four(const std::set<int>& three_pattern) const;
    int count_straight_four_capable_threes(const std::vector<std::set<int>>& three_patterns) const;
    bool is_double_three_allowed_recursive(const std::vector<std::set<int>>& three_patterns, 
                                         int depth = 0, int max_depth = 3) const;
    bool is_straight_four(const std::set<int>& pattern) const;
    std::vector<int> find_three_to_four_placements(const std::set<int>& three_pattern) const;
    bool is_three_pattern(const std::vector<std::pair<int, int>>& segment, int action) const;
    bool is_four_pattern(const std::vector<std::pair<int, int>>& segment) const;
    
    // Utility methods
    std::vector<std::pair<int, int>> build_entire_line(int x0, int y0, int dx, int dy) const;
};

#endif // GOMOKU_RULES_H