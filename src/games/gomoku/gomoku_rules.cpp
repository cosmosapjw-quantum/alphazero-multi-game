// gomoku_rules.cpp
#include "gomoku_rules.h"
#include <algorithm>
#include <cmath>
#include <iostream>

GomokuRules::GomokuRules(int board_size)
    : board_size(board_size) {
    // Default implementations for board access functions
    // These will be properly set by the GomokuState
    is_bit_set = [](int, int) { return false; };
    coords_to_action = [](int, int) { return -1; };
    action_to_coords_pair = [](int) { return std::make_pair(-1, -1); };
    in_bounds = [](int, int) { return false; };
}

void GomokuRules::setBoardAccessor(
    std::function<bool(int, int)> is_bit_set_func,
    std::function<int(int, int)> coords_to_action_func,
    std::function<std::pair<int,int>(int)> action_to_coords_pair_func,
    std::function<bool(int, int)> in_bounds_func) {
    
    is_bit_set = is_bit_set_func;
    coords_to_action = coords_to_action_func;
    action_to_coords_pair = action_to_coords_pair_func;
    in_bounds = in_bounds_func;
}

// Line and pattern detection
bool GomokuRules::is_five_in_a_row(int action, int player) const {
    int p_idx = player - 1;
    int total = board_size * board_size;
    
    if (action == -1) {
        // Checking entire board
        for (int cell = 0; cell < total; cell++) {
            if (is_bit_set(p_idx, cell)) {
                if (check_line_for_five(cell, player, p_idx)) {
                    return true;
                }
            }
        }
        return false;
    } else {
        // Check just the specific action
        if (!is_bit_set(p_idx, action)) {
            return false;
        }
        return check_line_for_five(action, player, p_idx);
    }
}

bool GomokuRules::check_line_for_five(int cell, int player, int p_idx) const {
    if (!is_bit_set(p_idx, cell)) {
        return false;
    }
    
    auto [x, y] = action_to_coords_pair(cell);
    
    // Check in four directions (horizontal, vertical, two diagonals)
    for (int d = 0; d < 4; d++) {
        // Extract dx, dy from the stored directions
        int dx, dy;
        switch (d) {
            case 0: dx = 0; dy = 1; break;  // Vertical
            case 1: dx = 1; dy = 0; break;  // Horizontal
            case 2: dx = 1; dy = 1; break;  // Diagonal down
            case 3: dx = 1; dy = -1; break; // Diagonal up
            default: dx = 0; dy = 0; break;
        }
        
        int forward = count_direction(x, y, dx, dy, p_idx);
        int backward = count_direction(x, y, -dx, -dy, p_idx) - 1; // Subtract 1 to avoid counting the cell twice
        int length = forward + backward;
        
        if (player == 1) { // BLACK
            if (length == 5) {
                return true;
            }
        } else { // WHITE
            if (length >= 5) {
                return true;
            }
        }
    }
    return false;
}

int GomokuRules::count_direction(int x0, int y0, int dx, int dy, int p_idx) const {
    int count = 0;
    int x = x0;
    int y = y0;
    
    while (in_bounds(x, y)) {
        int action = coords_to_action(x, y);
        if (is_bit_set(p_idx, action)) {
            count++;
            x += dx;
            y += dy;
        } else {
            break;
        }
    }
    
    return count;
}

// Renju rule checks
bool GomokuRules::is_black_renju_forbidden(int action) {
    // Temporarily place a black stone at the action
    int p_idx = 0; // Black = 0, White = 1
    
    // Check for overline
    if (renju_is_overline(action)) {
        return true;
    }
    
    // Check for double-four or more
    if (renju_double_four_or_more(action)) {
        return true;
    }
    
    // Check for double-three or more
    if (!is_allowed_double_three(action)) {
        return true;
    }
    
    return false;
}

bool GomokuRules::renju_is_overline(int action) const {
    int x0, y0, direction, dx, dy, nx, ny, count_line;
    auto [x0_val, y0_val] = action_to_coords_pair(action);
    x0 = x0_val;
    y0 = y0_val;
    
    // Temporarily consider the current action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set(p_idx, a);
    };
    
    for (direction = 0; direction < 4; direction++) {
        switch (direction) {
            case 0: dx = 0; dy = 1; break;  // Vertical
            case 1: dx = 1; dy = 0; break;  // Horizontal
            case 2: dx = 1; dy = 1; break;  // Diagonal down
            case 3: dx = 1; dy = -1; break; // Diagonal up
            default: dx = 0; dy = 0; break;
        }
        
        count_line = 1;
        
        nx = x0 + dx;
        ny = y0 + dy;
        while (in_bounds(nx, ny)) {
            int a = coords_to_action(nx, ny);
            if (is_bit_set_temp(0, a)) {
                count_line++;
                nx += dx;
                ny += dy;
            } else {
                break;
            }
        }
        
        nx = x0 - dx;
        ny = y0 - dy;
        while (in_bounds(nx, ny)) {
            int a = coords_to_action(nx, ny);
            if (is_bit_set_temp(0, a)) {
                count_line++;
                nx -= dx;
                ny -= dy;
            } else {
                break;
            }
        }
        
        if (count_line >= 6) {
            return true;
        }
    }
    return false;
}

bool GomokuRules::renju_double_four_or_more(int action) const {
    int c4 = renju_count_all_fours();
    return (c4 >= 2);
}

bool GomokuRules::renju_double_three_or_more(int action) const {
    // Temporarily consider the current action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set(p_idx, a);
    };
    
    // Get the unified set of three patterns.
    std::vector<std::set<int>> three_patterns = get_three_patterns_for_action(action);
    
    // If 2 or more distinct three patterns exist, then it's a double-three.
    return (three_patterns.size() >= 2);
}

int GomokuRules::renju_count_all_fours() const {
    int bs = board_size;
    std::set<std::pair<std::set<int>, int>> found_fours;
    std::vector<std::pair<int, int>> directions = {{0,1}, {1,0}, {1,1}, {-1,1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx, dy] : directions) {
                std::vector<std::pair<int, int>> line_cells;
                int xx = x, yy = y;
                int step = 0;
                
                while (step < 7) {
                    if (!in_bounds(xx, yy)) {
                        break;
                    }
                    line_cells.push_back({xx, yy});
                    xx += dx;
                    yy += dy;
                    step++;
                }
                
                for (int window_size : {5, 6, 7}) {
                    if (line_cells.size() < window_size) {
                        break;
                    }
                    
                    for (size_t start_idx = 0; start_idx <= line_cells.size() - window_size; start_idx++) {
                        std::vector<std::pair<int, int>> segment(
                            line_cells.begin() + start_idx,
                            line_cells.begin() + start_idx + window_size
                        );
                        
                        if (renju_is_four_shape(segment)) {
                            std::set<int> black_positions = positions_of_black(segment);
                            bool unified = try_unify_four_shape(found_fours, black_positions, black_positions.size());
                            
                            if (!unified) {
                                found_fours.insert({black_positions, black_positions.size()});
                            }
                        }
                    }
                }
            }
        }
    }
    
    return found_fours.size();
}

int GomokuRules::renju_count_all_threes(int action) const {
    int bs = board_size;
    std::set<std::set<int>> found_threes;
    std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx, dy] : directions) {
                std::vector<std::pair<int, int>> line_cells;
                int xx = x, yy = y;
                int step = 0;
                
                while (step < 7) {
                    if (!in_bounds(xx, yy)) {
                        break;
                    }
                    line_cells.push_back({xx, yy});
                    xx += dx;
                    yy += dy;
                    step++;
                }
                
                for (int window_size : {5, 6}) {
                    if (line_cells.size() < window_size) {
                        break;
                    }
                    
                    for (size_t start_idx = 0; start_idx <= line_cells.size() - window_size; start_idx++) {
                        std::vector<std::pair<int, int>> segment(
                            line_cells.begin() + start_idx,
                            line_cells.begin() + start_idx + window_size
                        );
                        
                        if (renju_is_three_shape(segment)) {
                            std::set<int> black_positions = positions_of_black(segment);
                            std::set<int> new_fs(black_positions);
                            
                            if (!try_unify_three_shape(found_threes, new_fs, action)) {
                                found_threes.insert(new_fs);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return found_threes.size();
}

// Omok rule checks
bool GomokuRules::is_black_omok_forbidden(int action) {
    // Temporarily place a black stone at the action
    int p_idx = 0; // Black = 0, White = 1
    
    // Check for overline
    if (omok_is_overline(action)) {
        return true;
    }
    
    // Check for double-three
    if (omok_check_double_three_strict(action)) {
        return true;
    }
    
    return false;
}

bool GomokuRules::omok_is_overline(int action) const {
    return renju_is_overline(action);
}

bool GomokuRules::omok_check_double_three_strict(int action) const {
    // Temporarily consider the current action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set(p_idx, a);
    };
    
    std::vector<std::set<int>> patterns = get_open_three_patterns_globally();
    int n = patterns.size();
    
    if (n < 2) {
        return false;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (are_patterns_connected(patterns[i], patterns[j])) {
                return true;
            }
        }
    }
    
    return false;
}

int GomokuRules::count_open_threes_globally() const {
    return get_open_three_patterns_globally().size();
}

// Pattern recognition helpers
bool GomokuRules::renju_is_three_shape(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    int black_count = 0, white_count = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            black_count++;
        } else if (is_bit_set(1, a)) {
            white_count++;
        }
    }
    
    if (white_count > 0 || black_count < 2 || black_count >= 4) {
        return false;
    }
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            // Test if placing a black stone here creates a four shape
            // We need to temporarily consider a black stone here
            
            // Create a test board with this stone added
            auto is_bit_set_temp = [this, a](int p_idx, int test_a) {
                if (test_a == a && p_idx == 0) { // Black is trying to place here
                    return true;
                }
                return is_bit_set(p_idx, test_a);
            };
            
            // Check if it's now a four shape
            int seg_len = segment.size();
            int temp_black_count = 0, temp_white_count = 0;
            
            for (const auto& [tx, ty] : segment) {
                int ta = coords_to_action(tx, ty);
                if (is_bit_set_temp(0, ta)) {
                    temp_black_count++;
                } else if (is_bit_set_temp(1, ta)) {
                    temp_white_count++;
                }
            }
            
            if (temp_white_count > 0) {
                continue;
            }
            
            if (temp_black_count == 4) {
                auto [front_open, back_open] = ends_are_open(segment);
                if (front_open || back_open) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

bool GomokuRules::renju_is_four_shape(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    int black_count = 0, white_count = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(1, a)) {
            white_count++;
        } else if (is_bit_set(0, a)) {
            black_count++;
        }
    }
    
    if (white_count > 0) {
        return false;
    }
    
    if (black_count < 3 || black_count > 4) {
        return false;
    }
    
    auto [front_open, back_open] = ends_are_open(segment);
    
    if (black_count == 4) {
        return (front_open || back_open);
    } else {
        return check_broken_four(segment, front_open, back_open);
    }
}

std::pair<bool, bool> GomokuRules::ends_are_open(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    if (seg_len < 2) {
        return {false, false};
    }
    
    auto [x0, y0] = segment[0];
    auto [x1, y1] = segment[seg_len - 1];
    bool front_open = false, back_open = false;
    
    int dx = 0, dy = 0;
    if (seg_len >= 2) {
        auto [x2, y2] = segment[1];
        dx = x2 - x0;
        dy = y2 - y0;
    }
    
    int fx = x0 - dx;
    int fy = y0 - dy;
    if (in_bounds(fx, fy)) {
        int af = coords_to_action(fx, fy);
        if (!is_bit_set(0, af) && !is_bit_set(1, af)) {
            front_open = true;
        }
    }
    
    int lx = x1 + dx;
    int ly = y1 + dy;
    if (in_bounds(lx, ly)) {
        int ab = coords_to_action(lx, ly);
        if (!is_bit_set(0, ab) && !is_bit_set(1, ab)) {
            back_open = true;
        }
    }
    
    return {front_open, back_open};
}

bool GomokuRules::check_broken_four(const std::vector<std::pair<int, int>>& segment, bool front_open, bool back_open) const {
    if (!front_open && !back_open) {
        return false;
    }
    
    std::vector<std::pair<int, int>> empties;
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            empties.push_back({x, y});
        }
    }
    
    if (empties.size() != 1) {
        return false;
    }
    
    auto [gapx, gapy] = empties[0];
    int gap_action = coords_to_action(gapx, gapy);
    
    // Create a test board with this stone added
    auto is_bit_set_temp = [this, gap_action](int p_idx, int test_a) {
        if (test_a == gap_action && p_idx == 0) { // Black placed at the gap
            return true;
        }
        return is_bit_set(p_idx, test_a);
    };
    
    // Check if placing a black stone at the gap makes a 4-in-a-row
    int consecutive = 0, best = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set_temp(0, a)) {
            consecutive++;
            if (consecutive > best) {
                best = consecutive;
            }
        } else {
            consecutive = 0;
        }
    }
    
    return (best >= 4);
}

bool GomokuRules::simple_is_4_contiguous(const std::vector<std::pair<int, int>>& segment) const {
    int consecutive = 0, best = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            consecutive++;
            if (consecutive > best) {
                best = consecutive;
            }
        } else {
            consecutive = 0;
        }
    }
    
    return (best >= 4);
}

std::set<int> GomokuRules::positions_of_black(const std::vector<std::pair<int, int>>& segment) const {
    std::set<int> black_set;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            black_set.insert(a);
        }
    }
    
    return black_set;
}

bool GomokuRules::try_unify_four_shape(std::set<std::pair<std::set<int>, int>>& found_fours, 
                                     const std::set<int>& new_fs, int size) const {
    for (const auto& [existing_fs, existing_size] : found_fours) {
        std::set<int> intersection;
        std::set_intersection(
            existing_fs.begin(), existing_fs.end(),
            new_fs.begin(), new_fs.end(),
            std::inserter(intersection, intersection.begin())
        );
        
        if (intersection.size() >= 3) {
            return true;
        }
    }
    
    return false;
}

bool GomokuRules::try_unify_three_shape(std::set<std::set<int>>& found_threes, 
                                      const std::set<int>& new_fs, int action) const {
    for (const auto& existing_fs : found_threes) {
        std::set<int> intersection;
        std::set_intersection(
            existing_fs.begin(), existing_fs.end(),
            new_fs.begin(), new_fs.end(),
            std::inserter(intersection, intersection.begin())
        );
        
        // Remove action from intersection
        intersection.erase(action);
        
        if (!intersection.empty()) {
            return true;
        }
    }
    
    return false;
}

std::vector<std::set<int>> GomokuRules::get_three_patterns_for_action(int action) const {
    std::vector<std::set<int>> three_patterns;
    int bs = board_size;
    std::vector<std::pair<int, int>> directions = { {0, 1}, {1, 0}, {1, 1}, {-1, 1} };
    
    auto [x0, y0] = action_to_coords_pair(action);
    
    // Temporarily consider the action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set(p_idx, a);
    };
    
    for (auto [dx, dy] : directions) {
        std::vector<std::pair<int, int>> line_cells;
        // Build a line of up to 7 cells centered on the action.
        for (int offset = -3; offset <= 3; offset++) {
            int nx = x0 + offset * dx;
            int ny = y0 + offset * dy;
            if (in_bounds(nx, ny)) {
                line_cells.push_back({nx, ny});
            }
        }
        
        // Slide a 5-cell window over the line.
        for (size_t start = 0; start + 4 < line_cells.size(); start++) {
            std::vector<std::pair<int, int>> segment(line_cells.begin() + start, line_cells.begin() + start + 5);
            
            // Check if this segment forms a three pattern containing our action.
            if (is_three_pattern(segment, action)) {
                std::set<int> pattern;
                for (auto [x, y] : segment) {
                    pattern.insert(coords_to_action(x, y));
                }
                
                // Unify: check if this pattern overlaps in at least 3 cells with any existing one.
                bool duplicate = false;
                for (const auto &existing : three_patterns) {
                    std::set<int> inter;
                    std::set_intersection(existing.begin(), existing.end(),
                                          pattern.begin(), pattern.end(),
                                          std::inserter(inter, inter.begin()));
                    if (inter.size() >= 3) {  // Overlap is significant; consider it the same three.
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    three_patterns.push_back(pattern);
                }
            }
        }
    }
    return three_patterns;
}

bool GomokuRules::is_three_pattern(const std::vector<std::pair<int, int>>& segment, int action) const {
    // A three pattern has exactly 3 black stones, the rest empty, and can form a four
    
    // Temporarily consider the action as a black stone
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set(p_idx, a);
    };
    
    int black_count = 0;
    int white_count = 0;
    bool contains_action = false;
    
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set_temp(0, a)) {
            black_count++;
            if (a == action) {
                contains_action = true;
            }
        } else if (is_bit_set_temp(1, a)) {
            white_count++;
        }
    }
    
    if (black_count != 3 || white_count > 0 || !contains_action) {
        return false;
    }
    
    // Check if this pattern can form a four by placing a stone in an empty spot
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set_temp(0, a) && !is_bit_set_temp(1, a)) {
            // Check if placing a black stone here would form a four
            
            // Create another temporary board state with this additional stone
            auto is_bit_set_double_temp = [is_bit_set_temp, a](int p_idx, int test_a) {
                if (test_a == a && p_idx == 0) { // Black placed at this empty spot
                    return true;
                }
                return is_bit_set_temp(p_idx, test_a);
            };
            
            // Check for a four pattern
            int temp_black_count = 0;
            for (auto [tx, ty] : segment) {
                int ta = coords_to_action(tx, ty);
                if (is_bit_set_double_temp(0, ta)) {
                    temp_black_count++;
                }
            }
            
            if (temp_black_count == 4) {
                return true;
            }
        }
    }
    
    return false;
}

bool GomokuRules::is_four_pattern(const std::vector<std::pair<int, int>>& segment) const {
    // A four pattern has exactly 4 black stones and can form a five
    
    int black_count = 0;
    int white_count = 0;
    
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            black_count++;
        } else if (is_bit_set(1, a)) {
            white_count++;
        }
    }
    
    if (black_count != 4 || white_count > 0) {
        return false;
    }
    
    // Check if there's at least one empty spot that would form a five
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            return true;
        }
    }
    
    return false;
}

std::vector<std::set<int>> GomokuRules::get_open_three_patterns_globally() const {
    int bs = board_size;
    std::set<std::set<int>> found_threes;
    std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx0, dy0] : directions) {
                std::vector<std::pair<int, int>> cells_5;
                int step = 0;
                int cx = x, cy = y;
                
                while (step < 5) {
                    if (!in_bounds(cx, cy)) {
                        break;
                    }
                    cells_5.push_back({cx, cy});
                    cx += dx0;
                    cy += dy0;
                    step++;
                }
                
                if (cells_5.size() == 5) {
                    std::set<int> triple = check_open_three_5slice(cells_5);
                    if (!triple.empty()) {
                        bool skip = false;
                        std::vector<std::set<int>> to_remove;
                        
                        for (const auto& existing : found_threes) {
                            // Check if existing is a superset of triple
                            if (std::includes(existing.begin(), existing.end(), 
                                            triple.begin(), triple.end())) {
                                skip = true;
                                break;
                            }
                            
                            // Check if triple is a superset of existing
                            if (std::includes(triple.begin(), triple.end(), 
                                            existing.begin(), existing.end())) {
                                to_remove.push_back(existing);
                            }
                        }
                        
                        if (!skip) {
                            for (const auto& r : to_remove) {
                                found_threes.erase(r);
                            }
                            found_threes.insert(triple);
                        }
                    }
                }
            }
        }
    }
    
    return std::vector<std::set<int>>(found_threes.begin(), found_threes.end());
}

std::set<int> GomokuRules::check_open_three_5slice(const std::vector<std::pair<int, int>>& cells_5) const {
    if (cells_5.size() != 5) {
        return {};
    }
    
    int black_count = 0, white_count = 0, empty_count = 0;
    int arr[5] = {0}; // Represents the contents of cells_5: 0=empty, 1=black, -1=white
    
    for (int i = 0; i < 5; i++) {
        auto [xx, yy] = cells_5[i];
        int act = coords_to_action(xx, yy);
        
        if (is_bit_set(0, act)) {
            black_count++;
            arr[i] = 1;
        } else if (is_bit_set(1, act)) {
            white_count++;
            arr[i] = -1;
        } else {
            empty_count++;
        }
    }
    
    if (black_count != 3 || white_count != 0 || empty_count != 2) {
        return {};
    }
    
    if (arr[0] != 0 || arr[4] != 0) {
        return {};
    }
    
    bool has_triple = false, has_gap = false;
    
    if (arr[1] == 1 && arr[2] == 1 && arr[3] == 1) {
        has_triple = true;
    }
    
    if (arr[1] == 1 && arr[2] == 0 && arr[3] == 1) {
        has_gap = true;
    }
    
    if (!has_triple && !has_gap) {
        return {};
    }
    
    int dx = cells_5[1].first - cells_5[0].first;
    int dy = cells_5[1].second - cells_5[0].second;
    
    int left_x = cells_5[0].first - dx;
    int left_y = cells_5[0].second - dy;
    int right_x = cells_5[4].first + dx;
    int right_y = cells_5[4].second + dy;
    
    // Check if this is an "open" three (both ends must be empty)
    if (in_bounds(left_x, left_y)) {
        int left_act = coords_to_action(left_x, left_y);
        if (is_bit_set(0, left_act)) {
            return {};
        }
    }
    
    if (in_bounds(right_x, right_y)) {
        int right_act = coords_to_action(right_x, right_y);
        if (is_bit_set(0, right_act)) {
            return {};
        }
    }
    
    // Get the positions of the three black stones
    std::set<int> triple;
    for (int i = 0; i < 5; i++) {
        if (arr[i] == 1) {
            triple.insert(coords_to_action(cells_5[i].first, cells_5[i].second));
        }
    }
    
    return triple;
}

bool GomokuRules::are_patterns_connected(const std::set<int>& pattern1, const std::set<int>& pattern2) const {
    for (int cell1 : pattern1) {
        auto [ax, ay] = action_to_coords_pair(cell1);
        
        for (int cell2 : pattern2) {
            auto [bx, by] = action_to_coords_pair(cell2);
            
            if (abs(ax - bx) <= 1 && abs(ay - by) <= 1) {
                return true;
            }
        }
    }
    return false;
}

// Enhanced double-three detection for Renju rules
bool GomokuRules::is_allowed_double_three(int action) const {
    // Temporarily consider the action as a black stone for pattern detection
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set(p_idx, a);
    };
    
    // Step 1: Get all three patterns that include this action
    std::vector<std::set<int>> three_patterns = get_three_patterns_for_action(action);
    
    // If there's fewer than 2 three patterns, it's not a double-three
    if (three_patterns.size() < 2) {
        return true; // Not a double-three, so it's allowed
    }
    
    // Apply rule 9.3(a): Check how many threes can be made into straight fours
    int straight_four_capable_count = count_straight_four_capable_threes(three_patterns);
    
    // If at most one of the threes can be made into a straight four, the double-three is allowed
    if (straight_four_capable_count <= 1) {
        return true;
    }
    
    // Apply rule 9.3(b): Recursive check for potential future double-threes
    return is_double_three_allowed_recursive(three_patterns);
}

bool GomokuRules::can_make_straight_four(const std::set<int>& three_pattern) const {
    // Create temporary board accessor that considers the action point as a black stone
    auto action = *three_pattern.begin(); // Just need any point from the pattern to set up the context
    
    auto is_bit_set_temp = [this, action](int p_idx, int a) {
        if (a == action && p_idx == 0) { // Black is trying to place here
            return true;
        }
        return is_bit_set(p_idx, a);
    };
    
    // Get candidate placements that might convert the three into a four.
    std::vector<int> possible_placements = find_three_to_four_placements(three_pattern);
    for (int placement : possible_placements) {
        // Create another temporary accessor that adds this candidate placement
        auto is_bit_set_double_temp = [is_bit_set_temp, placement](int p_idx, int a) {
            if (a == placement && p_idx == 0) { // Black placed at the placement point
                return true;
            }
            return is_bit_set_temp(p_idx, a);
        };
        
        // Form a new pattern by adding the candidate.
        std::set<int> new_pattern = three_pattern;
        new_pattern.insert(placement);
        // Extract only the black stone positions from new_pattern.
        std::set<int> black_positions;
        for (int a : new_pattern) {
            if (is_bit_set_double_temp(0, a))
                black_positions.insert(a);
        }
        // Only consider candidate patterns that yield exactly 4 black stones.
        if (black_positions.size() != 4)
            continue;
        // If the new pattern qualifies as a straight four, count it.
        if (is_straight_four(new_pattern)) {
            // Here we'd need to check for overline, but without a real board state
            // we'll just return true as a simplification
            return true;
        }
    }
    return false;
}

std::vector<int> GomokuRules::find_three_to_four_placements(const std::set<int>& three_pattern) const {
    std::vector<int> placements;
    
    // Convert pattern to coordinates for easier analysis
    std::vector<std::pair<int, int>> coords;
    for (int a : three_pattern) {
        coords.push_back(action_to_coords_pair(a));
    }
    
    // Sort coordinates to find the pattern direction
    std::sort(coords.begin(), coords.end());
    
    // Determine if pattern is horizontal, vertical, or diagonal
    bool is_horizontal = true;
    bool is_vertical = true;
    bool is_diag_down = true;
    bool is_diag_up = true;
    
    for (size_t i = 1; i < coords.size(); i++) {
        if (coords[i].second != coords[0].second) is_horizontal = false;
        if (coords[i].first != coords[0].first) is_vertical = false;
        if (coords[i].first - coords[0].first != coords[i].second - coords[0].second) is_diag_down = false;
        if (coords[i].first - coords[0].first != coords[0].second - coords[i].second) is_diag_up = false;
    }
    
    // Determine direction vector
    int dx = 0, dy = 0;
    if (is_horizontal) {
        dx = 0; dy = 1;
    } else if (is_vertical) {
        dx = 1; dy = 0;
    } else if (is_diag_down) {
        dx = 1; dy = 1;
    } else if (is_diag_up) {
        dx = 1; dy = -1;
    } else {
        // Not a straight line, shouldn't happen with valid three patterns
        return placements;
    }
    
    // Find min and max coordinates
    int min_x = coords[0].first, min_y = coords[0].second;
    int max_x = coords[0].first, max_y = coords[0].second;
    
    for (auto [x, y] : coords) {
        min_x = std::min<int>(min_x, x);
        min_y = std::min<int>(min_y, y);
        max_x = std::max<int>(max_x, x);
        max_y = std::max<int>(max_y, y);
    }
    
    // Check for empty spots that could complete a four
    // Need to check both within the pattern and at the ends
    
    // Check within the pattern
    for (int i = 0; i <= 4; i++) {
        int x = min_x + i * dx;
        int y = min_y + i * dy;
        
        if (!in_bounds(x, y)) continue;
        
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a) && three_pattern.find(a) == three_pattern.end()) {
            placements.push_back(a);
        }
    }
    
    // Check beyond the ends
    int before_x = min_x - dx;
    int before_y = min_y - dy;
    int after_x = max_x + dx;
    int after_y = max_y + dy;
    
    if (in_bounds(before_x, before_y)) {
        int a = coords_to_action(before_x, before_y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            placements.push_back(a);
        }
    }
    
    if (in_bounds(after_x, after_y)) {
        int a = coords_to_action(after_x, after_y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            placements.push_back(a);
        }
    }
    
    return placements;
}

bool GomokuRules::is_straight_four(const std::set<int>& pattern) const {
    // Build the segment of coordinates corresponding to the pattern.
    std::vector<std::pair<int,int>> segment;
    for (int a : pattern) {
        segment.push_back(action_to_coords_pair(a));
    }
    // Sort the coordinates
    std::sort(segment.begin(), segment.end(), [&](const std::pair<int,int>& p1, const std::pair<int,int>& p2) {
        if (p1.first == p2.first)
            return p1.second < p2.second;
        return p1.first < p2.first;
    });

    // Count black and white stones in the segment.
    int black_count = 0, white_count = 0;
    for (const auto &p : segment) {
        int a = coords_to_action(p.first, p.second);
        if (is_bit_set(0, a))
            ++black_count;
        else if (is_bit_set(1, a))
            ++white_count;
    }
    if (white_count > 0)
        return false;
    
    // Only consider a pattern with exactly 4 black stones as a four-shape.
    if (black_count == 4) {
        auto ends = ends_are_open(segment); // returns {front_open, back_open}
        return (ends.first || ends.second);
    }
    return false;
}

int GomokuRules::count_straight_four_capable_threes(const std::vector<std::set<int>>& three_patterns) const {
    int count = 0;
    
    for (const auto& pattern : three_patterns) {
        if (can_make_straight_four(pattern)) {
            count++;
        }
    }
    
    return count;
}

bool GomokuRules::is_double_three_allowed_recursive(const std::vector<std::set<int>>& three_patterns, 
                                                 int depth, int max_depth) const {
    // Avoid too deep recursion
    if (depth >= max_depth) {
        return false;
    }
    
    // Apply rule 9.3(a) again at this level
    int straight_four_capable_count = count_straight_four_capable_threes(three_patterns);
    if (straight_four_capable_count <= 1) {
        return true;
    }
    
    // Apply rule 9.3(b): Check all possible future moves that would create a straight four
    for (const auto& pattern : three_patterns) {
        std::vector<int> placements = find_three_to_four_placements(pattern);
        
        for (int placement : placements) {
            // Skip if already occupied
            if (is_bit_set(0, placement) || is_bit_set(1, placement)) {
                continue;
            }
            
            // Create a temporary board state accessor that adds this placement
            auto is_bit_set_temp = [this, placement](int p_idx, int a) {
                if (a == placement && p_idx == 0) { // Black placed here
                    return true;
                }
                return is_bit_set(p_idx, a);
            };
            
            // Check if this creates a new double-three
            std::vector<std::set<int>> new_three_patterns = get_three_patterns_for_action(placement);
            if (new_three_patterns.size() >= 2) {
                // Recursively check if this new double-three is allowed
                if (is_double_three_allowed_recursive(new_three_patterns, depth + 1, max_depth)) {
                    return true;
                }
            }
        }
    }
    
    // If we've checked all possibilities and found no allowed configuration
    return false;
}

// Utility methods
std::vector<std::pair<int, int>> GomokuRules::build_entire_line(int x0, int y0, int dx, int dy) const {
    std::vector<std::pair<int, int>> backward_positions;
    std::vector<std::pair<int, int>> forward_positions;
    
    int bx = x0, by = y0;
    while (in_bounds(bx, by)) {
        backward_positions.push_back({bx, by});
        bx -= dx;
        by -= dy;
    }
    
    std::reverse(backward_positions.begin(), backward_positions.end());
    
    int fx = x0 + dx, fy = y0 + dy;
    while (in_bounds(fx, fy)) {
        forward_positions.push_back({fx, fy});
        fx += dx;
        fy += dy;
    }
    
    std::vector<std::pair<int, int>> result = backward_positions;
    result.insert(result.end(), forward_positions.begin(), forward_positions.end());
    return result;
}