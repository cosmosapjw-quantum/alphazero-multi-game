// gomoku_state.cpp
#include "alphazero/games/gomoku/gomoku_state.h"
#include <algorithm>
#include <random>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <string>
#include <sstream>

namespace alphazero {
namespace gomoku {

// Constructor with initialization of caching fields
GomokuState::GomokuState(int board_size, bool use_renju, bool use_omok, int seed, bool use_pro_long_opening) 
    : IGameState(core::GameType::GOMOKU),
      board_size(board_size),
      current_player(BLACK),
      action(-1),
      use_renju(use_renju),
      use_omok(use_omok),
      use_pro_long_opening(use_pro_long_opening),
      black_first_stone(-1),
      valid_moves_dirty(true),
      cached_winner(0),
      winner_check_dirty(true),
      hash_signature(0),
      hash_dirty(true),
      move_history() {
    
    int total_cells = board_size * board_size;
    num_words = (total_cells + 63) / 64;
    
    // Initialize bitboards with zeros
    player_bitboards.resize(2, std::vector<uint64_t>(num_words, 0));
    
    // Directions for line scanning (dx,dy pairs)
    dirs[0] = 0;   // dx=0
    dirs[1] = 1;   // dy=1   (vertical)
    dirs[2] = 1;   // dx=1
    dirs[3] = 0;   // dy=0   (horizontal)
    dirs[4] = 1;   // dx=1
    dirs[5] = 1;   // dy=1   (diag-down)
    dirs[6] = -1;  // dx=-1
    dirs[7] = 1;   // dy=1   (diag-up)
    
    // Create rules instance
    rules = std::make_shared<GomokuRules>(board_size);
    
    // Set up board access functions for rules
    rules->setBoardAccessor(
        [this](int p_idx, int a) { return this->is_bit_set(p_idx, a); },
        [this](int x, int y) { return this->coords_to_action(x, y); },
        [this](int a) { return this->action_to_coords_pair(a); },
        [this](int x, int y) { return this->in_bounds(x, y); }
    );
    
    // Optional seed initialization
    if (seed != 0) {
        std::srand(seed);
    } else {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }
}

// Copy constructor with cache preservation
GomokuState::GomokuState(const GomokuState& other) 
    : IGameState(core::GameType::GOMOKU),
      board_size(other.board_size),
      current_player(other.current_player),
      player_bitboards(other.player_bitboards),
      num_words(other.num_words),
      action(other.action),
      use_renju(other.use_renju),
      use_omok(other.use_omok),
      use_pro_long_opening(other.use_pro_long_opening),
      black_first_stone(other.black_first_stone),
      cached_valid_moves(other.cached_valid_moves),
      valid_moves_dirty(other.valid_moves_dirty),
      cached_winner(other.cached_winner),
      winner_check_dirty(other.winner_check_dirty),
      hash_signature(other.hash_signature),
      hash_dirty(other.hash_dirty),
      move_history(other.move_history) {
    
    // Copy the directions array
    for (int i = 0; i < 8; i++) {
        dirs[i] = other.dirs[i];
    }
    
    // Create new rules instance
    rules = std::make_shared<GomokuRules>(board_size);
    
    // Set up board access functions for rules
    rules->setBoardAccessor(
        [this](int p_idx, int a) { return this->is_bit_set(p_idx, a); },
        [this](int x, int y) { return this->coords_to_action(x, y); },
        [this](int a) { return this->action_to_coords_pair(a); },
        [this](int x, int y) { return this->in_bounds(x, y); }
    );
}

// Assignment operator implementation
GomokuState& GomokuState::operator=(const GomokuState& other) {
    if (this != &other) {
        board_size = other.board_size;
        current_player = other.current_player;
        player_bitboards = other.player_bitboards;
        num_words = other.num_words;
        action = other.action;
        use_renju = other.use_renju;
        use_omok = other.use_omok;
        use_pro_long_opening = other.use_pro_long_opening;
        black_first_stone = other.black_first_stone;
        cached_valid_moves = other.cached_valid_moves;
        valid_moves_dirty = other.valid_moves_dirty;
        cached_winner = other.cached_winner;
        winner_check_dirty = other.winner_check_dirty;
        hash_signature = other.hash_signature;
        hash_dirty = other.hash_dirty;
        move_history = other.move_history;
        
        // Copy the directions array
        for (int i = 0; i < 8; i++) {
            dirs[i] = other.dirs[i];
        }
        
        // Create new rules instance
        rules = std::make_shared<GomokuRules>(board_size);
        
        // Set up board access functions for rules
        rules->setBoardAccessor(
            [this](int p_idx, int a) { return this->is_bit_set(p_idx, a); },
            [this](int x, int y) { return this->coords_to_action(x, y); },
            [this](int a) { return this->action_to_coords_pair(a); },
            [this](int x, int y) { return this->in_bounds(x, y); }
        );
    }
    return *this;
}

// Create a deep copy
GomokuState GomokuState::copy() const {
    return GomokuState(*this);
}

// Implement IGameState interface

std::vector<int> GomokuState::getLegalMoves() const {
    return get_valid_moves();  // Use existing method
}

bool GomokuState::isLegalMove(int action) const {
    return is_move_valid(action);  // Use existing method
}

void GomokuState::makeMove(int action) {
    make_move(action, current_player);  // Use existing method
}

bool GomokuState::undoMove() {
    if (move_history.empty()) {
        return false;
    }
    
    int last_move = move_history.back();
    try {
        undo_move(last_move);
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool GomokuState::isTerminal() const {
    return is_terminal();  // Use existing method
}

core::GameResult GomokuState::getGameResult() const {
    int winner = get_winner();
    if (winner == 0) {
        if (is_stalemate()) {
            return core::GameResult::DRAW;
        }
        return core::GameResult::ONGOING;
    } else if (winner == 1) {
        return core::GameResult::WIN_PLAYER1;
    } else {
        return core::GameResult::WIN_PLAYER2;
    }
}

std::vector<std::vector<std::vector<float>>> GomokuState::getTensorRepresentation() const {
    return to_tensor();  // Use existing method
}

std::vector<std::vector<std::vector<float>>> GomokuState::getEnhancedTensorRepresentation() const {
    // Enhanced version that includes attack-defense scoring
    std::vector<std::vector<std::vector<float>>> tensor = to_tensor();
    
    // Add additional planes for previous moves
    std::vector<int> prev_black_moves = get_previous_moves(BLACK, 3);
    std::vector<int> prev_white_moves = get_previous_moves(WHITE, 3);
    
    // Add planes for CoordConv (coordinate-aware convolution)
    // This helps the network understand spatial relationships better
    
    // Resize tensor to add additional planes
    int orig_planes = tensor.size();
    tensor.resize(orig_planes + 6 + 2); // +6 for move history, +2 for CoordConv
    
    // Add previous move history planes
    for (int i = 0; i < 3; i++) {
        tensor[orig_planes + i] = std::vector<std::vector<float>>(
            board_size, std::vector<float>(board_size, 0.0f));
        
        if (prev_black_moves[i] != -1) {
            auto [x, y] = action_to_coords_pair(prev_black_moves[i]);
            tensor[orig_planes + i][x][y] = 1.0f;
        }
    }
    
    for (int i = 0; i < 3; i++) {
        tensor[orig_planes + 3 + i] = std::vector<std::vector<float>>(
            board_size, std::vector<float>(board_size, 0.0f));
        
        if (prev_white_moves[i] != -1) {
            auto [x, y] = action_to_coords_pair(prev_white_moves[i]);
            tensor[orig_planes + 3 + i][x][y] = 1.0f;
        }
    }
    
    // Add CoordConv planes
    int coord_idx = orig_planes + 6;
    tensor[coord_idx] = std::vector<std::vector<float>>(
        board_size, std::vector<float>(board_size, 0.0f));
    tensor[coord_idx + 1] = std::vector<std::vector<float>>(
        board_size, std::vector<float>(board_size, 0.0f));
    
    for (int x = 0; x < board_size; x++) {
        for (int y = 0; y < board_size; y++) {
            tensor[coord_idx][x][y] = static_cast<float>(x) / (board_size - 1);
            tensor[coord_idx + 1][x][y] = static_cast<float>(y) / (board_size - 1);
        }
    }
    
    return tensor;
}

uint64_t GomokuState::getHash() const {
    return compute_hash_signature();  // Use existing method
}

std::unique_ptr<core::IGameState> GomokuState::clone() const {
    return std::make_unique<GomokuState>(*this);
}

std::string GomokuState::actionToString(int action) const {
    if (action < 0 || action >= board_size * board_size) {
        return "invalid";
    }
    
    auto [x, y] = action_to_coords_pair(action);
    char col = 'A' + y;
    if (col >= 'I') col++; // Skip 'I' to follow Go/Gomoku convention
    int row = board_size - x;
    
    std::stringstream ss;
    ss << col << row;
    return ss.str();
}

std::optional<int> GomokuState::stringToAction(const std::string& moveStr) const {
    if (moveStr.length() < 2 || moveStr.length() > 3) {
        return std::nullopt;
    }
    
    char col = moveStr[0];
    if (col >= 'a' && col <= 'z') col = col - 'a' + 'A';  // Convert to uppercase
    
    if (col < 'A' || col > 'Z') return std::nullopt;
    if (col >= 'I') col--; // Adjust for skipping 'I'
    
    int y = col - 'A';
    
    // Parse the row
    int row;
    try {
        row = std::stoi(moveStr.substr(1));
    } catch (...) {
        return std::nullopt;
    }
    
    int x = board_size - row;
    
    if (x < 0 || x >= board_size || y < 0 || y >= board_size) {
        return std::nullopt;
    }
    
    return coords_to_action(x, y);
}

std::string GomokuState::toString() const {
    std::stringstream ss;
    
    // Print column headers
    ss << "  ";
    for (int y = 0; y < board_size; y++) {
        char col = 'A' + y;
        if (col >= 'I') col++; // Skip 'I'
        ss << " " << col;
    }
    ss << std::endl;
    
    // Print board with row numbers
    auto board = get_board();
    for (int x = 0; x < board_size; x++) {
        int row = board_size - x;
        ss << (row < 10 ? " " : "") << row << " ";
        
        for (int y = 0; y < board_size; y++) {
            switch (board[x][y]) {
                case 0: ss << ". "; break;
                case 1: ss << "X "; break;
                case 2: ss << "O "; break;
                default: ss << "? "; break;
            }
        }
        ss << row << std::endl;
    }
    
    // Print column headers again
    ss << "  ";
    for (int y = 0; y < board_size; y++) {
        char col = 'A' + y;
        if (col >= 'I') col++; // Skip 'I'
        ss << " " << col;
    }
    ss << std::endl;
    
    // Current player
    ss << "Current player: " << (current_player == BLACK ? "Black (X)" : "White (O)") << std::endl;
    
    return ss.str();
}

bool GomokuState::equals(const IGameState& other) const {
    if (other.getGameType() != core::GameType::GOMOKU) {
        return false;
    }
    
    try {
        const GomokuState& otherGomoku = dynamic_cast<const GomokuState&>(other);
        return board_equal(otherGomoku);
    } catch (const std::bad_cast&) {
        return false;
    }
}

bool GomokuState::validate() const {
    // Basic validation checks
    if (board_size <= 0) {
        return false;
    }
    
    // Check if the number of stones makes sense
    int black_stones = 0;
    int white_stones = 0;
    
    for (int a = 0; a < board_size * board_size; a++) {
        if (is_bit_set(0, a)) black_stones++;
        if (is_bit_set(1, a)) white_stones++;
    }
    
    // Black goes first, so black_stones should be equal to or one more than white_stones
    if (black_stones < white_stones || black_stones > white_stones + 1) {
        return false;
    }
    
    // Current player check
    if (current_player != BLACK && current_player != WHITE) {
        return false;
    }
    
    return true;
}

// Bitboard operations

bool GomokuState::is_bit_set(int player_index, int action) const noexcept {
    // Early bounds check to avoid out-of-bounds access
    if (player_index < 0 || player_index >= 2 || action < 0 || action >= board_size * board_size) {
        return false;
    }
    
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Additional bounds check for word_idx
    if (word_idx >= num_words) {
        return false;
    }
    
    // Use uint64_t mask for proper bit manipulation
    uint64_t mask = static_cast<uint64_t>(1) << bit_idx;
    return (player_bitboards[player_index][word_idx] & mask) != 0;
}

void GomokuState::set_bit(int player_index, int action) {
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Use |= for optimal bit setting
    player_bitboards[player_index][word_idx] |= (static_cast<uint64_t>(1) << bit_idx);
    
    // Mark caches as dirty
    invalidate_caches();
}

void GomokuState::clear_bit(int player_index, int action) noexcept {
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Use &= with negated mask for optimal bit clearing
    player_bitboards[player_index][word_idx] &= ~(static_cast<uint64_t>(1) << bit_idx);
    
    // Mark caches as dirty
    invalidate_caches();
}

std::pair<int, int> GomokuState::action_to_coords_pair(int action) const noexcept {
    return {action / board_size, action % board_size};
}

int GomokuState::coords_to_action(int x, int y) const {
    return x * board_size + y;
}

int GomokuState::count_total_stones() const noexcept {
    int total = 0;
    
    for (int p = 0; p < 2; p++) {
        for (int w = 0; w < num_words; w++) {
            uint64_t chunk = player_bitboards[p][w];
            
            // Use __builtin_popcountll for fast bit counting if available
            #if defined(__GNUC__) || defined(__clang__)
                total += __builtin_popcountll(chunk);
            #else
                // Fallback to manual counting with Brian Kernighan's algorithm
                while (chunk != 0) {
                    chunk &= (chunk - 1);  // Clear lowest set bit
                    total++;
                }
            #endif
        }
    }
    
    return total;
}

// Game state methods

void GomokuState::refresh_winner_cache() const {
    // Check for a win by either player
    for (int p : {BLACK, WHITE}) {
        if (rules->is_five_in_a_row(-1, p)) {
            cached_winner = p;
            winner_check_dirty = false;
            return;
        }
    }
    
    cached_winner = 0;
    winner_check_dirty = false;
}

bool GomokuState::is_terminal() const {
    // Check winner cache first
    if (winner_check_dirty) {
        refresh_winner_cache();
    }
    
    // If we have a winner, game is over
    if (cached_winner != 0) {
        return true;
    }
    
    // Check for stalemate (board full)
    return is_stalemate();
}

bool GomokuState::is_stalemate() const {
    // Use cached valid moves if available
    if (!valid_moves_dirty) {
        return cached_valid_moves.empty();
    }
    
    // Simple check: if board is full, it's a stalemate
    int stones = count_total_stones();
    if (stones >= board_size * board_size) {
        return true;
    }
    
    // Otherwise, check if there are any valid moves
    refresh_valid_moves_cache();
    return cached_valid_moves.empty();
}

int GomokuState::get_winner() const {
    if (winner_check_dirty) {
        refresh_winner_cache();
    }
    
    return cached_winner;
}

std::vector<int> GomokuState::get_valid_moves() const {
    // Use cache if available
    if (!valid_moves_dirty) {
        return std::vector<int>(cached_valid_moves.begin(), cached_valid_moves.end());
    }
    
    // Refresh cache
    refresh_valid_moves_cache();
    
    // Return cached result
    return std::vector<int>(cached_valid_moves.begin(), cached_valid_moves.end());
}

void GomokuState::refresh_valid_moves_cache() const {
    cached_valid_moves.clear();
    int total = board_size * board_size;
    int stone_count = count_total_stones();
    
    // First identify all empty cells
    for (int a = 0; a < total; a++) {
        if (!is_occupied(a)) {
            if (use_pro_long_opening && current_player == BLACK) {
                if (!is_pro_long_move_ok(a, stone_count)) {
                    continue;
                }
            }
            
            // Check forbidden moves for Black
            if (current_player == BLACK) {
                if (use_renju) {
                    if (!rules->is_black_renju_forbidden(a)) {
                        cached_valid_moves.insert(a);
                    }
                } else if (use_omok) {
                    if (!rules->is_black_omok_forbidden(a)) {
                        cached_valid_moves.insert(a);
                    }
                } else {
                    cached_valid_moves.insert(a);
                }
            } else {
                cached_valid_moves.insert(a);
            }
        }
    }
    
    valid_moves_dirty = false;
}

bool GomokuState::is_move_valid(int action) const {
    // Quick bounds check
    int total = board_size * board_size;
    if (action < 0 || action >= total) {
        return false;
    }
    
    // Check if already occupied
    if (is_occupied(action)) {
        return false;
    }
    
    // Check cached valid moves if available
    if (!valid_moves_dirty) {
        return cached_valid_moves.find(action) != cached_valid_moves.end();
    }
    
    // Special case for pro-long opening
    if (use_pro_long_opening && current_player == BLACK) {
        if (!is_pro_long_move_ok(action, count_total_stones())) {
            return false;
        }
    }
    
    // Check forbidden moves for Black
    if (current_player == BLACK) {
        if (use_renju) {
            if (rules->is_black_renju_forbidden(action)) {
                return false;
            }
        } else if (use_omok) {
            if (rules->is_black_omok_forbidden(action)) {
                return false;
            }
        }
    }
    
    return true;
}

uint64_t GomokuState::compute_hash_signature() const {
    if (!hash_dirty) {
        return hash_signature;
    }
    
    uint64_t hash = 0;
    const int cells = board_size * board_size;
    
    // Use large prime multipliers for better distribution
    const uint64_t black_prime = 73856093;
    const uint64_t white_prime = 19349663;
    
    // Process in chunks of 64 bits
    for (int w = 0; w < num_words; w++) {
        uint64_t black_word = player_bitboards[0][w];
        uint64_t white_word = player_bitboards[1][w];
        
        // Process all bits set to 1
        for (int b = 0; b < 64; b++) {
            uint64_t mask = static_cast<uint64_t>(1) << b;
            int action = w * 64 + b;
            
            if (action >= cells) break;
            
            if (black_word & mask) {
                hash ^= (static_cast<uint64_t>(action) * black_prime);
            } else if (white_word & mask) {
                hash ^= (static_cast<uint64_t>(action) * white_prime);
            }
        }
    }
    
    // Include current player in hash
    if (current_player == BLACK) {
        hash ^= 0xABCDEF;
    }
    
    // Update cached hash
    hash_signature = hash;
    hash_dirty = false;
    
    return hash_signature;
}

bool GomokuState::board_equal(const GomokuState& other) const {
    // Quick check for different board sizes
    if (board_size != other.board_size || current_player != other.current_player) {
        return false;
    }
    
    // Compare hash signatures if available
    if (!hash_dirty && !other.hash_dirty) {
        return hash_signature == other.hash_signature;
    }
    
    // Compare individual bitboards
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < num_words; j++) {
            if (player_bitboards[i][j] != other.player_bitboards[i][j]) {
                return false;
            }
        }
    }
    
    return true;
}

void GomokuState::make_move(int action, int player) {
    // Quick validation
    if (action < 0 || action >= board_size * board_size) {
        throw std::runtime_error("Move " + std::to_string(action) + " out of range.");
    }
    if (is_occupied(action)) {
        throw std::runtime_error("Cell " + std::to_string(action) + " is already occupied.");
    }
    
    // Rule validation if needed
    if (use_pro_long_opening && player == BLACK) {
        if (!is_pro_long_move_ok(action, count_total_stones())) {
            throw std::runtime_error("Pro-Long Opening constraint violated.");
        }
    }
    
    if (player == BLACK) {
        if (use_renju && rules->is_black_renju_forbidden(action)) {
            throw std::runtime_error("Forbidden Move by Black (Renju).");
        } else if (use_omok && rules->is_black_omok_forbidden(action)) {
            throw std::runtime_error("Forbidden Move by Black (Omok).");
        }
    }
    
    // Place the stone with bitboard operations
    set_bit(player - 1, action);
    this->action = action;
    
    // Update black's first stone if needed
    if (player == BLACK && black_first_stone < 0) {
        black_first_stone = action;
    }
    
    // Update player turn
    current_player = 3 - player;
    
    // Add to move history
    move_history.push_back(action);
    
    // Invalidate caches
    invalidate_caches();
}

void GomokuState::undo_move(int action) {
    int total = board_size * board_size;
    if (action < 0 || action >= total) {
        throw std::runtime_error("Undo " + std::to_string(action) + " out of range.");
    }

    int prev_player = 3 - current_player;
    int p_idx = prev_player - 1;

    if (!is_bit_set(p_idx, action)) {
        throw std::runtime_error("Undo error: Stone not found for last mover.");
    }

    // Remove the stone
    clear_bit(p_idx, action);
    this->action = -1;
    
    // Update player turn
    current_player = prev_player;

    if (prev_player == BLACK && black_first_stone == action) {
        black_first_stone = -1;
    }

    // Update move history
    if (!move_history.empty()) {
        move_history.pop_back();
    }

    // Invalidate caches
    invalidate_caches();
}

void GomokuState::invalidate_caches() {
    valid_moves_dirty = true;
    winner_check_dirty = true;
    hash_dirty = true;
}

bool GomokuState::is_occupied(int action) const {
    // Use bitwise OR to check both players in one operation
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    if (word_idx >= num_words) {
        return true; // Out of bounds is considered occupied
    }
    
    uint64_t mask = static_cast<uint64_t>(1) << bit_idx;
    return ((player_bitboards[0][word_idx] | player_bitboards[1][word_idx]) & mask) != 0;
}

std::vector<std::vector<int>> GomokuState::get_board() const {
    std::vector<std::vector<int>> arr(board_size, std::vector<int>(board_size, 0));
    int total = board_size * board_size;
    
    for (int p_idx = 0; p_idx < 2; p_idx++) {
        for (int w = 0; w < num_words; w++) {
            uint64_t chunk = player_bitboards[p_idx][w];
            if (chunk == 0) {
                continue;
            }
            
            for (int b = 0; b < 64; b++) {
                if ((chunk & (static_cast<uint64_t>(1) << b)) != 0) {
                    int action = w * 64 + b;
                    if (action >= total) {
                        break;
                    }
                    int x = action / board_size;
                    int y = action % board_size;
                    arr[x][y] = (p_idx + 1);
                }
            }
        }
    }
    
    return arr;
}

// MCTS/NN support functions
GomokuState GomokuState::apply_action(int action) const {
    GomokuState new_state(*this);
    new_state.make_move(action, current_player);
    return new_state;
}

std::vector<std::vector<std::vector<float>>> GomokuState::to_tensor() const {
    std::vector<std::vector<std::vector<float>>> tensor(3, 
        std::vector<std::vector<float>>(board_size, 
            std::vector<float>(board_size, 0.0f)));
    
    int p_idx = current_player - 1;
    int opp_idx = 1 - p_idx;
    int total = board_size * board_size;
    
    for (int a = 0; a < total; a++) {
        int x = a / board_size;
        int y = a % board_size;
        
        if (is_bit_set(p_idx, a)) {
            tensor[0][x][y] = 1.0f;
        } else if (is_bit_set(opp_idx, a)) {
            tensor[1][x][y] = 1.0f;
        }
    }
    
    if (current_player == BLACK) {
        for (int i = 0; i < board_size; i++) {
            for (int j = 0; j < board_size; j++) {
                tensor[2][i][j] = 1.0f;
            }
        }
    }
    
    return tensor;
}

int GomokuState::get_action(const GomokuState& child_state) const {
    int total = board_size * board_size;
    for (int a = 0; a < total; a++) {
        if (is_occupied(a) != child_state.is_occupied(a)) {
            return a;
        }
    }
    return -1;
}

std::vector<int> GomokuState::get_previous_moves(int player, int count) const {
    std::vector<int> prev_moves(count, -1);  // Initialize with -1 (no move)
    
    int found = 0;
    // Iterate backward through move history
    for (int i = static_cast<int>(move_history.size()) - 1; i >= 0 && found < count; --i) {
        int move = move_history[i];
        // Determine which player made this move based on position in history
        int move_player = (move_history.size() - i) % 2 == 1 ? current_player : 3 - current_player;
        
        if (move_player == player) {
            prev_moves[found] = move;
            found++;
        }
    }
    
    return prev_moves;
}

// Helper methods

bool GomokuState::in_bounds(int x, int y) const {
    return (0 <= x && x < board_size) && (0 <= y && y < board_size);
}

bool GomokuState::is_pro_long_move_ok(int action, int stone_count) const {
    int center = (board_size / 2) * board_size + (board_size / 2);
    
    if (stone_count == 0 || stone_count == 1) {
        return (action == center);
    } else if (stone_count == 2 || stone_count == 3) {
        if (black_first_stone < 0) {
            return false;
        }
        
        auto [x0, y0] = action_to_coords_pair(black_first_stone);
        auto [x1, y1] = action_to_coords_pair(action);
        int dist = abs(x1 - x0) + abs(y1 - y0);
        return (dist >= 4);
    }
    
    return true;
}

} // namespace gomoku
} // namespace alphazero