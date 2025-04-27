// src/games/go/go_state.cpp
#include "alphazero/games/go/go_state.h"
#include <sstream>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace alphazero {
namespace go {

GoState::GoState(int board_size, float komi, bool chinese_rules)
    : IGameState(core::GameType::GO),
      board_size_(board_size),
      current_player_(1),  // Black plays first
      komi_(komi),
      chinese_rules_(chinese_rules),
      ko_point_(-1),
      consecutive_passes_(0),
      hash_dirty_(true),
      zobrist_(core::GameType::GO, board_size, 2) {
    
    // Initialize board
    board_.resize(board_size * board_size, 0);
    captured_stones_.resize(3, 0);  // Index 0 unused, 1 for black, 2 for white
    
    // Initialize hash
    hash_ = 0;
}

// Copy constructor
GoState::GoState(const GoState& other)
    : IGameState(core::GameType::GO),
      board_size_(other.board_size_),
      current_player_(other.current_player_),
      board_(other.board_),
      komi_(other.komi_),
      chinese_rules_(other.chinese_rules_),
      ko_point_(other.ko_point_),
      captured_stones_(other.captured_stones_),
      consecutive_passes_(other.consecutive_passes_),
      move_history_(other.move_history_),
      position_history_(other.position_history_),
      zobrist_(core::GameType::GO, other.board_size_, 2),
      hash_(other.hash_),
      hash_dirty_(other.hash_dirty_) {
}

// Assignment operator
GoState& GoState::operator=(const GoState& other) {
    if (this != &other) {
        board_size_ = other.board_size_;
        current_player_ = other.current_player_;
        board_ = other.board_;
        komi_ = other.komi_;
        chinese_rules_ = other.chinese_rules_;
        ko_point_ = other.ko_point_;
        captured_stones_ = other.captured_stones_;
        consecutive_passes_ = other.consecutive_passes_;
        move_history_ = other.move_history_;
        position_history_ = other.position_history_;
        hash_ = other.hash_;
        hash_dirty_ = other.hash_dirty_;
    }
    return *this;
}

std::vector<int> GoState::getLegalMoves() const {
    std::vector<int> legal_moves;
    legal_moves.reserve(board_size_ * board_size_ + 1);  // +1 for pass
    
    // Pass is always legal
    legal_moves.push_back(board_size_ * board_size_);
    
    // Check each board position
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        if (isValidMove(pos)) {
            legal_moves.push_back(pos);
        }
    }
    
    return legal_moves;
}

bool GoState::isLegalMove(int action) const {
    // Pass is always legal
    if (action == board_size_ * board_size_) {
        return true;
    }
    
    return isValidMove(action);
}

void GoState::makeMove(int action) {
    // Record position for ko detection
    uint64_t old_hash = getHash();
    position_history_.push_back(old_hash);
    
    // Reset ko point by default
    int old_ko = ko_point_;
    ko_point_ = -1;
    
    if (action == board_size_ * board_size_) {
        // Handle pass
        consecutive_passes_++;
        move_history_.push_back(action);
        
        // Switch player
        current_player_ = (current_player_ == 1) ? 2 : 1;
        
        invalidateHash();
        return;
    }
    
    // Reset consecutive passes
    consecutive_passes_ = 0;
    
    // Place stone
    setStone(action, current_player_);
    
    // Process captures
    std::vector<StoneGroup> opponent_groups = findGroups(3 - current_player_);
    int captured_count = 0;
    
    for (const auto& group : opponent_groups) {
        if (group.liberties.empty()) {
            captured_count += group.stones.size();
            captureGroup(group);
        }
    }
    
    // Update capture count
    captured_stones_[current_player_] += captured_count;
    
    // Check if move results in a single stone capture, which might create a ko
    if (captured_count == 1) {
        // Check if the stone we just placed has only one liberty
        std::vector<StoneGroup> our_groups = findGroups(current_player_);
        for (const auto& group : our_groups) {
            if (group.stones.count(action) > 0 && group.stones.size() == 1 && group.liberties.size() == 1) {
                // This is a potential ko point
                ko_point_ = *group.liberties.begin();
                break;
            }
        }
    }
    
    // Check for suicide
    if (isSuicidalMove(action)) {
        // This is against the rules, but handled in isValidMove
        // This should never happen if we check legality before making the move
        std::cerr << "Warning: Suicidal move detected" << std::endl;
    }
    
    // Record move
    move_history_.push_back(action);
    
    // Switch player
    current_player_ = (current_player_ == 1) ? 2 : 1;
    
    invalidateHash();
}

bool GoState::undoMove() {
    if (move_history_.empty()) {
        return false;
    }
    
    // TODO: Implement undo functionality
    // This is quite complex in Go due to captures
    // For now, we'll leave this as a stub
    
    return false;
}

bool GoState::isTerminal() const {
    // Game is over after two consecutive passes
    return consecutive_passes_ >= 2;
}

core::GameResult GoState::getGameResult() const {
    if (!isTerminal()) {
        return core::GameResult::ONGOING;
    }
    
    // Calculate final scores
    auto [black_score, white_score] = calculateScores();
    
    if (std::abs(black_score - white_score) < 0.01f) {
        return core::GameResult::DRAW;
    } else if (black_score > white_score) {
        return core::GameResult::WIN_PLAYER1;
    } else {
        return core::GameResult::WIN_PLAYER2;
    }
}

int GoState::getCurrentPlayer() const {
    return current_player_;
}

int GoState::getBoardSize() const {
    return board_size_;
}

int GoState::getActionSpaceSize() const {
    return board_size_ * board_size_ + 1;  // +1 for pass
}

std::vector<std::vector<std::vector<float>>> GoState::getTensorRepresentation() const {
    // Create 3 planes: black stones, white stones, turn indicator
    std::vector<std::vector<std::vector<float>>> tensor(3, 
        std::vector<std::vector<float>>(board_size_, 
            std::vector<float>(board_size_, 0.0f)));
    
    // Fill stone positions
    for (int i = 0; i < board_size_; ++i) {
        for (int j = 0; j < board_size_; ++j) {
            int pos = i * board_size_ + j;
            int stone = board_[pos];
            
            if (stone == 1) {  // Black
                tensor[0][i][j] = 1.0f;
            } else if (stone == 2) {  // White
                tensor[1][i][j] = 1.0f;
            }
        }
    }
    
    // Fill turn indicator plane
    float turn_value = (current_player_ == 1) ? 1.0f : 0.0f;
    for (int i = 0; i < board_size_; ++i) {
        for (int j = 0; j < board_size_; ++j) {
            tensor[2][i][j] = turn_value;
        }
    }
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> GoState::getEnhancedTensorRepresentation() const {
    // Start with basic tensor
    std::vector<std::vector<std::vector<float>>> tensor = getTensorRepresentation();
    
    // Add history planes (last 7 moves for each player)
    for (int player = 1; player <= 2; ++player) {
        std::vector<int> recent_moves;
        
        // Collect recent moves for this player
        for (auto it = move_history_.rbegin(); it != move_history_.rend() && recent_moves.size() < 7; ++it) {
            if (*it < board_size_ * board_size_) {  // Ignore passes
                int move_player = (move_history_.size() - (it - move_history_.rbegin())) % 2 + 1;
                if (move_player == player) {
                    recent_moves.push_back(*it);
                }
            }
        }
        
        // Create history planes
        for (size_t i = 0; i < std::min(size_t(7), recent_moves.size()); ++i) {
            std::vector<std::vector<float>> plane(board_size_, std::vector<float>(board_size_, 0.0f));
            
            int pos = recent_moves[i];
            auto [x, y] = actionToCoord(pos);
            plane[x][y] = 1.0f;
            
            tensor.push_back(plane);
        }
        
        // Add empty planes if needed
        for (size_t i = recent_moves.size(); i < 7; ++i) {
            tensor.push_back(std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, 0.0f)));
        }
    }
    
    // Add liberty count planes (1, 2, 3+ liberties for each player)
    for (int player = 1; player <= 2; ++player) {
        std::vector<std::vector<float>> liberty1(board_size_, std::vector<float>(board_size_, 0.0f));
        std::vector<std::vector<float>> liberty2(board_size_, std::vector<float>(board_size_, 0.0f));
        std::vector<std::vector<float>> liberty3plus(board_size_, std::vector<float>(board_size_, 0.0f));
        
        // Find groups and count liberties
        std::vector<StoneGroup> groups = findGroups(player);
        
        for (const auto& group : groups) {
            int liberty_count = group.liberties.size();
            
            for (int stone : group.stones) {
                auto [x, y] = actionToCoord(stone);
                
                if (liberty_count == 1) {
                    liberty1[x][y] = 1.0f;
                } else if (liberty_count == 2) {
                    liberty2[x][y] = 1.0f;
                } else if (liberty_count >= 3) {
                    liberty3plus[x][y] = 1.0f;
                }
            }
        }
        
        tensor.push_back(liberty1);
        tensor.push_back(liberty2);
        tensor.push_back(liberty3plus);
    }
    
    // Add ko point plane
    std::vector<std::vector<float>> ko_plane(board_size_, std::vector<float>(board_size_, 0.0f));
    if (ko_point_ >= 0) {
        auto [ko_x, ko_y] = actionToCoord(ko_point_);
        ko_plane[ko_x][ko_y] = 1.0f;
    }
    tensor.push_back(ko_plane);
    
    // Add capture count planes
    float black_captures_norm = std::min(1.0f, captured_stones_[2] / 20.0f);
    float white_captures_norm = std::min(1.0f, captured_stones_[1] / 20.0f);
    
    tensor.push_back(std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, black_captures_norm)));
    tensor.push_back(std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, white_captures_norm)));
    
    return tensor;
}

uint64_t GoState::getHash() const {
    if (!hash_dirty_) {
        return hash_;
    }
    
    hash_ = 0;
    
    // Hash stone positions
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        int stone = board_[pos];
        if (stone > 0) {
            hash_ ^= zobrist_.getPieceHash(stone - 1, pos);
        }
    }
    
    // Hash current player
    if (current_player_ == 2) {
        hash_ ^= zobrist_.getPlayerHash(0);
    }
    
    // Hash ko point
    if (ko_point_ >= 0) {
        hash_ ^= zobrist_.getFeatureHash(0, ko_point_);
    }
    
    hash_dirty_ = false;
    return hash_;
}

std::unique_ptr<core::IGameState> GoState::clone() const {
    return std::make_unique<GoState>(*this);
}

std::string GoState::actionToString(int action) const {
    // Pass move
    if (action == board_size_ * board_size_) {
        return "pass";
    }
    
    auto [x, y] = actionToCoord(action);
    char col = static_cast<char>('A' + y);
    if (col >= 'I') col++;  // Skip 'I' (Go convention)
    
    // Go coordinates start from bottom-left, our internal representation starts from top-left
    int row = board_size_ - x;
    
    std::stringstream ss;
    ss << col << row;
    return ss.str();
}

std::optional<int> GoState::stringToAction(const std::string& moveStr) const {
    // Check for pass
    if (moveStr == "pass" || moveStr == "PASS") {
        return board_size_ * board_size_;
    }
    
    if (moveStr.length() < 2 || moveStr.length() > 3) {
        return std::nullopt;
    }
    
    char col = std::toupper(moveStr[0]);
    if (col >= 'I') col--;  // Adjust for 'I' skipping
    
    int file = col - 'A';
    if (file < 0 || file >= board_size_) {
        return std::nullopt;
    }
    
    // Parse row number
    int row;
    try {
        row = std::stoi(moveStr.substr(1));
    } catch (...) {
        return std::nullopt;
    }
    
    // Convert to 0-indexed from bottom
    row = board_size_ - row;
    if (row < 0 || row >= board_size_) {
        return std::nullopt;
    }
    
    return row * board_size_ + file;
}

// Detailed implementation for other methods would follow...
// The complete implementation would be much larger.

} // namespace go
} // namespace alphazero