// src/games/go/go_state.cpp
#include "alphazero/games/go/go_state.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace alphazero {
namespace go {

// Constructor
GoState::GoState(int board_size, float komi, bool chinese_rules)
    : IGameState(core::GameType::GO),
      board_size_(board_size),
      current_player_(1),  // Black goes first
      komi_(komi),
      chinese_rules_(chinese_rules),
      ko_point_(-1),
      consecutive_passes_(0),
      hash_dirty_(true),
      zobrist_(core::GameType::GO, board_size, 2)  // 2 colors
{
    // Validate board size
    if (board_size != 9 && board_size != 13 && board_size != 19) {
        board_size_ = 19;  // Default to standard 19x19 if invalid
    }

    // Initialize board with empty intersections
    board_.resize(board_size_ * board_size_, 0);
    
    // Initialize capture counts
    captured_stones_.resize(3, 0);  // Index 0 unused, 1=Black, 2=White
    
    // Initialize Zobrist hash
    hash_ = 0;
    hash_dirty_ = true;
    
    // Initialize rules
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_);
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );
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
      zobrist_(other.zobrist_),
      hash_(other.hash_),
      hash_dirty_(other.hash_dirty_)
{
    // Initialize rules
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_);
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );
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
        
        // Reinitialize rules
        rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_);
        
        // Set up board accessor functions for rules
        rules_->setBoardAccessor(
            [this](int pos) { return this->getStone(pos); },
            [this](int pos) { return this->isInBounds(pos); },
            [this](int pos) { return this->getAdjacentPositions(pos); }
        );
    }
    return *this;
}

// IGameState interface implementation
std::vector<int> GoState::getLegalMoves() const {
    std::vector<int> legalMoves;
    
    // Add pass move (-1)
    legalMoves.push_back(-1);
    
    // Check all board positions
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        if (isValidMove(pos)) {
            legalMoves.push_back(pos);
        }
    }
    
    return legalMoves;
}

bool GoState::isLegalMove(int action) const {
    if (action == -1) {
        return true;  // Pass is always legal
    }
    
    // First check basic validity
    if (!isValidMove(action)) {
        return false;
    }
    
    // Create a temporary copy to test for superko (position repetition)
    GoState tempState(*this);
    
    // Temporarily apply move
    tempState.setStone(action, tempState.current_player_);
    
    // Process any captures that would occur
    std::vector<StoneGroup> opponentGroups = tempState.rules_->findGroups(3 - tempState.current_player_);
    for (const auto& group : opponentGroups) {
        if (group.liberties.empty()) {
            tempState.captureGroup(group);
        }
    }
    
    // Check if this position has appeared before (superko rule)
    uint64_t newHash = tempState.getHash();
    for (uint64_t hash : tempState.position_history_) {
        if (hash == newHash) {
            return false;  // Position repetition found, move is illegal
        }
    }
    
    return true;
}

void GoState::makeMove(int action) {
    if (!isLegalMove(action)) {
        throw std::runtime_error("Illegal move attempted");
    }
    
    // Handle pass
    if (action == -1) {
        consecutive_passes_++;
        ko_point_ = -1;  // Clear ko point on pass
        
        // Record move
        move_history_.push_back(action);
    } else {
        // Reset consecutive passes
        consecutive_passes_ = 0;
        
        // Place stone
        setStone(action, current_player_);
        
        // Save current position before processing captures
        uint64_t currentHash = getHash();
        
        // Check for captures
        std::vector<StoneGroup> opponentGroups = rules_->findGroups(3 - current_player_);
        std::vector<StoneGroup> capturedGroups;
        int capturedStones = 0;
        
        for (const auto& group : opponentGroups) {
            if (group.liberties.empty()) {
                capturedGroups.push_back(group);
                capturedStones += group.stones.size();
            }
        }
        
        // Check for ko
        if (capturedGroups.size() == 1 && capturedGroups[0].stones.size() == 1) {
            ko_point_ = *capturedGroups[0].stones.begin();
        } else {
            ko_point_ = -1;
        }
        
        // Process captures
        for (const auto& group : capturedGroups) {
            captureGroup(group);
        }
        
        // Update capture count
        captured_stones_[current_player_] += capturedStones;
        
        // Record move
        move_history_.push_back(action);
        
        // Get the updated hash after captures
        invalidateHash();
        uint64_t newHash = getHash();
        
        // Record position for ko detection
        position_history_.push_back(newHash);
    }
    
    // Switch players
    current_player_ = 3 - current_player_;
    
    // Invalidate hash
    invalidateHash();
}

bool GoState::undoMove() {
    if (move_history_.empty()) {
        return false;
    }
    
    // Get last move
    int lastMove = move_history_.back();
    move_history_.pop_back();
    
    // Remove last position from history
    if (!position_history_.empty()) {
        position_history_.pop_back();
    }
    
    // Switch back to previous player
    current_player_ = 3 - current_player_;
    
    // Handle pass
    if (lastMove == -1) {
        if (consecutive_passes_ > 0) {
            consecutive_passes_--;
        }
        
        // Restore ko point from history if available
        ko_point_ = -1;
        
        // Invalidate hash
        invalidateHash();
        return true;
    }
    
    // TODO: Implement full undo with stone restoration
    // This would require storing more state information in move_history_
    
    // For now, just return false to indicate undo not fully supported
    return false;
}

bool GoState::isTerminal() const {
    // Game ends when both players pass consecutively
    return consecutive_passes_ >= 2;
}

core::GameResult GoState::getGameResult() const {
    if (!isTerminal()) {
        return core::GameResult::ONGOING;
    }
    
    // Calculate scores
    auto [blackScore, whiteScore] = rules_->calculateScores(captured_stones_, komi_);
    
    if (blackScore > whiteScore) {
        return core::GameResult::WIN_PLAYER1;  // Black wins
    } else if (whiteScore > blackScore) {
        return core::GameResult::WIN_PLAYER2;  // White wins
    } else {
        return core::GameResult::DRAW;  // Draw
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
    // Basic 3-plane representation (black, white, turn)
    std::vector<std::vector<std::vector<float>>> tensor(3, 
        std::vector<std::vector<float>>(board_size_, 
            std::vector<float>(board_size_, 0.0f)));
    
    // Fill first two planes with stone positions
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            int stone = getStone(pos);
            
            if (stone == 1) {
                tensor[0][y][x] = 1.0f;  // Black stones
            } else if (stone == 2) {
                tensor[1][y][x] = 1.0f;  // White stones
            }
        }
    }
    
    // Third plane: current player
    float playerValue = (current_player_ == 1) ? 1.0f : 0.0f;
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            tensor[2][y][x] = playerValue;
        }
    }
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> GoState::getEnhancedTensorRepresentation() const {
    // Start with the basic representation
    std::vector<std::vector<std::vector<float>>> tensor = getTensorRepresentation();
    
    // Add additional planes for enhanced features
    
    // 4: Liberties of black groups (normalized)
    // 5: Liberties of white groups (normalized)
    std::vector<std::vector<float>> blackLiberties(board_size_, std::vector<float>(board_size_, 0.0f));
    std::vector<std::vector<float>> whiteLiberties(board_size_, std::vector<float>(board_size_, 0.0f));
    
    // Calculate liberties for each group
    auto blackGroups = rules_->findGroups(1);
    auto whiteGroups = rules_->findGroups(2);
    
    for (const auto& group : blackGroups) {
        float libertyCount = static_cast<float>(group.liberties.size());
        float normalizedLiberties = std::min(1.0f, libertyCount / 10.0f);  // Normalize to [0,1]
        
        for (int pos : group.stones) {
            int y = pos / board_size_;
            int x = pos % board_size_;
            blackLiberties[y][x] = normalizedLiberties;
        }
    }
    
    for (const auto& group : whiteGroups) {
        float libertyCount = static_cast<float>(group.liberties.size());
        float normalizedLiberties = std::min(1.0f, libertyCount / 10.0f);  // Normalize to [0,1]
        
        for (int pos : group.stones) {
            int y = pos / board_size_;
            int x = pos % board_size_;
            whiteLiberties[y][x] = normalizedLiberties;
        }
    }
    
    tensor.push_back(blackLiberties);
    tensor.push_back(whiteLiberties);
    
    // 6: Ko point
    std::vector<std::vector<float>> koPlane(board_size_, std::vector<float>(board_size_, 0.0f));
    if (ko_point_ >= 0) {
        int y = ko_point_ / board_size_;
        int x = ko_point_ % board_size_;
        koPlane[y][x] = 1.0f;
    }
    tensor.push_back(koPlane);
    
    // 7-8: Distance transforms from borders
    std::vector<std::vector<float>> distanceX(board_size_, std::vector<float>(board_size_, 0.0f));
    std::vector<std::vector<float>> distanceY(board_size_, std::vector<float>(board_size_, 0.0f));
    
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            // Normalize distances to [0,1]
            distanceX[y][x] = static_cast<float>(std::min(x, board_size_ - 1 - x)) / (board_size_ / 2);
            distanceY[y][x] = static_cast<float>(std::min(y, board_size_ - 1 - y)) / (board_size_ / 2);
        }
    }
    
    tensor.push_back(distanceX);
    tensor.push_back(distanceY);
    
    return tensor;
}

uint64_t GoState::getHash() const {
    if (hash_dirty_) {
        updateHash();
    }
    return hash_;
}

std::unique_ptr<core::IGameState> GoState::clone() const {
    return std::make_unique<GoState>(*this);
}

std::string GoState::actionToString(int action) const {
    if (action == -1) {
        return "pass";
    }
    
    if (action < 0 || action >= board_size_ * board_size_) {
        return "invalid";
    }
    
    std::pair<int, int> coords = actionToCoord(action);
    int x = coords.first;
    int y = coords.second;
    
    // Convert to Go coordinates (A-T, skipping I, 1-19)
    char colChar = 'A' + x;
    if (colChar >= 'I') {
        colChar++;  // Skip 'I'
    }
    
    return std::string(1, colChar) + std::to_string(board_size_ - y);
}

std::optional<int> GoState::stringToAction(const std::string& moveStr) const {
    if (moveStr == "pass" || moveStr == "PASS" || moveStr == "Pass") {
        return -1;
    }
    
    if (moveStr.length() < 2) {
        return std::nullopt;
    }
    
    char colChar = std::toupper(moveStr[0]);
    
    // Adjust for 'I' being skipped in Go notation
    if (colChar >= 'J') {
        colChar--;
    }
    
    int x = colChar - 'A';
    
    // Parse row
    int y;
    try {
        y = board_size_ - std::stoi(moveStr.substr(1));
    } catch (...) {
        return std::nullopt;
    }
    
    if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
        return std::nullopt;
    }
    
    return coordToAction(x, y);
}

std::string GoState::toString() const {
    std::stringstream ss;
    
    // Print column headers
    ss << "   ";
    for (int x = 0; x < board_size_; ++x) {
        char colChar = 'A' + x;
        if (colChar >= 'I') {
            colChar++;  // Skip 'I'
        }
        ss << colChar << " ";
    }
    ss << std::endl;
    
    // Print board
    for (int y = 0; y < board_size_; ++y) {
        ss << std::setw(2) << (board_size_ - y) << " ";
        
        for (int x = 0; x < board_size_; ++x) {
            int stone = getStone(x, y);
            if (stone == 0) {
                // Check if this is a ko point
                if (coordToAction(x, y) == ko_point_) {
                    ss << "k ";
                } else {
                    ss << ". ";
                }
            } else if (stone == 1) {
                ss << "X ";  // Black
            } else if (stone == 2) {
                ss << "O ";  // White
            }
        }
        
        ss << (board_size_ - y) << std::endl;
    }
    
    // Print column headers again
    ss << "   ";
    for (int x = 0; x < board_size_; ++x) {
        char colChar = 'A' + x;
        if (colChar >= 'I') {
            colChar++;  // Skip 'I'
        }
        ss << colChar << " ";
    }
    ss << std::endl;
    
    // Print game info
    ss << "Current player: " << (current_player_ == 1 ? "Black" : "White") << std::endl;
    ss << "Captures - Black: " << captured_stones_[1] << ", White: " << captured_stones_[2] << std::endl;
    ss << "Komi: " << komi_ << std::endl;
    ss << "Rules: " << (chinese_rules_ ? "Chinese" : "Japanese") << std::endl;
    
    if (isTerminal()) {
        auto [blackScore, whiteScore] = rules_->calculateScores(captured_stones_, komi_);
        
        ss << "Game over!" << std::endl;
        ss << "Final score - Black: " << blackScore << ", White: " << whiteScore 
           << " (with komi " << komi_ << ")" << std::endl;
        
        if (blackScore > whiteScore) {
            ss << "Black wins by " << (blackScore - whiteScore) << " points" << std::endl;
        } else if (whiteScore > blackScore) {
            ss << "White wins by " << (whiteScore - blackScore) << " points" << std::endl;
        } else {
            ss << "Game ended in a draw" << std::endl;
        }
    }
    
    return ss.str();
}

bool GoState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::GO) {
        return false;
    }
    
    try {
        const GoState& otherGo = dynamic_cast<const GoState&>(other);
        
        if (board_size_ != otherGo.board_size_ || 
            current_player_ != otherGo.current_player_ ||
            ko_point_ != otherGo.ko_point_) {
            return false;
        }
        
        // Compare board positions
        return board_ == otherGo.board_;
    } catch (const std::bad_cast&) {
        return false;
    }
}

std::vector<int> GoState::getMoveHistory() const {
    return move_history_;
}

bool GoState::validate() const {
    // Check board size
    if (board_size_ != 9 && board_size_ != 13 && board_size_ != 19) {
        return false;
    }
    
    // Check current player
    if (current_player_ != 1 && current_player_ != 2) {
        return false;
    }
    
    // Check ko point
    if (ko_point_ >= board_size_ * board_size_) {
        return false;
    }
    
    return true;
}

// Go-specific methods
int GoState::getStone(int pos) const {
    if (pos < 0 || pos >= board_size_ * board_size_) {
        return 0;  // Out of bounds, return empty
    }
    return board_[pos];
}

int GoState::getStone(int x, int y) const {
    if (!isInBounds(x, y)) {
        return 0;  // Out of bounds, return empty
    }
    return board_[y * board_size_ + x];
}

void GoState::setStone(int pos, int stone) {
    if (pos < 0 || pos >= board_size_ * board_size_) {
        return;  // Out of bounds, do nothing
    }
    board_[pos] = stone;
    invalidateHash();
}

void GoState::setStone(int x, int y, int stone) {
    if (!isInBounds(x, y)) {
        return;  // Out of bounds, do nothing
    }
    board_[y * board_size_ + x] = stone;
    invalidateHash();
}

int GoState::getCapturedStones(int player) const {
    if (player != 1 && player != 2) {
        return 0;
    }
    return captured_stones_[player];
}

float GoState::getKomi() const {
    return komi_;
}

bool GoState::isChineseRules() const {
    return chinese_rules_;
}

std::pair<int, int> GoState::actionToCoord(int action) const {
    if (action < 0 || action >= board_size_ * board_size_) {
        return {-1, -1};
    }
    
    int y = action / board_size_;
    int x = action % board_size_;
    
    return {x, y};
}

int GoState::coordToAction(int x, int y) const {
    if (!isInBounds(x, y)) {
        return -1;
    }
    
    return y * board_size_ + x;
}

int GoState::getKoPoint() const {
    return ko_point_;
}

std::vector<int> GoState::getTerritoryOwnership() const {
    return rules_->getTerritoryOwnership();
}

bool GoState::isInsideTerritory(int pos, int player) const {
    std::vector<int> territory = getTerritoryOwnership();
    if (pos < 0 || pos >= static_cast<int>(territory.size())) {
        return false;
    }
    return territory[pos] == player;
}

// Helper methods
std::vector<int> GoState::getAdjacentPositions(int pos) const {
    std::vector<int> adjacentPositions;
    int x, y;
    std::tie(x, y) = actionToCoord(pos);
    
    // Check orthogonally adjacent positions
    for (const auto& direction : std::vector<std::pair<int, int>>{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}) {
        int newX = x + direction.first;
        int newY = y + direction.second;
        
        if (isInBounds(newX, newY)) {
            adjacentPositions.push_back(coordToAction(newX, newY));
        }
    }
    
    return adjacentPositions;
}

bool GoState::isInBounds(int x, int y) const {
    return x >= 0 && x < board_size_ && y >= 0 && y < board_size_;
}

bool GoState::isInBounds(int pos) const {
    return pos >= 0 && pos < board_size_ * board_size_;
}

void GoState::invalidateHash() {
    hash_dirty_ = true;
}

void GoState::captureGroup(const StoneGroup& group) {
    // Remove all stones in the group
    for (int pos : group.stones) {
        setStone(pos, 0);
    }
}

bool GoState::isValidMove(int action) const {
    if (action < 0 || action >= board_size_ * board_size_) {
        return false;
    }
    
    // Check if the intersection is empty
    if (getStone(action) != 0) {
        return false;
    }
    
    // Check if this is a ko point
    if (rules_->isKoViolation(action, ko_point_)) {
        return false;
    }
    
    // Check for suicide rule
    if (rules_->isSuicidalMove(action, current_player_)) {
        return false;
    }
    
    return true;
}

void GoState::updateHash() const {
    hash_ = 0;
    
    // Hash board position
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        int stone = getStone(pos);
        if (stone != 0) {
            int pieceIdx = stone - 1;  // Convert to 0-based index
            hash_ ^= zobrist_.getPieceHash(pieceIdx, pos);
        }
    }
    
    // Hash current player
    hash_ ^= zobrist_.getPlayerHash(current_player_ - 1);
    
    // Hash ko point
    if (ko_point_ >= 0) {
        hash_ ^= zobrist_.getFeatureHash(0, ko_point_);
    }
    
    hash_dirty_ = false;
}

} // namespace go
} // namespace alphazero