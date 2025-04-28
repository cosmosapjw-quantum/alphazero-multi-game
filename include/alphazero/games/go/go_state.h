// include/alphazero/games/go/go_state.h
#ifndef GO_STATE_H
#define GO_STATE_H

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <optional>
#include "alphazero/core/igamestate.h"
#include "alphazero/core/zobrist_hash.h"
#include "alphazero/games/go/go_rules.h"

namespace alphazero {
namespace go {

/**
 * @brief Implementation of Go game state
 */
class GoState : public core::IGameState {
public:
    /**
     * @brief Constructor
     * 
     * @param board_size Board size (9, 13, or 19)
     * @param komi Komi value
     * @param chinese_rules Whether to use Chinese rules
     */
    GoState(int board_size = 19, float komi = 7.5f, bool chinese_rules = true);
    
    /**
     * @brief Copy constructor
     */
    GoState(const GoState& other);
    
    /**
     * @brief Assignment operator
     */
    GoState& operator=(const GoState& other);
    
    // IGameState interface implementation
    std::vector<int> getLegalMoves() const override;
    bool isLegalMove(int action) const override;
    void makeMove(int action) override;
    bool undoMove() override;
    bool isTerminal() const override;
    core::GameResult getGameResult() const override;
    int getCurrentPlayer() const override;
    int getBoardSize() const override;
    int getActionSpaceSize() const override;
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override;
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override;
    uint64_t getHash() const override;
    std::unique_ptr<core::IGameState> clone() const override;
    std::string actionToString(int action) const override;
    std::optional<int> stringToAction(const std::string& moveStr) const override;
    std::string toString() const override;
    bool equals(const core::IGameState& other) const override;
    std::vector<int> getMoveHistory() const override;
    bool validate() const override;
    
    // Go-specific methods
    /**
     * @brief Get stone at a position
     * 
     * @param pos Position index
     * @return 0 for empty, 1 for black, 2 for white
     */
    int getStone(int pos) const;
    
    /**
     * @brief Get stone at coordinates
     * 
     * @param x Row
     * @param y Column
     * @return 0 for empty, 1 for black, 2 for white
     */
    int getStone(int x, int y) const;
    
    /**
     * @brief Place a stone at a position
     * 
     * @param pos Position index
     * @param stone Stone value (1 for black, 2 for white)
     */
    void setStone(int pos, int stone);
    
    /**
     * @brief Place a stone at coordinates
     * 
     * @param x Row
     * @param y Column
     * @param stone Stone value (1 for black, 2 for white)
     */
    void setStone(int x, int y, int stone);
    
    /**
     * @brief Get captured stones count
     * 
     * @param player Player (1 for black, 2 for white)
     * @return Number of stones captured by the player
     */
    int getCapturedStones(int player) const;
    
    /**
     * @brief Get komi value
     * 
     * @return Komi value
     */
    float getKomi() const;
    
    /**
     * @brief Check if using Chinese rules
     * 
     * @return true if using Chinese rules, false if Japanese
     */
    bool isChineseRules() const;
    
    /**
     * @brief Convert action to coordinates
     * 
     * @param action Action index
     * @return Pair of (x, y) coordinates
     */
    std::pair<int, int> actionToCoord(int action) const;
    
    /**
     * @brief Convert coordinates to action
     * 
     * @param x Row
     * @param y Column
     * @return Action index
     */
    int coordToAction(int x, int y) const;
    
    /**
     * @brief Get current ko point
     * 
     * @return Ko point index, or -1 if none
     */
    int getKoPoint() const;
    
    /**
     * @brief Get territory ownership
     * 
     * @return Vector of territory ownership (0 for neutral, 1 for black, 2 for white)
     */
    std::vector<int> getTerritoryOwnership() const;
    
    /**
     * @brief Check if a point is inside territory
     * 
     * @param pos Position index
     * @param player Player (1 for black, 2 for white)
     * @return true if position is inside player's territory
     */
    bool isInsideTerritory(int pos, int player) const;
    
private:
    int board_size_;
    int current_player_;
    std::vector<int> board_;
    float komi_;
    bool chinese_rules_;
    
    // Game state tracking
    int ko_point_;
    std::vector<int> captured_stones_;
    int consecutive_passes_;
    std::vector<int> move_history_;
    std::vector<uint64_t> position_history_;
    
    // Zobrist hashing
    core::ZobristHash zobrist_;
    mutable uint64_t hash_;
    mutable bool hash_dirty_;
    
    // Rules
    std::shared_ptr<GoRules> rules_;
    
    // Helper methods
    std::vector<int> getAdjacentPositions(int pos) const;
    bool isInBounds(int x, int y) const;
    bool isInBounds(int pos) const;
    void invalidateHash();
    void captureGroup(const StoneGroup& group);
    
    // Hash calculation
    void updateHash() const;
    
    // Check if a move is valid
    bool isValidMove(int action) const;
};

} // namespace go
} // namespace alphazero

#endif // GO_STATE_H