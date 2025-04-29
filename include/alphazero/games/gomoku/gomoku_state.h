// gomoku_state.h
#ifndef GOMOKU_STATE_H
#define GOMOKU_STATE_H

#include <vector>
#include <set>
#include <cstdint>
#include <utility>
#include <unordered_set>
#include <memory>

#include "alphazero/core/igamestate.h"
#include "alphazero/core/utils/hash_specializations.h"
#include "alphazero/games/gomoku/gomoku_rules.h"

namespace alphazero {
namespace gomoku {

// Constants
const int BLACK = 1;
const int WHITE = 2;

/**
 * @brief Implementation of Gomoku game state
 */
class GomokuState : public core::IGameState {
public:
    /**
     * @brief Constructor
     * 
     * @param board_size Board size
     * @param use_renju Whether to use Renju rules
     * @param use_omok Whether to use Omok rules
     * @param seed Random seed for initialization
     * @param use_pro_long_opening Whether to enforce pro-long opening
     */
    GomokuState(int board_size = 15, 
                bool use_renju = false, 
                bool use_omok = false, 
                int seed = 0, 
                bool use_pro_long_opening = false);
    
    /**
     * @brief Copy constructor
     */
    GomokuState(const GomokuState& other);
    
    /**
     * @brief Assignment operator
     */
    GomokuState& operator=(const GomokuState& other);
    
    // IGameState interface implementation
    std::vector<int> getLegalMoves() const override;
    bool isLegalMove(int action) const override;
    void makeMove(int action) override;
    bool undoMove() override;
    bool isTerminal() const override;
    core::GameResult getGameResult() const override;
    int getCurrentPlayer() const override { return current_player; }
    int getBoardSize() const override { return board_size; }
    int getActionSpaceSize() const override { return board_size * board_size; }
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override;
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override;
    uint64_t getHash() const override;
    std::unique_ptr<core::IGameState> clone() const override;
    std::string actionToString(int action) const override;
    std::optional<int> stringToAction(const std::string& moveStr) const override;
    std::string toString() const override;
    bool equals(const core::IGameState& other) const override;
    std::vector<int> getMoveHistory() const override { return move_history; }
    bool validate() const override;
    
    // Public fields
    int board_size;
    int current_player;        // 1=BLACK, 2=WHITE
    std::vector<std::vector<uint64_t>> player_bitboards;  // shape=(2, num_words)
    int num_words;
    int action;                // last move made, or -1 if none
    
    bool use_renju;
    bool use_omok;
    bool use_pro_long_opening;
    int black_first_stone;
    
    // Bitboard operations
    bool is_bit_set(int player_index, int action) const noexcept;
    void set_bit(int player_index, int action);
    void clear_bit(int player_index, int action) noexcept;
    
    std::pair<int, int> action_to_coords_pair(int action) const noexcept;
    int coords_to_action(int x, int y) const;
    int count_total_stones() const noexcept;
    
    // Public interface for game play
    bool board_equal(const GomokuState& other) const;
    bool is_occupied(int action) const;
    bool is_stalemate() const;
    std::vector<std::vector<int>> get_board() const;
    
    // For MCTS integration
    GomokuState apply_action(int action) const;
    std::vector<std::vector<std::vector<float>>> to_tensor() const;
    int get_action(const GomokuState& child_state) const;
    
    // Deep copy (keeping for backward compatibility)
    GomokuState copy() const;
    
    // Get previous moves for a player
    std::vector<int> get_previous_moves(int player, int count = 7) const;
    
    // Expose rules checker for pattern recognition
    std::shared_ptr<GomokuRules> getRules() const { return rules; }
    
    // Check if Renju rules are being used
    bool isUsingRenjuRules() const { return use_renju; }
    
    // Compute and update hash signature
    uint64_t compute_hash_signature() const;

private:
    // Cached valid moves to avoid recomputing
    mutable std::unordered_set<int> cached_valid_moves;
    mutable bool valid_moves_dirty; // Flag to indicate if cache needs refreshing
    
    // Cached terminal state status
    mutable int cached_winner;
    mutable bool winner_check_dirty;
    
    // Cached board representation for faster equality checks
    mutable uint64_t hash_signature;
    mutable bool hash_dirty;
    
    // Move history
    std::vector<int> move_history;
    
    // Rules implementation
    std::shared_ptr<GomokuRules> rules;
    
    // Directions array
    int dirs[8];
    
    // Invalidate caches after state changes
    void invalidate_caches();
    
    // Refresh valid moves cache
    void refresh_valid_moves_cache() const;
    
    // Refresh winner cache
    void refresh_winner_cache() const;
    
    // Helper methods
    bool in_bounds(int x, int y) const;
    bool is_pro_long_move_ok(int action, int stone_count) const;
    
    // Private implementation aliases for interface methods
    bool is_terminal() const;
    int get_winner() const;
    std::vector<int> get_valid_moves() const;
    bool is_move_valid(int action) const;
    void make_move(int action, int player);
    void undo_move(int action);
};

} // namespace gomoku
} // namespace alphazero

#endif // GOMOKU_STATE_H