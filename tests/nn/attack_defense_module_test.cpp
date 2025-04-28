#include <gtest/gtest.h>
#include "alphazero/nn/attack_defense_module.h"
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace nn {

class AttackDefenseModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a module for a 9x9 board
        module = std::make_unique<AttackDefenseModule>(9);
        
        // Create an empty 9x9 board representation
        board_batch = {std::vector<std::vector<int>>(9, std::vector<int>(9, 0))};
        
        // Default move in center
        chosen_moves = {4 * 9 + 4};
        
        // Default player (BLACK = 1)
        player_batch = {1};
    }
    
    std::unique_ptr<AttackDefenseModule> module;
    std::vector<std::vector<std::vector<int>>> board_batch;
    std::vector<int> chosen_moves;
    std::vector<int> player_batch;
};

TEST_F(AttackDefenseModuleTest, EmptyBoard) {
    // On an empty board, there should be no attack or defense bonuses
    auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
    
    EXPECT_EQ(attack.size(), 1);
    EXPECT_EQ(defense.size(), 1);
    
    // No threats on empty board
    EXPECT_FLOAT_EQ(attack[0], 0.0f);
    EXPECT_FLOAT_EQ(defense[0], 0.0f);
}

TEST_F(AttackDefenseModuleTest, AttackingMove) {
    // Create a position where the center move creates a threat
    // Place two BLACK stones in a row, then the center move would create a three-in-a-row
    board_batch[0][4][2] = 1; // BLACK
    board_batch[0][4][3] = 1; // BLACK
    
    // The chosen move is at (4,4) which would create a three-in-a-row
    chosen_moves[0] = 4 * 9 + 4;
    
    // Compute bonuses
    auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
    
    // Should detect an attack (creating a new threat)
    EXPECT_GT(attack[0], 0.0f);
    // No defense bonus expected here
    EXPECT_FLOAT_EQ(defense[0], 0.0f);
}

TEST_F(AttackDefenseModuleTest, DefensiveMove) {
    // Create a position where the center move blocks an opponent's threat
    // Place two WHITE stones in a row, and the center move would block a potential win
    board_batch[0][4][2] = 2; // WHITE
    board_batch[0][4][3] = 2; // WHITE
    
    // The chosen move is at (4,4) which would block WHITE's potential three-in-a-row
    chosen_moves[0] = 4 * 9 + 4;
    
    // BLACK is the current player
    player_batch[0] = 1;
    
    // Compute bonuses
    auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
    
    // Should detect a defense (blocking an opponent's threat)
    EXPECT_GT(defense[0], 0.0f);
}

TEST_F(AttackDefenseModuleTest, BothAttackAndDefense) {
    // Create a position where center move both creates a threat and blocks one
    // Two BLACK stones in a row on one line
    board_batch[0][4][2] = 1; // BLACK
    board_batch[0][4][3] = 1; // BLACK
    
    // Two WHITE stones in a row on another line
    board_batch[0][2][4] = 2; // WHITE
    board_batch[0][3][4] = 2; // WHITE
    
    // The center move at (4,4) both extends BLACK's stones and blocks WHITE's threat
    chosen_moves[0] = 4 * 9 + 4;
    
    // BLACK is the current player
    player_batch[0] = 1;
    
    // Compute bonuses
    auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
    
    // Should detect both attack and defense
    EXPECT_GT(attack[0], 0.0f);
    EXPECT_GT(defense[0], 0.0f);
}

TEST_F(AttackDefenseModuleTest, MultipleBatchElements) {
    // Test with multiple boards in the batch
    
    // Create a second board
    board_batch.push_back(std::vector<std::vector<int>>(9, std::vector<int>(9, 0)));
    
    // First board: attacking position
    board_batch[0][4][2] = 1; // BLACK
    board_batch[0][4][3] = 1; // BLACK
    
    // Second board: defensive position
    board_batch[1][4][2] = 2; // WHITE
    board_batch[1][4][3] = 2; // WHITE
    
    // Chosen moves for both boards
    chosen_moves = {4 * 9 + 4, 4 * 9 + 4};
    
    // Players for both boards
    player_batch = {1, 1}; // BLACK for both
    
    // Compute bonuses
    auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
    
    // Should have bonuses for both boards
    EXPECT_EQ(attack.size(), 2);
    EXPECT_EQ(defense.size(), 2);
    
    // First board: attack bonus, no defense
    EXPECT_GT(attack[0], 0.0f);
    EXPECT_FLOAT_EQ(defense[0], 0.0f);
    
    // Second board: defense bonus, no attack
    EXPECT_FLOAT_EQ(attack[1], 0.0f);
    EXPECT_GT(defense[1], 0.0f);
}

TEST_F(AttackDefenseModuleTest, DiagonalThreats) {
    // Create a diagonal threat pattern
    board_batch[0][2][2] = 1; // BLACK
    board_batch[0][3][3] = 1; // BLACK
    
    // The chosen move is at (4,4) which would create a three-in-a-row diagonally
    chosen_moves[0] = 4 * 9 + 4;
    
    // Compute bonuses
    auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
    
    // Should detect an attack in the diagonal
    EXPECT_GT(attack[0], 0.0f);
}

} // namespace nn
} // namespace alphazero