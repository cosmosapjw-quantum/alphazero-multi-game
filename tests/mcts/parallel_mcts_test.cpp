#include <gtest/gtest.h>
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/games/gomoku/gomoku_state.h"

namespace alphazero {
namespace mcts {

class ParallelMCTSTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a game state for Gomoku
        state = std::make_unique<gomoku::GomokuState>(15, false);
        
        // Create a neural network
        nn = nn::NeuralNetwork::create("", core::GameType::GOMOKU, 15);
        
        // Create a transposition table
        tt = std::make_unique<TranspositionTable>(1024, 16);
        
        // Create MCTS with 2 threads and 100 simulations
        mcts = std::make_unique<ParallelMCTS>(*state, nn.get(), tt.get(), 2, 100);
    }
    
    std::unique_ptr<gomoku::GomokuState> state;
    std::unique_ptr<nn::NeuralNetwork> nn;
    std::unique_ptr<TranspositionTable> tt;
    std::unique_ptr<ParallelMCTS> mcts;
};

TEST_F(ParallelMCTSTest, Search) {
    // Run search
    mcts->search();
    
    // Select an action
    int action = mcts->selectAction();
    
    // Action should be valid
    EXPECT_GE(action, 0);
    EXPECT_LT(action, state->getActionSpaceSize());
    
    // Make the move
    state->makeMove(action);
    
    // Update MCTS with the move
    mcts->updateWithMove(action);
    
    // Should be player 2's turn now
    EXPECT_EQ(state->getCurrentPlayer(), 2);
}

TEST_F(ParallelMCTSTest, ActionProbabilities) {
    // Run search
    mcts->search();
    
    // Get action probabilities with temperature 1.0
    auto probs = mcts->getActionProbabilities(1.0f);
    
    // Should have probabilities for all actions
    EXPECT_EQ(probs.size(), state->getActionSpaceSize());
    
    // Sum should be close to 1.0
    float sum = 0.0f;
    for (float p : probs) {
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 0.01f);
    
    // Get action probabilities with temperature 0.1 (more deterministic)
    auto probs2 = mcts->getActionProbabilities(0.1f);
    
    // Should be more concentrated than with temperature 1.0
    // Calculate entropy as a proxy for concentration
    float entropy1 = 0.0f;
    float entropy2 = 0.0f;
    
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > 0.0f) {
            entropy1 -= probs[i] * std::log(probs[i]);
        }
        if (probs2[i] > 0.0f) {
            entropy2 -= probs2[i] * std::log(probs2[i]);
        }
    }
    
    // Lower temperature should have lower entropy (more concentrated)
    EXPECT_LT(entropy2, entropy1);
}

TEST_F(ParallelMCTSTest, DirichletNoise) {
    // Set deterministic mode for reproducible test
    mcts->setDeterministicMode(true);
    
    // Run search without noise
    mcts->search();
    auto probsWithoutNoise = mcts->getActionProbabilities(1.0f);
    
    // Reset MCTS
    mcts = std::make_unique<ParallelMCTS>(*state, nn.get(), tt.get(), 2, 100);
    mcts->setDeterministicMode(true);
    
    // Add Dirichlet noise
    mcts->addDirichletNoise(0.03f, 0.25f);
    
    // Run search with noise
    mcts->search();
    auto probsWithNoise = mcts->getActionProbabilities(1.0f);
    
    // The distributions should be different
    bool different = false;
    for (size_t i = 0; i < probsWithoutNoise.size(); ++i) {
        if (std::abs(probsWithoutNoise[i] - probsWithNoise[i]) > 0.01f) {
            different = true;
            break;
        }
    }
    
    EXPECT_TRUE(different);
}

TEST_F(ParallelMCTSTest, SelectActionTrainingVsEvaluation) {
    // Set deterministic mode for reproducible test
    mcts->setDeterministicMode(true);
    
    // Run search
    mcts->search();
    
    // Select action in evaluation mode (deterministic)
    int actionEval = mcts->selectAction(false, 0.0f);
    
    // Select actions in training mode with high temperature
    std::set<int> trainingActions;
    for (int i = 0; i < 10; ++i) {
        int actionTrain = mcts->selectAction(true, 10.0f);
        trainingActions.insert(actionTrain);
    }
    
    // Training mode with high temperature should produce variety
    EXPECT_GT(trainingActions.size(), 1);
    
    // Training mode with temperature 0 should match evaluation mode
    int actionTrainDeterministic = mcts->selectAction(true, 0.0f);
    EXPECT_EQ(actionTrainDeterministic, actionEval);
}

TEST_F(ParallelMCTSTest, SelfPlay) {
    // Play a full game of Gomoku (could take a while)
    // Set a smaller board and fewer simulations for the test
    state = std::make_unique<gomoku::GomokuState>(9, false);
    mcts = std::make_unique<ParallelMCTS>(*state, nn.get(), tt.get(), 2, 50);
    
    int moveCount = 0;
    const int maxMoves = 30; // Limit moves to avoid too long test
    
    while (!state->isTerminal() && moveCount < maxMoves) {
        // Run search
        mcts->search();
        
        // Select action
        int action = mcts->selectAction(true, 1.0f);
        
        // Make move
        state->makeMove(action);
        
        // Update MCTS
        mcts->updateWithMove(action);
        
        moveCount++;
    }
    
    // Game should either be terminal or hit move limit
    EXPECT_TRUE(state->isTerminal() || moveCount >= maxMoves);
}

} // namespace mcts
} // namespace alphazero