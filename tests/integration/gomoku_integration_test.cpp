#include <gtest/gtest.h>
#include "alphazero/core/igamestate.h"
#include "alphazero/games/gomoku/gomoku_state.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/mcts/transposition_table.h"
#include "alphazero/selfplay/game_record.h"

namespace alphazero {
namespace integration {

class GomokuIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create game state
        gameState = core::createGameState(core::GameType::GOMOKU, 9);
        
        // Create neural network
        network = nn::NeuralNetwork::create("", core::GameType::GOMOKU, 9);
        
        // Create transposition table
        tt = std::make_unique<mcts::TranspositionTable>(1024, 16);
        
        // Create MCTS with 2 threads and fewer simulations for testing
        mcts = std::make_unique<mcts::ParallelMCTS>(*gameState, network.get(), tt.get(), 2, 50);
    }
    
    std::unique_ptr<core::IGameState> gameState;
    std::unique_ptr<nn::NeuralNetwork> network;
    std::unique_ptr<mcts::TranspositionTable> tt;
    std::unique_ptr<mcts::ParallelMCTS> mcts;
};

TEST_F(GomokuIntegrationTest, SelfPlayAndGameRecord) {
    // Create a game record
    selfplay::GameRecord record(core::GameType::GOMOKU, 9, false);
    
    // Play a partial game and record moves
    const int movesToPlay = 5;
    
    for (int i = 0; i < movesToPlay && !gameState->isTerminal(); ++i) {
        // Run search
        mcts->search();
        
        // Get action probabilities
        std::vector<float> probs = mcts->getActionProbabilities(1.0f);
        
        // Select move
        int action = mcts->selectAction(true, 1.0f);
        
        // Get value estimate
        float value = mcts->getRootValue();
        
        // Record the move
        record.addMove(action, probs, value, 0);
        
        // Make the move
        gameState->makeMove(action);
        
        // Update MCTS
        mcts->updateWithMove(action);
    }
    
    // Record should have the correct number of moves
    EXPECT_EQ(record.getMoves().size(), std::min(movesToPlay, static_cast<int>(gameState->getMoveHistory().size())));
    
    // Convert record to JSON
    std::string json = record.toJson();
    EXPECT_FALSE(json.empty());
    
    // Parse JSON back to a record
    selfplay::GameRecord parsedRecord = selfplay::GameRecord::fromJson(json);
    
    // Should have same metadata
    auto [gameType, boardSize, useVariantRules] = parsedRecord.getMetadata();
    EXPECT_EQ(gameType, core::GameType::GOMOKU);
    EXPECT_EQ(boardSize, 9);
    EXPECT_FALSE(useVariantRules);
    
    // Should have same number of moves
    EXPECT_EQ(parsedRecord.getMoves().size(), record.getMoves().size());
    
    // Moves should match
    for (size_t i = 0; i < record.getMoves().size(); ++i) {
        EXPECT_EQ(parsedRecord.getMoves()[i].action, record.getMoves()[i].action);
        EXPECT_FLOAT_EQ(parsedRecord.getMoves()[i].value, record.getMoves()[i].value);
    }
}

TEST_F(GomokuIntegrationTest, PlayAgainstSelf) {
    // Reset state and MCTS
    gameState = core::createGameState(core::GameType::GOMOKU, 9);
    mcts = std::make_unique<mcts::ParallelMCTS>(*gameState, network.get(), tt.get(), 2, 50);
    
    // Play a full game unless it takes too long
    const int maxMoves = 20;
    int moveCount = 0;
    
    // Keep track of positions for repetition detection
    std::vector<uint64_t> positionHashes;
    positionHashes.push_back(gameState->getHash());
    
    while (!gameState->isTerminal() && moveCount < maxMoves) {
        // Run search
        mcts->search();
        
        // Select move
        int action = mcts->selectAction(false, 0.0f);
        
        // Make the move
        gameState->makeMove(action);
        
        // Update MCTS
        mcts->updateWithMove(action);
        
        // Store hash
        positionHashes.push_back(gameState->getHash());
        
        // Check for repetitions (unlikely in Gomoku but good practice)
        bool hasRepetition = false;
        for (size_t i = 0; i < positionHashes.size() - 1; ++i) {
            if (positionHashes[i] == positionHashes.back()) {
                hasRepetition = true;
                break;
            }
        }
        EXPECT_FALSE(hasRepetition) << "Position repetition detected";
        
        moveCount++;
    }
    
    // Game should either be terminal or hit move limit
    EXPECT_TRUE(gameState->isTerminal() || moveCount >= maxMoves);
    
    // If game is terminal, result should be valid
    if (gameState->isTerminal()) {
        auto result = gameState->getGameResult();
        EXPECT_TRUE(result == core::GameResult::WIN_PLAYER1 || 
                    result == core::GameResult::WIN_PLAYER2 || 
                    result == core::GameResult::DRAW);
    }
}

TEST_F(GomokuIntegrationTest, TensorRepresentations) {
    // Get tensor representation
    auto tensor = gameState->getTensorRepresentation();
    
    // Basic shape checks
    EXPECT_GT(tensor.size(), 0); // At least one plane
    EXPECT_GT(tensor[0].size(), 0); // Non-empty planes
    EXPECT_GT(tensor[0][0].size(), 0);
    
    // Dimensions should match the board size
    EXPECT_EQ(tensor[0].size(), gameState->getBoardSize());
    EXPECT_EQ(tensor[0][0].size(), gameState->getBoardSize());
    
    // Get enhanced tensor representation
    auto enhancedTensor = gameState->getEnhancedTensorRepresentation();
    
    // Should have more planes than basic representation
    EXPECT_GT(enhancedTensor.size(), tensor.size());
    
    // Neural network should accept the tensor
    auto [policy, value] = network->predict(*gameState);
    
    // Policy should have size equal to action space
    EXPECT_EQ(policy.size(), gameState->getActionSpaceSize());
    
    // Value should be in range [-1, 1]
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
}

} // namespace integration
} // namespace alphazero