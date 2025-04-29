#include <gtest/gtest.h>
#include "alphazero/core/game_factory.h"

namespace alphazero {
namespace core {

class GameFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Nothing needed for setup
    }
};

TEST_F(GameFactoryTest, CreateGomokuState) {
    // Test creating Gomoku states with different parameters
    
    // Default parameters
    auto gomoku1 = GameFactory::createGomokuState();
    EXPECT_NE(gomoku1, nullptr);
    EXPECT_EQ(gomoku1->getGameType(), GameType::GOMOKU);
    EXPECT_EQ(gomoku1->getBoardSize(), 15); // Default size
    
    // Custom board size
    auto gomoku2 = GameFactory::createGomokuState(9);
    EXPECT_EQ(gomoku2->getBoardSize(), 9);
    
    // With Renju rules
    auto gomoku3 = GameFactory::createGomokuState(15, true);
    EXPECT_NE(gomoku3, nullptr);
    
    // With Omok rules
    auto gomoku4 = GameFactory::createGomokuState(15, false, true);
    EXPECT_NE(gomoku4, nullptr);
}

TEST_F(GameFactoryTest, CreateChessState) {
    // Test creating Chess states with different parameters
    
    // Default parameters
    auto chess1 = GameFactory::createChessState();
    EXPECT_NE(chess1, nullptr);
    EXPECT_EQ(chess1->getGameType(), GameType::CHESS);
    EXPECT_EQ(chess1->getBoardSize(), 8); // Always 8x8
    
    // With Chess960 rules
    auto chess2 = GameFactory::createChessState(true);
    EXPECT_NE(chess2, nullptr);
    
    // With custom FEN
    auto chess3 = GameFactory::createChessState(false, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    EXPECT_NE(chess3, nullptr);
}

TEST_F(GameFactoryTest, CreateGoState) {
    // Test creating Go states with different parameters
    
    // Default parameters
    auto go1 = GameFactory::createGoState();
    EXPECT_NE(go1, nullptr);
    EXPECT_EQ(go1->getGameType(), GameType::GO);
    EXPECT_EQ(go1->getBoardSize(), 19); // Default size
    
    // Custom board size
    auto go2 = GameFactory::createGoState(9);
    EXPECT_EQ(go2->getBoardSize(), 9);
    
    // With custom komi
    auto go3 = GameFactory::createGoState(19, 6.5);
    EXPECT_NE(go3, nullptr);
}

TEST_F(GameFactoryTest, CreateGameState) {
    // Test the generic createGameState function
    
    // Gomoku with default parameters
    auto gomoku = createGameState(GameType::GOMOKU);
    EXPECT_NE(gomoku, nullptr);
    EXPECT_EQ(gomoku->getGameType(), GameType::GOMOKU);
    
    // Chess with default parameters
    auto chess = createGameState(GameType::CHESS);
    EXPECT_NE(chess, nullptr);
    EXPECT_EQ(chess->getGameType(), GameType::CHESS);
    
    // Go with default parameters
    auto go = createGameState(GameType::GO);
    EXPECT_NE(go, nullptr);
    EXPECT_EQ(go->getGameType(), GameType::GO);
    
    // Custom board size
    auto gomoku2 = createGameState(GameType::GOMOKU, 9);
    EXPECT_EQ(gomoku2->getBoardSize(), 9);
    
    // Variant rules
    auto chess2 = createGameState(GameType::CHESS, 0, true);
    EXPECT_NE(chess2, nullptr);
}

TEST_F(GameFactoryTest, IsGameSupported) {
    // Test checking if a game type is supported
    
    EXPECT_TRUE(GameFactory::isGameSupported(GameType::GOMOKU));
    EXPECT_TRUE(GameFactory::isGameSupported(GameType::CHESS));
    EXPECT_TRUE(GameFactory::isGameSupported(GameType::GO));
}

TEST_F(GameFactoryTest, GetDefaultBoardSize) {
    // Test getting default board size for different games
    
    EXPECT_EQ(GameFactory::getDefaultBoardSize(GameType::GOMOKU), 15);
    EXPECT_EQ(GameFactory::getDefaultBoardSize(GameType::CHESS), 8);
    EXPECT_EQ(GameFactory::getDefaultBoardSize(GameType::GO), 19);
    
    // Invalid game type should throw
    EXPECT_THROW(GameFactory::getDefaultBoardSize(static_cast<GameType>(-1)), std::invalid_argument);
}

} // namespace core
} // namespace alphazero