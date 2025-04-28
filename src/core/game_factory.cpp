// src/core/game_factory.cpp
#include "alphazero/core/game_factory.h"
#include "alphazero/games/gomoku/gomoku_state.h"
#include "alphazero/games/chess/chess_state.h"
#include "alphazero/games/go/go_state.h"
#include <stdexcept>

namespace alphazero {
namespace core {

std::unique_ptr<IGameState> GameFactory::createGomokuState(
    int boardSize,
    bool useRenju,
    bool useOmok,
    int seed,
    bool useProLongOpening) {
    
    try {
        return std::make_unique<alphazero::gomoku::GomokuState>(
            boardSize,
            useRenju,
            useOmok,
            seed,
            useProLongOpening
        );
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create Gomoku state: " + std::string(e.what()));
    }
}

std::unique_ptr<IGameState> GameFactory::createChessState(
    bool chess960,
    const std::string& fen) {
    
    try {
        return std::make_unique<alphazero::chess::ChessState>(
            chess960,
            fen
        );
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create Chess state: " + std::string(e.what()));
    }
}

std::unique_ptr<IGameState> GameFactory::createGoState(
    int boardSize,
    float komi,
    bool chineseRules) {
    
    try {
        return std::make_unique<alphazero::go::GoState>(
            boardSize,
            komi,
            chineseRules
        );
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create Go state: " + std::string(e.what()));
    }
}

bool GameFactory::isGameSupported(GameType type) {
    switch (type) {
        case GameType::GOMOKU:
        case GameType::CHESS:
        case GameType::GO:
            return true;
        default:
            return false;
    }
}

int GameFactory::getDefaultBoardSize(GameType type) {
    switch (type) {
        case GameType::GOMOKU:
            return 15;
        case GameType::CHESS:
            return 8;
        case GameType::GO:
            return 19;
        default:
            throw std::invalid_argument("Unsupported game type");
    }
}

} // namespace core
} // namespace alphazero