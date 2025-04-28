// igamestate.cpp
#include "alphazero/core/igamestate.h"
#include "alphazero/core/game_factory.h"
#include <stdexcept>

namespace alphazero {
namespace core {

std::unique_ptr<IGameState> createGameState(GameType type, int boardSize, bool variantRules) {
    try {
        switch (type) {
            case GameType::GOMOKU:
                return GameFactory::createGomokuState(
                    boardSize > 0 ? boardSize : 15,  // Default 15x15 for Gomoku
                    variantRules,  // Renju rules if true
                    false,         // Omok rules - disabled by default
                    0,             // Random seed
                    false          // Pro-Long opening - disabled by default
                );
            
            case GameType::CHESS:
                return GameFactory::createChessState(
                    variantRules  // Chess960 rules if true
                );
            
            case GameType::GO:
                return GameFactory::createGoState(
                    boardSize > 0 ? boardSize : 19,  // Default 19x19 for Go
                    7.5f,         // Default komi
                    true          // Chinese rules by default
                );
            
            default:
                throw std::invalid_argument("Unsupported game type");
        }
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create game state: " + std::string(e.what()));
    }
}

} // namespace core
} // namespace alphazero