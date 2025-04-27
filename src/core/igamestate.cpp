// igamestate.cpp
#include "alphazero/core/igamestate.h"
#include "alphazero/games/gomoku/gomoku_state.h"
#include <stdexcept>
#include <string>

namespace alphazero {
namespace core {

std::unique_ptr<IGameState> createGameState(GameType type, int boardSize, bool variantRules) {
    try {
        switch (type) {
            case GameType::GOMOKU:
                return std::make_unique<alphazero::gomoku::GomokuState>(
                    boardSize > 0 ? boardSize : 15,  // Default 15x15 for Gomoku
                    variantRules,  // Renju rules if true
                    false,         // Omok rules - disabled by default
                    0,             // Random seed
                    false          // Pro-Long opening - disabled by default
                );
            
            case GameType::CHESS:
                // TODO: Implement ChessState
                throw std::runtime_error("Chess not yet implemented");
            
            case GameType::GO:
                // TODO: Implement GoState
                throw std::runtime_error("Go not yet implemented");
            
            default:
                throw std::invalid_argument("Unsupported game type");
        }
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create game state: " + std::string(e.what()));
    }
}

} // namespace core
} // namespace alphazero