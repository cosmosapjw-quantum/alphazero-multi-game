// igamestate.cpp
#include "alphazero/core/igamestate.h"
#include "alphazero/core/game_factory.h"
#include "alphazero/core/registry.h"
#include <stdexcept>

namespace alphazero {
namespace core {

std::unique_ptr<IGameState> createGameState(GameType type, int boardSize, bool variantRules) {
    try {
        // Convert the enum-based call to the new string-based registry system
        VariantArgs args;
        
        // Set common arguments
        if (boardSize > 0) {
            args.set("boardSize", boardSize);
        }
        
        args.set("variantRules", variantRules);
        
        // Game-specific parameter handling
        switch (type) {
            case GameType::GOMOKU:
                // For Gomoku, variantRules means Renju rules
                args.set("useRenju", variantRules);
                args.set("useOmok", false);
                args.set("seed", 0);
                args.set("useProLongOpening", false);
                
                if (boardSize <= 0) {
                    args.set("boardSize", 15);  // Default for Gomoku
                }
                break;
                
            case GameType::CHESS:
                // For Chess, variantRules means Chess960
                args.set("chess960", variantRules);
                break;
                
            case GameType::GO:
                // For Go, set default komi and rules
                args.set("komi", 7.5f);
                args.set("chineseRules", true);
                
                if (boardSize <= 0) {
                    args.set("boardSize", 19);  // Default for Go
                }
                break;
                
            default:
                throw std::invalid_argument("Unsupported game type");
        }
        
        // Create the game using the registry
        GameId gameId = GameFactory::gameTypeToId(type);
        auto game = GameRegistry::instance().createGame(gameId, args);
        
        if (!game) {
            throw GameStateException("Failed to create game for type: " + gameId);
        }
        
        return game;
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create game state: " + std::string(e.what()));
    }
}

} // namespace core
} // namespace alphazero