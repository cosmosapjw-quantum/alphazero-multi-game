// compatibility.h
#pragma once
#include "registry.h"
#include <stdexcept>

namespace alphazero::core {

// Legacy enum type
enum class GameType {
    GOMOKU,
    CHESS,
    GO
};

// Conversion to string ID
inline GameId gameTypeToId(GameType type) {
    switch (type) {
        case GameType::GOMOKU: return "gomoku";
        case GameType::CHESS: return "chess";
        case GameType::GO: return "go";
        default: throw std::invalid_argument("Unknown game type");
    }
}

// Legacy factory function
inline std::unique_ptr<IGameState> createGameState(
    GameType type, int boardSize = 0, bool variantRules = false) {
    
    VariantArgs args;
    if (boardSize > 0) {
        args.set("boardSize", boardSize);
    }
    args.set("variantRules", variantRules);
    
    return GameRegistry::instance().createGame(gameTypeToId(type), args);
}

} // namespace alphazero::core