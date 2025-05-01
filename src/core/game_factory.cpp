// game_factory.cpp
#include "alphazero/core/game_factory.h"
#include <stdexcept>

namespace alphazero {
namespace core {

GameId GameFactory::gameTypeToId(GameType type) {
    switch (type) {
        case GameType::GOMOKU: return "gomoku";
        case GameType::CHESS: return "chess";
        case GameType::GO: return "go";
        default: throw std::invalid_argument("Unknown game type");
    }
}

std::unique_ptr<IGameState> GameFactory::createGame(const GameId& id, const VariantArgs& args) {
    auto result = GameRegistry::instance().createGame(id, args);
    if (!result) {
        throw GameStateException("Failed to create game: " + id);
    }
    return result;
}

std::unique_ptr<IGameState> GameFactory::createGomokuState(
    int boardSize, bool useRenju, bool useOmok, int seed, bool useProLongOpening) {
    
    VariantArgs args;
    args.set("boardSize", boardSize);
    args.set("useRenju", useRenju);
    args.set("useOmok", useOmok);
    args.set("seed", seed);
    args.set("useProLongOpening", useProLongOpening);
    
    try {
        return createGame("gomoku", args);
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create Gomoku state: " + std::string(e.what()));
    }
}

std::unique_ptr<IGameState> GameFactory::createChessState(
    bool chess960, const std::string& fen) {
    
    VariantArgs args;
    args.set("chess960", chess960);
    if (!fen.empty()) {
        args.set("fen", fen);
    }
    
    try {
        return createGame("chess", args);
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create Chess state: " + std::string(e.what()));
    }
}

std::unique_ptr<IGameState> GameFactory::createGoState(
    int boardSize, float komi, bool chineseRules) {
    
    VariantArgs args;
    args.set("boardSize", boardSize);
    args.set("komi", komi);
    args.set("chineseRules", chineseRules);
    
    try {
        return createGame("go", args);
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create Go state: " + std::string(e.what()));
    }
}

bool GameFactory::isGameSupported(GameType type) {
    try {
        return GameRegistry::instance().hasGame(gameTypeToId(type));
    } catch (const std::exception&) {
        return false;
    }
}

int GameFactory::getDefaultBoardSize(GameType type) {
    switch (type) {
        case GameType::GOMOKU: return 15;
        case GameType::CHESS: return 8;
        case GameType::GO: return 19;
        default: throw std::invalid_argument("Unsupported game type");
    }
}

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