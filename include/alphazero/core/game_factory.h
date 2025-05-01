// game_factory.h
#ifndef GAME_FACTORY_H
#define GAME_FACTORY_H

#include "igamestate.h"
#include "registry.h"
#include <memory>
#include <string>

namespace alphazero {
namespace core {

/**
 * @brief Factory for creating game states
 */
class GameFactory {
public:
    /**
     * @brief Create a Gomoku game state
     * 
     * @param boardSize Board size (default: 15)
     * @param useRenju Whether to use Renju rules
     * @param useOmok Whether to use Omok rules
     * @param seed Random seed
     * @param useProLongOpening Whether to use Pro-Long opening
     * @return Unique pointer to Gomoku game state
     */
    static std::unique_ptr<IGameState> createGomokuState(
        int boardSize = 15,
        bool useRenju = false,
        bool useOmok = false,
        int seed = 0,
        bool useProLongOpening = false);
    
    /**
     * @brief Create a Chess game state
     * 
     * @param chess960 Whether to use Chess960 rules
     * @param fen FEN string for initial position
     * @return Unique pointer to Chess game state
     */
    static std::unique_ptr<IGameState> createChessState(
        bool chess960 = false,
        const std::string& fen = "");
    
    /**
     * @brief Create a Go game state
     * 
     * @param boardSize Board size (9, 13, or 19)
     * @param komi Komi value
     * @param chineseRules Whether to use Chinese rules
     * @return Unique pointer to Go game state
     */
    static std::unique_ptr<IGameState> createGoState(
        int boardSize = 19,
        float komi = 7.5f,
        bool chineseRules = true);
    
    /**
     * @brief Check if a game type is supported
     * 
     * @param type Game type to check
     * @return true if supported, false otherwise
     */
    static bool isGameSupported(GameType type);
    
    /**
     * @brief Get the default board size for a game type
     * 
     * @param type Game type
     * @return Default board size
     */
    static int getDefaultBoardSize(GameType type);
    
    /**
     * @brief Convert game type to string ID
     * 
     * @param type Game type
     * @return Game ID string
     */
    static GameId gameTypeToId(GameType type);
    
    /**
     * @brief Create a game state by ID
     * 
     * @param id Game identifier
     * @param args Game creation arguments
     * @return Unique pointer to game state
     */
    static std::unique_ptr<IGameState> createGame(
        const GameId& id, const VariantArgs& args = {});
};

/**
 * @brief Create a game state (legacy function)
 * 
 * @param type Game type
 * @param boardSize Board size (0 for default)
 * @param variantRules Whether to use variant rules
 * @return Unique pointer to created game state
 */
std::unique_ptr<IGameState> createGameState(GameType type, 
                                           int boardSize = 0, 
                                           bool variantRules = false);

} // namespace core
} // namespace alphazero

#endif // GAME_FACTORY_H