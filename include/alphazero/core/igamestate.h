// igamestate.h
#ifndef IGAMESTATE_H
#define IGAMESTATE_H

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <stdexcept>

namespace alphazero {
namespace core {

/**
 * @brief Game type identifiers - maintained for backward compatibility
 */
enum class GameType {
    GOMOKU,
    CHESS,
    GO
};

/**
 * @brief Result of a game
 */
enum class GameResult {
    ONGOING,
    DRAW,
    WIN_PLAYER1,
    WIN_PLAYER2
};

/**
 * @brief Exception for game state errors
 */
class GameStateException : public std::runtime_error {
public:
    explicit GameStateException(const std::string& message) 
        : std::runtime_error(message) {}
};

/**
 * @brief Exception for illegal move attempts
 */
class IllegalMoveException : public GameStateException {
public:
    IllegalMoveException(const std::string& message, int action) 
        : GameStateException(message), action_(action) {}
    int getAction() const { return action_; }
private:
    int action_;
};

/**
 * @brief Interface for game state implementations
 * 
 * This abstract class defines the interface that all game implementations
 * must implement to work with the AI engine.
 */
class IGameState {
public:
    /**
     * @brief Constructor
     * 
     * @param type Game type
     */
    explicit IGameState(GameType type) : gameType_(type) {}
    
    /**
     * @brief Virtual destructor
     */
    virtual ~IGameState() = default;
    
    // Core game state methods that all games must implement
    
    /**
     * @brief Get all legal moves from current state
     * 
     * @return Vector of legal actions
     */
    virtual std::vector<int> getLegalMoves() const = 0;
    
    /**
     * @brief Check if a specific move is legal
     * 
     * @param action The action to check
     * @return true if move is legal, false otherwise
     */
    virtual bool isLegalMove(int action) const = 0;
    
    /**
     * @brief Execute a move, updating the game state
     * 
     * @param action The action to execute
     * @throws IllegalMoveException if the move is illegal
     */
    virtual void makeMove(int action) = 0;
    
    /**
     * @brief Undo the last move
     * 
     * @return true if a move was undone, false if no moves to undo
     */
    virtual bool undoMove() = 0;
    
    /**
     * @brief Check if the game state is terminal (game over)
     * 
     * @return true if terminal, false otherwise
     */
    virtual bool isTerminal() const = 0;
    
    /**
     * @brief Get the result of the game
     * 
     * @return Game result enum
     */
    virtual GameResult getGameResult() const = 0;
    
    /**
     * @brief Get the current player
     * 
     * @return Current player (1=player1, 2=player2)
     */
    virtual int getCurrentPlayer() const = 0;
    
    /**
     * @brief Get the board size
     * 
     * @return Board size (typically width or height)
     */
    virtual int getBoardSize() const = 0;
    
    /**
     * @brief Get the action space size
     * 
     * @return Total number of possible actions
     */
    virtual int getActionSpaceSize() const = 0;
    
    /**
     * @brief Get tensor representation for neural network
     * 
     * @return 3D tensor representation of the game state
     */
    virtual std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const = 0;
    
    /**
     * @brief Get enhanced tensor representation with additional features
     * 
     * @return Enhanced 3D tensor representation
     */
    virtual std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const = 0;
    
    /**
     * @brief Get hash for transposition table
     * 
     * @return 64-bit hash of the game state
     */
    virtual uint64_t getHash() const = 0;
    
    /**
     * @brief Clone the current state
     * 
     * @return Unique pointer to a new copy of this state
     */
    virtual std::unique_ptr<IGameState> clone() const = 0;
    
    /**
     * @brief Convert action to string representation
     * 
     * @param action The action to convert
     * @return String representation of the action
     */
    virtual std::string actionToString(int action) const = 0;
    
    /**
     * @brief Convert string representation to action
     * 
     * @param moveStr String representation of a move
     * @return Optional action (empty if invalid string)
     */
    virtual std::optional<int> stringToAction(const std::string& moveStr) const = 0;
    
    /**
     * @brief Get string representation of the game state
     * 
     * @return String representation
     */
    virtual std::string toString() const = 0;
    
    /**
     * @brief Check equality with another game state
     * 
     * @param other The other game state to compare with
     * @return true if equal, false otherwise
     */
    virtual bool equals(const IGameState& other) const = 0;
    
    /**
     * @brief Get the history of moves
     * 
     * @return Vector of actions representing move history
     */
    virtual std::vector<int> getMoveHistory() const = 0;
    
    /**
     * @brief Validate the game state for consistency
     * 
     * @return true if valid, false otherwise
     */
    virtual bool validate() const = 0;
    
    /**
     * @brief Get the game type
     * 
     * @return Game type enum
     */
    GameType getGameType() const { return gameType_; }
    
protected:
    GameType gameType_;  // Type of game this state represents
};

} // namespace core
} // namespace alphazero

#endif // IGAMESTATE_H