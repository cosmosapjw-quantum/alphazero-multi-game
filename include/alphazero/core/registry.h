// registry.h
#ifndef REGISTRY_H
#define REGISTRY_H

#include "igamestate.h"
#include "variant_args.h"
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace alphazero {
namespace core {

/**
 * @brief Game identifier type
 */
using GameId = std::string;

/**
 * @brief Registry for game factories
 * 
 * This class maintains a thread-safe registry of game creation functions,
 * allowing games to be registered and instantiated by ID.
 */
class GameRegistry {
public:
    /**
     * @brief Factory function type
     */
    using FactoryFn = std::function<std::unique_ptr<IGameState>(const VariantArgs&)>;
    
    /**
     * @brief Get singleton instance
     * 
     * @return Reference to the registry
     */
    static GameRegistry& instance() {
        static GameRegistry inst;
        return inst;
    }
    
    /**
     * @brief Register a game factory
     * 
     * @param id Game identifier
     * @param factoryFn Factory function
     * @return true if registered, false if ID already exists
     */
    bool registerGame(const GameId& id, FactoryFn factoryFn);
    
    /**
     * @brief Unregister a game
     * 
     * @param id Game identifier
     * @return true if unregistered, false if ID not found
     */
    bool unregisterGame(const GameId& id);
    
    /**
     * @brief Create a game instance
     * 
     * @param id Game identifier
     * @param args Arguments for game creation
     * @return Unique pointer to game state, or nullptr if ID not found
     */
    std::unique_ptr<IGameState> createGame(const GameId& id, const VariantArgs& args = {});
    
    /**
     * @brief Check if a game ID is registered
     * 
     * @param id Game identifier
     * @return true if registered, false otherwise
     */
    bool hasGame(const GameId& id) const;
    
    /**
     * @brief Get all registered game IDs
     * 
     * @return Vector of game IDs
     */
    std::vector<GameId> getGameIds() const;
    
private:
    /**
     * @brief Private constructor for singleton
     */
    GameRegistry() = default;
    
    /**
     * @brief Copy constructor (deleted)
     */
    GameRegistry(const GameRegistry&) = delete;
    
    /**
     * @brief Assignment operator (deleted)
     */
    GameRegistry& operator=(const GameRegistry&) = delete;
    
    /**
     * @brief Thread synchronization
     */
    mutable std::shared_mutex mutex_;
    
    /**
     * @brief Registry storage
     */
    std::unordered_map<GameId, FactoryFn> registry_;
};

} // namespace core
} // namespace alphazero

#endif // REGISTRY_H