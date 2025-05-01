// registry.cpp
#include "alphazero/core/registry.h"

namespace alphazero {
namespace core {

bool GameRegistry::registerGame(const GameId& id, FactoryFn factoryFn) {
    std::unique_lock lock(mutex_);
    return registry_.emplace(id, std::move(factoryFn)).second;
}

bool GameRegistry::unregisterGame(const GameId& id) {
    std::unique_lock lock(mutex_);
    return registry_.erase(id) > 0;
}

std::unique_ptr<IGameState> GameRegistry::createGame(const GameId& id, const VariantArgs& args) {
    std::shared_lock lock(mutex_);
    auto it = registry_.find(id);
    if (it == registry_.end()) {
        return nullptr;
    }
    return (it->second)(args);
}

bool GameRegistry::hasGame(const GameId& id) const {
    std::shared_lock lock(mutex_);
    return registry_.find(id) != registry_.end();
}

std::vector<GameId> GameRegistry::getGameIds() const {
    std::shared_lock lock(mutex_);
    std::vector<GameId> result;
    result.reserve(registry_.size());
    for (const auto& [id, _] : registry_) {
        result.push_back(id);
    }
    return result;
}

} // namespace core
} // namespace alphazero