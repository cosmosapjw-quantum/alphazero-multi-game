// game_registration.h
#ifndef GAME_REGISTRATION_H
#define GAME_REGISTRATION_H

#include "registry.h"

namespace alphazero {
namespace core {

/**
 * @brief Macro for self-registering games
 * 
 * This macro creates a static initializer that registers
 * a game with the registry when the library is loaded.
 */
#define REGISTER_GAME(id, factoryLambda) \
    namespace { \
        struct GameRegistrar_##id { \
            GameRegistrar_##id() { \
                alphazero::core::GameRegistry::instance().registerGame( \
                    #id, factoryLambda); \
            } \
            ~GameRegistrar_##id() { \
                alphazero::core::GameRegistry::instance().unregisterGame(#id); \
            } \
        }; \
        static GameRegistrar_##id registrar_##id; \
    }

} // namespace core
} // namespace alphazero

#endif // GAME_REGISTRATION_H