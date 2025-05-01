// plugin_api.h
#ifndef PLUGIN_API_H
#define PLUGIN_API_H

#include "registry.h"

// Platform-specific export macros
#if defined(_WIN32)
    #define PLUGIN_API __declspec(dllexport)
#else
    #define PLUGIN_API
#endif

// Plugin API version
#define PLUGIN_API_VERSION 1

extern "C" {
    /**
     * @brief Get plugin API version
     * 
     * @return API version number
     */
    PLUGIN_API int getPluginApiVersion();
    
    /**
     * @brief Register plugin with the registry
     * 
     * @param registry Game registry to register with
     */
    PLUGIN_API void registerPlugin(alphazero::core::GameRegistry& registry);
    
    /**
     * @brief Unregister plugin from the registry
     * 
     * @param registry Game registry to unregister from
     */
    PLUGIN_API void unregisterPlugin(alphazero::core::GameRegistry& registry);
}

#endif // PLUGIN_API_H