// plugin_loader.cpp
#include "alphazero/core/plugin_loader.h"
#include "alphazero/core/plugin_api.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

// Platform-specific dynamic library handling
#if defined(_WIN32)
    #include <windows.h>
    using LibHandle = HMODULE;
    #define LOAD_LIBRARY(path) LoadLibraryA((path).string().c_str())
    #define GET_SYMBOL(handle, name) GetProcAddress(handle, name)
    #define CLOSE_LIBRARY(handle) FreeLibrary(handle)
#else
    #include <dlfcn.h>
    using LibHandle = void*;
    #define LOAD_LIBRARY(path) dlopen((path).string().c_str(), RTLD_NOW)
    #define GET_SYMBOL(handle, name) dlsym(handle, name)
    #define CLOSE_LIBRARY(handle) dlclose(handle)
#endif

namespace alphazero {
namespace core {

// Private implementation
struct PluginLoader::PluginHandle {
    std::filesystem::path path;
    LibHandle handle;
    std::vector<GameId> registeredGames;
    
    // Function pointers for plugin API
    int (*getApiVersion)();
    void (*registerPlugin)(GameRegistry&);
    void (*unregisterPlugin)(GameRegistry&);
};

class PluginLoader::Impl {
public:
    std::vector<PluginHandle> plugins;
};

PluginLoader::PluginLoader() : pImpl(std::make_unique<Impl>()) {}

PluginLoader::~PluginLoader() {
    unloadAll();
}

bool PluginLoader::loadPlugin(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "Plugin file does not exist: " << path << std::endl;
        return false;
    }
    
    // Load the library
    LibHandle handle = LOAD_LIBRARY(path);
    if (!handle) {
        std::cerr << "Failed to load plugin: " << path << std::endl;
        return false;
    }
    
    // Get API version
    auto getApiVersionFn = reinterpret_cast<int(*)()>(
        GET_SYMBOL(handle, "getPluginApiVersion"));
    
    if (!getApiVersionFn) {
        std::cerr << "Plugin does not export getPluginApiVersion: " << path << std::endl;
        CLOSE_LIBRARY(handle);
        return false;
    }
    
    int apiVersion = getApiVersionFn();
    if (apiVersion != PLUGIN_API_VERSION) {
        std::cerr << "Plugin API version mismatch. Expected " << PLUGIN_API_VERSION 
                  << ", got " << apiVersion << ": " << path << std::endl;
        CLOSE_LIBRARY(handle);
        return false;
    }
    
    // Get register function
    auto registerPluginFn = reinterpret_cast<void(*)(GameRegistry&)>(
        GET_SYMBOL(handle, "registerPlugin"));
    
    if (!registerPluginFn) {
        std::cerr << "Plugin does not export registerPlugin: " << path << std::endl;
        CLOSE_LIBRARY(handle);
        return false;
    }
    
    // Get unregister function (optional)
    auto unregisterPluginFn = reinterpret_cast<void(*)(GameRegistry&)>(
        GET_SYMBOL(handle, "unregisterPlugin"));
    
    // Capture games before registration
    auto beforeGames = GameRegistry::instance().getGameIds();
    
    // Register plugin games
    registerPluginFn(GameRegistry::instance());
    
    // Detect newly registered games
    auto afterGames = GameRegistry::instance().getGameIds();
    std::vector<GameId> newGames;
    
    for (const auto& game : afterGames) {
        if (std::find(beforeGames.begin(), beforeGames.end(), game) == beforeGames.end()) {
            newGames.push_back(game);
        }
    }
    
    // Store plugin info
    PluginHandle pluginInfo;
    pluginInfo.path = path;
    pluginInfo.handle = handle;
    pluginInfo.registeredGames = std::move(newGames);
    pluginInfo.getApiVersion = getApiVersionFn;
    pluginInfo.registerPlugin = registerPluginFn;
    pluginInfo.unregisterPlugin = unregisterPluginFn;
    
    pImpl->plugins.push_back(std::move(pluginInfo));
    
    return true;
}

int PluginLoader::loadDirectory(const std::filesystem::path& directory) {
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        return 0;
    }
    
    int loadedCount = 0;
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        
        auto ext = entry.path().extension().string();
        
        #if defined(_WIN32)
        if (ext == ".dll")
        #else
        if (ext == ".so" || ext == ".dylib")
        #endif
        {
            if (loadPlugin(entry.path())) {
                loadedCount++;
            }
        }
    }
    
    return loadedCount;
}

void PluginLoader::unloadAll() {
    for (auto& plugin : pImpl->plugins) {
        // Try to unregister games
        if (plugin.unregisterPlugin) {
            plugin.unregisterPlugin(GameRegistry::instance());
        } else {
            // Manually unregister games if function not available
            for (const auto& gameId : plugin.registeredGames) {
                GameRegistry::instance().unregisterGame(gameId);
            }
        }
        
        // Close library
        CLOSE_LIBRARY(plugin.handle);
    }
    
    pImpl->plugins.clear();
}

std::vector<std::filesystem::path> PluginLoader::getLoadedPlugins() const {
    std::vector<std::filesystem::path> result;
    result.reserve(pImpl->plugins.size());
    
    for (const auto& plugin : pImpl->plugins) {
        result.push_back(plugin.path);
    }
    
    return result;
}

size_t PluginLoader::getPluginCount() const {
    return pImpl->plugins.size();
}

} // namespace core
} // namespace alphazero