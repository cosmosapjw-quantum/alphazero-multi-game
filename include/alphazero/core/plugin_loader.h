// plugin_loader.h
#ifndef PLUGIN_LOADER_H
#define PLUGIN_LOADER_H

#include "registry.h"
#include <filesystem>
#include <vector>
#include <string>

namespace alphazero {
namespace core {

/**
 * @brief Loader for game plugins
 * 
 * This class handles loading and unloading game plugins
 * from shared libraries.
 */
class PluginLoader {
public:
    /**
     * @brief Constructor
     */
    PluginLoader();
    
    /**
     * @brief Destructor
     */
    ~PluginLoader();
    
    /**
     * @brief Load a plugin from a shared library
     * 
     * @param path Path to the shared library
     * @return true if loaded successfully, false otherwise
     */
    bool loadPlugin(const std::filesystem::path& path);
    
    /**
     * @brief Load all plugins from a directory
     * 
     * @param directory Path to the directory
     * @return Number of successfully loaded plugins
     */
    int loadDirectory(const std::filesystem::path& directory);
    
    /**
     * @brief Unload all plugins
     */
    void unloadAll();
    
    /**
     * @brief Get paths of loaded plugins
     * 
     * @return Vector of plugin paths
     */
    std::vector<std::filesystem::path> getLoadedPlugins() const;
    
    /**
     * @brief Get number of loaded plugins
     * 
     * @return Plugin count
     */
    size_t getPluginCount() const;
    
private:
    struct PluginHandle;
    
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace alphazero

#endif // PLUGIN_LOADER_H