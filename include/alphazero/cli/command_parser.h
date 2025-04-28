// include/alphazero/cli/command_parser.h
#ifndef COMMAND_PARSER_H
#define COMMAND_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <functional>

namespace alphazero {
namespace cli {

/**
 * @brief Command parser for the CLI
 * 
 * This class provides utilities for parsing command-line input
 * and executing commands.
 */
class CommandParser {
public:
    /**
     * @brief Split a command line into tokens
     * 
     * @param line The input line to tokenize
     * @return Vector of tokens
     */
    static std::vector<std::string> tokenize(const std::string& line);
    
    /**
     * @brief Parse a key-value argument string
     * 
     * @param arg The argument string (e.g., "key=value")
     * @return Optional pair of key and value, or empty if parsing failed
     */
    static std::optional<std::pair<std::string, std::string>> parseKeyValue(const std::string& arg);
    
    /**
     * @brief Extract flags from arguments
     * 
     * @param args Vector of arguments
     * @param flagPrefix Prefix for flags (e.g., "--" or "-")
     * @return Map of flags to values (empty string for boolean flags)
     */
    static std::map<std::string, std::string> extractFlags(
        std::vector<std::string>& args, const std::string& flagPrefix = "--");
    
    /**
     * @brief Check if a flag is present in a flag map
     * 
     * @param flags Map of flags
     * @param flag Flag to check
     * @return true if flag is present, false otherwise
     */
    static bool hasFlag(const std::map<std::string, std::string>& flags, const std::string& flag);
    
    /**
     * @brief Get flag value as string
     * 
     * @param flags Map of flags
     * @param flag Flag to get
     * @param defaultValue Default value if flag is not present
     * @return Flag value or default value
     */
    static std::string getFlagValue(
        const std::map<std::string, std::string>& flags, 
        const std::string& flag, 
        const std::string& defaultValue = "");
    
    /**
     * @brief Get flag value as int
     * 
     * @param flags Map of flags
     * @param flag Flag to get
     * @param defaultValue Default value if flag is not present or conversion fails
     * @return Flag value as int or default value
     */
    static int getFlagValueInt(
        const std::map<std::string, std::string>& flags, 
        const std::string& flag, 
        int defaultValue = 0);
    
    /**
     * @brief Get flag value as float
     * 
     * @param flags Map of flags
     * @param flag Flag to get
     * @param defaultValue Default value if flag is not present or conversion fails
     * @return Flag value as float or default value
     */
    static float getFlagValueFloat(
        const std::map<std::string, std::string>& flags, 
        const std::string& flag, 
        float defaultValue = 0.0f);
    
    /**
     * @brief Get flag value as boolean
     * 
     * @param flags Map of flags
     * @param flag Flag to get
     * @param defaultValue Default value if flag is not present
     * @return Flag value as boolean or default value
     */
    static bool getFlagValueBool(
        const std::map<std::string, std::string>& flags, 
        const std::string& flag, 
        bool defaultValue = false);
};

} // namespace cli
} // namespace alphazero

#endif // COMMAND_PARSER_H