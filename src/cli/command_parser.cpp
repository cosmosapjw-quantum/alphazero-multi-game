// src/cli/command_parser.cpp
#include "alphazero/cli/command_parser.h"
#include <sstream>
#include <algorithm>

namespace alphazero {
namespace cli {

std::vector<std::string> CommandParser::tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    
    bool inQuotes = false;
    std::string quotedToken;
    
    while (iss >> token) {
        if (!inQuotes) {
            // Check if token starts with a quote
            if (token.size() > 0 && token[0] == '"') {
                if (token.size() > 1 && token.back() == '"') {
                    // Token is "word" - remove quotes and add
                    tokens.push_back(token.substr(1, token.size() - 2));
                } else {
                    // Token starts a quoted string
                    inQuotes = true;
                    quotedToken = token.substr(1);
                }
            } else {
                tokens.push_back(token);
            }
        } else {
            // We're inside a quoted string
            if (token.back() == '"') {
                // Token ends the quoted string
                quotedToken += " " + token.substr(0, token.size() - 1);
                tokens.push_back(quotedToken);
                inQuotes = false;
                quotedToken.clear();
            } else {
                // Token is part of the quoted string
                quotedToken += " " + token;
            }
        }
    }
    
    // Handle unclosed quotes
    if (inQuotes) {
        tokens.push_back(quotedToken);
    }
    
    return tokens;
}

std::optional<std::pair<std::string, std::string>> CommandParser::parseKeyValue(const std::string& arg) {
    size_t equalPos = arg.find('=');
    if (equalPos == std::string::npos) {
        return std::nullopt;
    }
    
    std::string key = arg.substr(0, equalPos);
    std::string value = arg.substr(equalPos + 1);
    
    // Trim whitespace
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);
    
    return std::make_pair(key, value);
}

std::map<std::string, std::string> CommandParser::extractFlags(
    std::vector<std::string>& args, const std::string& flagPrefix) {
    
    std::map<std::string, std::string> flags;
    auto it = args.begin();
    
    while (it != args.end()) {
        std::string arg = *it;
        
        // Check if argument is a flag
        if (arg.size() > flagPrefix.size() && 
            arg.substr(0, flagPrefix.size()) == flagPrefix) {
            
            // Extract flag name
            std::string flag = arg.substr(flagPrefix.size());
            
            // Get flag value if present
            std::string value;
            bool hasValue = false;
            
            // Check if flag contains an equal sign for value
            size_t equalPos = flag.find('=');
            if (equalPos != std::string::npos) {
                value = flag.substr(equalPos + 1);
                flag = flag.substr(0, equalPos);
                hasValue = true;
            } else if (it + 1 != args.end() && (it + 1)->size() > 0 && (*(it + 1))[0] != '-') {
                // Next argument is not a flag, use it as value
                value = *(it + 1);
                hasValue = true;
                ++it;  // Skip value in next iteration
            }
            
            // Store flag and value
            flags[flag] = value;
            
            // Remove flag from arguments
            it = args.erase(it);
            
            if (hasValue && equalPos == std::string::npos) {
                it = args.erase(it);
            }
        } else {
            ++it;
        }
    }
    
    return flags;
}

bool CommandParser::hasFlag(const std::map<std::string, std::string>& flags, const std::string& flag) {
    return flags.find(flag) != flags.end();
}

std::string CommandParser::getFlagValue(
    const std::map<std::string, std::string>& flags, 
    const std::string& flag, 
    const std::string& defaultValue) {
    
    auto it = flags.find(flag);
    if (it != flags.end()) {
        return it->second;
    }
    
    return defaultValue;
}

int CommandParser::getFlagValueInt(
    const std::map<std::string, std::string>& flags, 
    const std::string& flag, 
    int defaultValue) {
    
    auto it = flags.find(flag);
    if (it != flags.end()) {
        try {
            return std::stoi(it->second);
        } catch (const std::exception& e) {
            return defaultValue;
        }
    }
    
    return defaultValue;
}

float CommandParser::getFlagValueFloat(
    const std::map<std::string, std::string>& flags, 
    const std::string& flag, 
    float defaultValue) {
    
    auto it = flags.find(flag);
    if (it != flags.end()) {
        try {
            return std::stof(it->second);
        } catch (const std::exception& e) {
            return defaultValue;
        }
    }
    
    return defaultValue;
}

bool CommandParser::getFlagValueBool(
    const std::map<std::string, std::string>& flags, 
    const std::string& flag, 
    bool defaultValue) {
    
    auto it = flags.find(flag);
    if (it != flags.end()) {
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        
        if (value.empty() || value == "true" || value == "1" || value == "yes" || value == "y") {
            return true;
        } else if (value == "false" || value == "0" || value == "no" || value == "n") {
            return false;
        }
    }
    
    return defaultValue;
}

} // namespace cli
} // namespace alphazero