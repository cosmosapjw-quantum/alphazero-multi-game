// variant_args.h
#ifndef VARIANT_ARGS_H
#define VARIANT_ARGS_H

#include <unordered_map>
#include <string>
#include <any>
#include <stdexcept>
#include <typeinfo>

namespace alphazero {
namespace core {

/**
 * @brief Container for variant-typed arguments
 * 
 * This class provides a type-safe way to pass various parameters
 * to game constructors and other functions.
 */
class VariantArgs {
public:
    /**
     * @brief Default constructor
     */
    VariantArgs() = default;
    
    /**
     * @brief Set a value for a key
     * 
     * @tparam T Value type
     * @param key The key
     * @param value The value
     */
    template<typename T>
    void set(const std::string& key, T&& value) {
        args_[key] = std::forward<T>(value);
    }
    
    /**
     * @brief Get a value by key with type conversion
     * 
     * @tparam T Expected value type
     * @param key The key
     * @param defaultValue Default value if key not found
     * @return The value (or default if not found or wrong type)
     */
    template<typename T>
    T get(const std::string& key, const T& defaultValue = T{}) const {
        auto it = args_.find(key);
        if (it == args_.end()) {
            return defaultValue;
        }
        
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return defaultValue;
        }
    }
    
    /**
     * @brief Check if a key exists
     * 
     * @param key The key to check
     * @return true if key exists, false otherwise
     */
    bool has(const std::string& key) const {
        return args_.find(key) != args_.end();
    }
    
    /**
     * @brief Get all keys
     * 
     * @return Vector of all keys
     */
    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        result.reserve(args_.size());
        for (const auto& [key, _] : args_) {
            result.push_back(key);
        }
        return result;
    }
    
    /**
     * @brief Get size
     * 
     * @return Number of key-value pairs
     */
    size_t size() const {
        return args_.size();
    }
    
    /**
     * @brief Check if empty
     * 
     * @return true if no keys, false otherwise
     */
    bool empty() const {
        return args_.empty();
    }
    
private:
    std::unordered_map<std::string, std::any> args_;
};

} // namespace core
} // namespace alphazero

#endif // VARIANT_ARGS_H