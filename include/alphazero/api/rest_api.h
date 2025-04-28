// include/alphazero/api/rest_api.h
#ifndef REST_API_H
#define REST_API_H

#include <string>
#include <functional>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include "alphazero/core/igamestate.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"

namespace alphazero {
namespace api {

/**
 * @brief REST API handler for AlphaZero
 * 
 * This class handles REST API requests for the AlphaZero engine.
 */
class RestApi {
public:
    /**
     * @brief Constructor
     */
    RestApi();
    
    /**
     * @brief Handle a REST API request
     * 
     * @param method HTTP method (GET, POST, etc.)
     * @param path Request path
     * @param body Request body (JSON)
     * @return Response (JSON string)
     */
    std::string handleRequest(const std::string& method, 
                             const std::string& path, 
                             const std::string& body);
    
    /**
     * @brief Register routes
     */
    void registerRoutes();
    
private:
    // Game sessions (state/MCTS for each client)
    struct GameSession {
        std::unique_ptr<core::IGameState> state;
        std::unique_ptr<mcts::ParallelMCTS> mcts;
        std::string id;
        std::string gameType;
        std::string createdAt;
        std::string lastActivity;
    };
    
    // Route handlers
    using RouteHandler = std::function<nlohmann::json(const nlohmann::json&)>;
    std::map<std::string, std::map<std::string, RouteHandler>> routes_;
    
    // Game sessions
    std::map<std::string, std::unique_ptr<GameSession>> sessions_;
    std::shared_ptr<nn::NeuralNetwork> networkModel_;
    std::unique_ptr<mcts::TranspositionTable> tt_;
    
    // Helper method to generate a session ID
    std::string generateSessionId() const;
    
    // API route handlers
    nlohmann::json handleGameCreate(const nlohmann::json& request);
    nlohmann::json handleGameStatus(const nlohmann::json& request);
    nlohmann::json handleGameMove(const nlohmann::json& request);
    nlohmann::json handleGameAiMove(const nlohmann::json& request);
    nlohmann::json handleModelInfo(const nlohmann::json& request);
    
    // Clean up old sessions
    void cleanupSessions();
};

} // namespace api
} // namespace alphazero

#endif // REST_API_H