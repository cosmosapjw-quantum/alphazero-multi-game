// src/api/rest_api.cpp
#include "alphazero/api/rest_api.h"
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <regex>

namespace alphazero {
namespace api {

RestApi::RestApi() 
    : networkModel_(nn::NeuralNetwork::create("", core::GameType::GOMOKU)),
      tt_(std::make_unique<mcts::TranspositionTable>(1048576, 1024)) {
    
    registerRoutes();
    
    // Start cleanup thread
    std::thread cleanupThread([this]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::minutes(10));
            cleanupSessions();
        }
    });
    cleanupThread.detach();
}

void RestApi::registerRoutes() {
    // Game management
    routes_["POST"]["/api/games"] = [this](const nlohmann::json& request) {
        return handleGameCreate(request);
    };
    
    routes_["GET"]["/api/games/{id}"] = [this](const nlohmann::json& request) {
        return handleGameStatus(request);
    };
    
    routes_["POST"]["/api/games/{id}/move"] = [this](const nlohmann::json& request) {
        return handleGameMove(request);
    };
    
    routes_["POST"]["/api/games/{id}/ai_move"] = [this](const nlohmann::json& request) {
        return handleGameAiMove(request);
    };
    
    // Model info
    routes_["GET"]["/api/model"] = [this](const nlohmann::json& request) {
        return handleModelInfo(request);
    };
}

std::string RestApi::handleRequest(const std::string& method, 
                                  const std::string& path, 
                                  const std::string& body) {
    try {
        // Parse body if not empty
        nlohmann::json requestJson;
        if (!body.empty()) {
            requestJson = nlohmann::json::parse(body);
        }
        
        // Extract path parameters
        std::string routePath = path;
        std::smatch matches;
        std::regex idRegex("/api/games/([^/]+)(?:/.*)?");
        
        if (std::regex_search(path, matches, idRegex) && matches.size() > 1) {
            // Extract ID from path
            std::string id = matches[1];
            requestJson["id"] = id;
            
            // Normalize path by replacing ID with {id}
            routePath = std::regex_replace(path, std::regex(id), "{id}");
        }
        
        // Find handler for method and path
        if (routes_.count(method) > 0 && routes_[method].count(routePath) > 0) {
            // Call handler
            nlohmann::json response = routes_[method][routePath](requestJson);
            return response.dump();
        } else {
            // Route not found
            nlohmann::json errorResponse = {
                {"error", "Not found"},
                {"message", "The requested route was not found"}
            };
            return errorResponse.dump();
        }
    } catch (const std::exception& e) {
        // Handle error
        nlohmann::json errorResponse = {
            {"error", "Internal server error"},
            {"message", e.what()}
        };
        return errorResponse.dump();
    }
}

nlohmann::json RestApi::handleGameCreate(const nlohmann::json& request) {
    // Get game type from request
    std::string gameTypeStr = request.value("game", "gomoku");
    int boardSize = request.value("board_size", 0);
    bool variantRules = request.value("variant_rules", false);
    
    // Convert game type string to enum
    core::GameType gameType;
    if (gameTypeStr == "gomoku") {
        gameType = core::GameType::GOMOKU;
    } else if (gameTypeStr == "chess") {
        gameType = core::GameType::CHESS;
    } else if (gameTypeStr == "go") {
        gameType = core::GameType::GO;
    } else {
        throw std::invalid_argument("Invalid game type: " + gameTypeStr);
    }
    
    // Create game session
    auto session = std::make_unique<GameSession>();
    session->id = generateSessionId();
    session->gameType = gameTypeStr;
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%dT%H:%M:%SZ");
    session->createdAt = ss.str();
    session->lastActivity = ss.str();
    
    // Create game state
    session->state = core::createGameState(gameType, boardSize, variantRules);
    
    // Create MCTS
    int numThreads = 4;
    int numSimulations = 800;
    session->mcts = std::make_unique<mcts::ParallelMCTS>(
        *(session->state), 
        networkModel_.get(), 
        tt_.get(), 
        numThreads, 
        numSimulations
    );
    
    // Store session
    std::string sessionId = session->id;
    sessions_[sessionId] = std::move(session);
    
    // Return session info
    nlohmann::json response = {
        {"id", sessionId},
        {"game", gameTypeStr},
        {"board_size", boardSize > 0 ? boardSize : session->state->getBoardSize()},
        {"variant_rules", variantRules},
        {"created_at", sessions_[sessionId]->createdAt}
    };
    
    return response;
}

nlohmann::json RestApi::handleGameStatus(const nlohmann::json& request) {
    // Get session ID
    std::string id = request.value("id", "");
    if (id.empty() || sessions_.find(id) == sessions_.end()) {
        throw std::invalid_argument("Invalid game ID");
    }
    
    auto& session = sessions_[id];
    
    // Update last activity time
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%dT%H:%M:%SZ");
    session->lastActivity = ss.str();
    
    // Get game state
    auto& state = session->state;
    
    // Get board representation
    std::string boardStr = state->toString();
    
    // Build response
    nlohmann::json response = {
        {"id", id},
        {"game", session->gameType},
        {"board_size", state->getBoardSize()},
        {"board", boardStr},
        {"current_player", state->getCurrentPlayer()},
        {"is_terminal", state->isTerminal()},
        {"game_result", static_cast<int>(state->getGameResult())},
        {"move_history", state->getMoveHistory()},
        {"last_activity", session->lastActivity}
    };
    
    // If terminal, include result info
    if (state->isTerminal()) {
        core::GameResult result = state->getGameResult();
        std::string resultStr;
        
        switch (result) {
            case core::GameResult::WIN_PLAYER1:
                resultStr = "Player 1 wins";
                break;
            case core::GameResult::WIN_PLAYER2:
                resultStr = "Player 2 wins";
                break;
            case core::GameResult::DRAW:
                resultStr = "Draw";
                break;
            default:
                resultStr = "Unknown";
                break;
        }
        
        response["result_str"] = resultStr;
    }
    
    return response;
}

nlohmann::json RestApi::handleGameMove(const nlohmann::json& request) {
    // Get session ID
    std::string id = request.value("id", "");
    if (id.empty() || sessions_.find(id) == sessions_.end()) {
        throw std::invalid_argument("Invalid game ID");
    }
    
    auto& session = sessions_[id];
    
    // Get move from request
    if (!request.contains("move")) {
        throw std::invalid_argument("Missing 'move' parameter");
    }
    
    // Parse move (either as string or integer)
    int action = -1;
    if (request["move"].is_string()) {
        std::string moveStr = request["move"];
        auto actionOpt = session->state->stringToAction(moveStr);
        if (!actionOpt) {
            throw std::invalid_argument("Invalid move string: " + moveStr);
        }
        action = *actionOpt;
    } else if (request["move"].is_number()) {
        action = request["move"];
    } else {
        throw std::invalid_argument("Move must be a string or integer");
    }
    
    // Check if move is legal
    if (!session->state->isLegalMove(action)) {
        throw std::invalid_argument("Illegal move: " + std::to_string(action));
    }
    
    // Make move
    session->state->makeMove(action);
    
    // Update MCTS
    session->mcts->updateWithMove(action);
    
    // Update last activity time
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%dT%H:%M:%SZ");
    session->lastActivity = ss.str();
    
    // Return updated game state
    return handleGameStatus(request);
}

nlohmann::json RestApi::handleGameAiMove(const nlohmann::json& request) {
    // Get session ID
    std::string id = request.value("id", "");
    if (id.empty() || sessions_.find(id) == sessions_.end()) {
        throw std::invalid_argument("Invalid game ID");
    }
    
    auto& session = sessions_[id];
    
    // Check if game is already over
    if (session->state->isTerminal()) {
        throw std::invalid_argument("Game is already over");
    }
    
    // Get parameters
    int simulations = request.value("simulations", 800);
    float temperature = request.value("temperature", 0.0f);
    bool trainingMode = request.value("training_mode", false);
    
    // Update MCTS parameters if needed
    session->mcts->setNumSimulations(simulations);
    
    // Search for best move
    session->mcts->search();
    
    // Select move
    int action = session->mcts->selectAction(trainingMode, temperature);
    
    // Get policy distribution
    std::vector<float> policy = session->mcts->getActionProbabilities(temperature);
    
    // Make move
    session->state->makeMove(action);
    
    // Update MCTS
    session->mcts->updateWithMove(action);
    
    // Update last activity time
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%dT%H:%M:%SZ");
    session->lastActivity = ss.str();
    
    // Build response
    nlohmann::json response = handleGameStatus(request);
    response["ai_move"] = action;
    response["ai_move_str"] = session->state->actionToString(action);
    response["policy"] = policy;
    response["simulations"] = simulations;
    response["temperature"] = temperature;
    
    return response;
}

nlohmann::json RestApi::handleModelInfo(const nlohmann::json& request) {
    // Return model information
    nlohmann::json response = {
        {"device", networkModel_->getDeviceInfo()},
        {"model_info", networkModel_->getModelInfo()},
        {"model_size_bytes", networkModel_->getModelSizeBytes()},
        {"inference_time_ms", networkModel_->getInferenceTimeMs()},
        {"batch_size", networkModel_->getBatchSize()},
        {"gpu_available", networkModel_->isGpuAvailable()}
    };
    
    return response;
}

std::string RestApi::generateSessionId() const {
    static const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dist(0, sizeof(charset) - 2);
    
    std::string id;
    id.reserve(32);
    
    for (int i = 0; i < 32; ++i) {
        id += charset[dist(gen)];
    }
    
    return id;
}

void RestApi::cleanupSessions() {
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    
    // Sessions older than 1 hour will be removed
    const int SESSION_TIMEOUT_HOURS = 1;
    
    // Iterate through sessions
    auto it = sessions_.begin();
    while (it != sessions_.end()) {
        const auto& session = it->second;
        
        // Parse last activity time
        std::tm tm = {};
        std::istringstream ss(session->lastActivity);
        ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        auto lastActivityTime = std::mktime(&tm);
        
        // Check if session has expired
        auto timeDiff = std::difftime(nowTime, lastActivityTime);
        if (timeDiff > SESSION_TIMEOUT_HOURS * 3600) {
            // Remove session
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace api
} // namespace alphazero