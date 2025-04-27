// src/selfplay/game_record.cpp
#include "alphazero/selfplay/game_record.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace alphazero {
namespace selfplay {

using json = nlohmann::json;

std::string MoveData::toJson() const {
    json j;
    j["action"] = action;
    j["policy"] = policy;
    j["value"] = value;
    j["thinking_time_ms"] = thinking_time_ms;
    return j.dump();
}

MoveData MoveData::fromJson(const std::string& jsonStr) {
    json j = json::parse(jsonStr);
    MoveData data;
    data.action = j["action"];
    data.policy = j["policy"].get<std::vector<float>>();
    data.value = j["value"];
    data.thinking_time_ms = j["thinking_time_ms"];
    return data;
}

GameRecord::GameRecord(core::GameType gameType, int boardSize, bool useVariantRules)
    : gameType_(gameType), boardSize_(boardSize), useVariantRules_(useVariantRules),
      result_(core::GameResult::ONGOING), timestamp_(std::chrono::system_clock::now()) {
}

void GameRecord::addMove(int action, const std::vector<float>& policy, float value, int64_t thinkingTimeMs) {
    MoveData move;
    move.action = action;
    move.policy = policy;
    move.value = value;
    move.thinking_time_ms = thinkingTimeMs;
    moves_.push_back(move);
}

void GameRecord::setResult(core::GameResult result) {
    result_ = result;
}

std::tuple<core::GameType, int, bool> GameRecord::getMetadata() const {
    return {gameType_, boardSize_, useVariantRules_};
}

const std::vector<MoveData>& GameRecord::getMoves() const {
    return moves_;
}

core::GameResult GameRecord::getResult() const {
    return result_;
}

std::string GameRecord::toJson() const {
    json j;
    j["game_type"] = static_cast<int>(gameType_);
    j["board_size"] = boardSize_;
    j["use_variant_rules"] = useVariantRules_;
    j["result"] = static_cast<int>(result_);
    
    // Convert timestamp to ISO string
    auto time_t = std::chrono::system_clock::to_time_t(timestamp_);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%FT%TZ");
    j["timestamp"] = ss.str();
    
    // Convert moves to JSON
    json moves_json = json::array();
    for (const auto& move : moves_) {
        json move_json;
        move_json["action"] = move.action;
        move_json["policy"] = move.policy;
        move_json["value"] = move.value;
        move_json["thinking_time_ms"] = move.thinking_time_ms;
        moves_json.push_back(move_json);
    }
    j["moves"] = moves_json;
    
    return j.dump(4);  // Pretty print with 4-space indent
}

GameRecord GameRecord::fromJson(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);
        
        core::GameType gameType = static_cast<core::GameType>(j["game_type"].get<int>());
        int boardSize = j["board_size"];
        bool useVariantRules = j["use_variant_rules"];
        
        GameRecord record(gameType, boardSize, useVariantRules);
        record.result_ = static_cast<core::GameResult>(j["result"].get<int>());
        
        // Parse moves
        for (const auto& move_json : j["moves"]) {
            MoveData move;
            move.action = move_json["action"];
            move.policy = move_json["policy"].get<std::vector<float>>();
            move.value = move_json["value"];
            move.thinking_time_ms = move_json["thinking_time_ms"];
            record.moves_.push_back(move);
        }
        
        return record;
    } catch (const json::exception& e) {
        throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
    }
}

bool GameRecord::saveToFile(const std::string& filename) const {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << toJson();
        return true;
    } catch (...) {
        return false;
    }
}

GameRecord GameRecord::loadFromFile(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        
        return fromJson(buffer.str());
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load game record: " + std::string(e.what()));
    }
}

} // namespace selfplay
} // namespace alphazero