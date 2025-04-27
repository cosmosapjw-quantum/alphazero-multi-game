// src/elo/elo_tracker.cpp
#include "alphazero/types.h"
#include "alphazero/elo/elo_tracker.h"
#include <fstream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace alphazero {
namespace elo {

using json = nlohmann::json;

std::string MatchResult::toJson() const {
    json j;
    j["player1"] = player1;
    j["player2"] = player2;
    j["wins1"] = wins1;
    j["wins2"] = wins2;
    j["draws"] = draws;
    j["gameType"] = static_cast<int>(gameType);
    j["boardSize"] = boardSize;
    j["variantRules"] = variantRules;
    j["date"] = date;
    
    return j.dump();
}

MatchResult MatchResult::fromJson(const std::string& jsonStr) {
    json j = json::parse(jsonStr);
    
    MatchResult result;
    result.player1 = j["player1"];
    result.player2 = j["player2"];
    result.wins1 = j["wins1"];
    result.wins2 = j["wins2"];
    result.draws = j["draws"];
    result.gameType = static_cast<core::GameType>(j["gameType"].get<int>());
    result.boardSize = j["boardSize"];
    result.variantRules = j["variantRules"];
    result.date = j["date"];
    
    return result;
}

std::string PlayerRating::toJson() const {
    json j;
    j["name"] = name;
    j["rating"] = rating;
    j["games"] = games;
    j["wins"] = wins;
    j["losses"] = losses;
    j["draws"] = draws;
    
    return j.dump();
}

PlayerRating PlayerRating::fromJson(const std::string& jsonStr) {
    json j = json::parse(jsonStr);
    
    PlayerRating rating;
    rating.name = j["name"];
    rating.rating = j["rating"];
    rating.games = j["games"];
    rating.wins = j["wins"];
    rating.losses = j["losses"];
    rating.draws = j["draws"];
    
    return rating;
}

EloTracker::EloTracker(double initial_rating, double k_factor)
    : initial_rating_(initial_rating), k_factor_(k_factor) {
}

void EloTracker::addMatchResult(const MatchResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Add to history
    match_history_.push_back(result);
    
    // Update ratings
    std::shared_ptr<PlayerRating> player1 = getOrCreatePlayerRating(result.player1);
    std::shared_ptr<PlayerRating> player2 = getOrCreatePlayerRating(result.player2);
    
    double total_games = result.wins1 + result.wins2 + result.draws;
    
    if (total_games > 0) {
        // Calculate scores
        double score1 = (result.wins1 + 0.5 * result.draws) / total_games;
        double score2 = (result.wins2 + 0.5 * result.draws) / total_games;
        
        // Update ratings
        updateRating(result.player1, player2->rating, score1);
        updateRating(result.player2, player1->rating, score2);
        
        // Update statistics
        player1->games += total_games;
        player1->wins += result.wins1;
        player1->losses += result.wins2;
        player1->draws += result.draws;
        
        player2->games += total_games;
        player2->wins += result.wins2;
        player2->losses += result.wins1;
        player2->draws += result.draws;
    }
}

std::shared_ptr<PlayerRating> EloTracker::getPlayerRating(const std::string& player_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = ratings_.find(player_name);
    if (it != ratings_.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::vector<std::shared_ptr<PlayerRating>> EloTracker::getAllRatings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<PlayerRating>> all_ratings;
    all_ratings.reserve(ratings_.size());
    
    for (const auto& [name, rating] : ratings_) {
        all_ratings.push_back(rating);
    }
    
    return all_ratings;
}

std::vector<MatchResult> EloTracker::getMatchHistory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return match_history_;
}

bool EloTracker::saveRatings(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        json j = json::array();
        
        for (const auto& [name, rating] : ratings_) {
            json rating_json;
            rating_json["name"] = rating->name;
            rating_json["rating"] = rating->rating;
            rating_json["games"] = rating->games;
            rating_json["wins"] = rating->wins;
            rating_json["losses"] = rating->losses;
            rating_json["draws"] = rating->draws;
            
            j.push_back(rating_json);
        }
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << j.dump(4);
        return true;
    } catch (...) {
        return false;
    }
}

bool EloTracker::loadRatings(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        json j;
        file >> j;
        
        ratings_.clear();
        
        for (const auto& rating_json : j) {
            auto rating = std::make_shared<PlayerRating>();
            rating->name = rating_json["name"];
            rating->rating = rating_json["rating"];
            rating->games = rating_json["games"];
            rating->wins = rating_json["wins"];
            rating->losses = rating_json["losses"];
            rating->draws = rating_json["draws"];
            
            ratings_[rating->name] = rating;
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

bool EloTracker::saveMatchHistory(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        json j = json::array();
        
        for (const auto& result : match_history_) {
            json result_json;
            result_json["player1"] = result.player1;
            result_json["player2"] = result.player2;
            result_json["wins1"] = result.wins1;
            result_json["wins2"] = result.wins2;
            result_json["draws"] = result.draws;
            result_json["gameType"] = static_cast<int>(result.gameType);
            result_json["boardSize"] = result.boardSize;
            result_json["variantRules"] = result.variantRules;
            result_json["date"] = result.date;
            
            j.push_back(result_json);
        }
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << j.dump(4);
        return true;
    } catch (...) {
        return false;
    }
}

bool EloTracker::loadMatchHistory(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        json j;
        file >> j;
        
        match_history_.clear();
        
        for (const auto& result_json : j) {
            MatchResult result;
            result.player1 = result_json["player1"];
            result.player2 = result_json["player2"];
            result.wins1 = result_json["wins1"];
            result.wins2 = result_json["wins2"];
            result.draws = result_json["draws"];
            result.gameType = static_cast<core::GameType>(result_json["gameType"].get<int>());
            result.boardSize = result_json["boardSize"];
            result.variantRules = result_json["variantRules"];
            result.date = result_json["date"];
            
            match_history_.push_back(result);
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

double EloTracker::calculateExpectedScore(double rating1, double rating2) {
    return 1.0 / (1.0 + std::pow(10.0, (rating2 - rating1) / 400.0));
}

double EloTracker::calculateNewRating(double oldRating, double expectedScore, 
                                    double actualScore, double k) {
    return oldRating + k * (actualScore - expectedScore);
}

void EloTracker::updateRating(const std::string& player_name, double opponent_rating, double score) {
    std::shared_ptr<PlayerRating> player = getOrCreatePlayerRating(player_name);
    
    double expected_score = calculateExpectedScore(player->rating, opponent_rating);
    player->rating = calculateNewRating(player->rating, expected_score, score, k_factor_);
}

std::shared_ptr<PlayerRating> EloTracker::getOrCreatePlayerRating(const std::string& player_name) {
    auto it = ratings_.find(player_name);
    if (it != ratings_.end()) {
        return it->second;
    }
    
    // Create new player
    auto player = std::make_shared<PlayerRating>();
    player->name = player_name;
    player->rating = initial_rating_;
    player->games = 0;
    player->wins = 0;
    player->losses = 0;
    player->draws = 0;
    
    ratings_[player_name] = player;
    return player;
}

} // namespace elo
} // namespace alphazero