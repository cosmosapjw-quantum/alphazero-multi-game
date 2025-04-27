// include/alphazero/elo/elo_tracker.h
#ifndef ELO_TRACKER_H
#define ELO_TRACKER_H

// Include our types header first to prevent pthread conflicts
#include "alphazero/types.h"

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <mutex>
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace elo {

/**
 * @brief Match result
 */
struct MatchResult {
    std::string player1;
    std::string player2;
    int wins1;
    int wins2;
    int draws;
    core::GameType gameType;
    int boardSize;
    bool variantRules;
    std::string date;
    
    // Serialization
    std::string toJson() const;
    static MatchResult fromJson(const std::string& json);
};

/**
 * @brief Player rating
 */
struct PlayerRating {
    std::string name;
    double rating;
    int games;
    int wins;
    int losses;
    int draws;
    
    // Serialization
    std::string toJson() const;
    static PlayerRating fromJson(const std::string& json);
};

/**
 * @brief ELO rating tracker
 */
class EloTracker {
public:
    /**
     * @brief Constructor
     * 
     * @param initial_rating Initial rating for new players
     * @param k_factor K-factor for ELO calculation
     */
    EloTracker(double initial_rating = 1500.0, double k_factor = 32.0);
    
    /**
     * @brief Add a match result
     * 
     * @param result Match result
     */
    void addMatchResult(const MatchResult& result);
    
    /**
     * @brief Get player rating
     * 
     * @param player_name Player name
     * @return Player rating, or nullptr if player not found
     */
    std::shared_ptr<PlayerRating> getPlayerRating(const std::string& player_name) const;
    
    /**
     * @brief Get all player ratings
     * 
     * @return Vector of player ratings
     */
    std::vector<std::shared_ptr<PlayerRating>> getAllRatings() const;
    
    /**
     * @brief Get match history
     * 
     * @return Vector of match results
     */
    std::vector<MatchResult> getMatchHistory() const;
    
    /**
     * @brief Save ratings to file
     * 
     * @param filename Filename to save to
     * @return true if successful, false otherwise
     */
    bool saveRatings(const std::string& filename) const;
    
    /**
     * @brief Load ratings from file
     * 
     * @param filename Filename to load from
     * @return true if successful, false otherwise
     */
    bool loadRatings(const std::string& filename);
    
    /**
     * @brief Save match history to file
     * 
     * @param filename Filename to save to
     * @return true if successful, false otherwise
     */
    bool saveMatchHistory(const std::string& filename) const;
    
    /**
     * @brief Load match history from file
     * 
     * @param filename Filename to load from
     * @return true if successful, false otherwise
     */
    bool loadMatchHistory(const std::string& filename);
    
    /**
     * @brief Calculate expected win probability
     * 
     * @param rating1 Rating of player 1
     * @param rating2 Rating of player 2
     * @return Expected win probability for player 1
     */
    static double calculateExpectedScore(double rating1, double rating2);
    
    /**
     * @brief Calculate new rating
     * 
     * @param oldRating Old rating
     * @param expectedScore Expected score
     * @param actualScore Actual score
     * @param k K-factor
     * @return New rating
     */
    static double calculateNewRating(double oldRating, double expectedScore, 
                                   double actualScore, double k);
    
private:
    double initial_rating_;
    double k_factor_;
    
    std::vector<MatchResult> match_history_;
    std::map<std::string, std::shared_ptr<PlayerRating>> ratings_;
    
    mutable std::mutex mutex_;
    
    /**
     * @brief Update player rating
     * 
     * @param player_name Player name
     * @param opponent_rating Opponent rating
     * @param score Score (1 for win, 0.5 for draw, 0 for loss)
     */
    void updateRating(const std::string& player_name, double opponent_rating, double score);
    
    /**
     * @brief Get or create player rating
     * 
     * @param player_name Player name
     * @return Player rating
     */
    std::shared_ptr<PlayerRating> getOrCreatePlayerRating(const std::string& player_name);
};

} // namespace elo
} // namespace alphazero

#endif // ELO_TRACKER_H