// include/alphazero/selfplay/self_play_manager.h
#ifndef SELF_PLAY_MANAGER_H
#define SELF_PLAY_MANAGER_H

#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <future>
#include "alphazero/core/igamestate.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/selfplay/game_record.h"

namespace alphazero {
namespace selfplay {

/**
 * @brief Manager for self-play game generation
 */
class SelfPlayManager {
public:
    /**
     * @brief Constructor
     * 
     * @param neuralNetwork Neural network to use for evaluation
     * @param numGames Number of games to generate
     * @param numSimulations Number of MCTS simulations per move
     * @param numThreads Number of threads to use
     */
    SelfPlayManager(nn::NeuralNetwork* neuralNetwork, 
                   int numGames = 100, 
                   int numSimulations = 800, 
                   int numThreads = 4);
    
    /**
     * @brief Destructor
     */
    ~SelfPlayManager();
    
    /**
     * @brief Generate self-play games
     * 
     * @param gameType Type of game to play
     * @param boardSize Board size
     * @param useVariantRules Whether to use variant rules
     * @return Vector of game records
     */
    std::vector<GameRecord> generateGames(
        core::GameType gameType, 
        int boardSize = 0,
        bool useVariantRules = false);
    
    /**
     * @brief Set the exploration parameters
     * 
     * @param dirichletAlpha Dirichlet noise alpha parameter
     * @param dirichletEpsilon Weight of Dirichlet noise
     * @param initialTemperature Initial temperature for move selection
     * @param temperatureDropMove Move number after which to drop temperature
     * @param finalTemperature Final temperature after the drop
     */
    void setExplorationParams(
        float dirichletAlpha = 0.03f,
        float dirichletEpsilon = 0.25f,
        float initialTemperature = 1.0f,
        int temperatureDropMove = 30,
        float finalTemperature = 0.0f);
    
    /**
     * @brief Set the callback for progress updates
     * 
     * @param callback Function to call with progress updates
     */
    void setProgressCallback(std::function<void(int, int, int, int)> callback);
    
    /**
     * @brief Set whether to save games to files
     * 
     * @param saveGames Whether to save games
     * @param outputDir Directory to save games to
     */
    void setSaveGames(bool saveGames, const std::string& outputDir = "games");
    
    /**
     * @brief Set whether to abort self-play
     * 
     * @param abort Whether to abort
     */
    void setAbort(bool abort);
    
    /**
     * @brief Check if self-play is running
     * 
     * @return true if running, false otherwise
     */
    bool isRunning() const;
    
private:
    nn::NeuralNetwork* neuralNetwork_;     // Neural network for evaluation
    int numGames_;                         // Number of games to generate
    int numSimulations_;                   // Number of MCTS simulations per move
    int numThreads_;                       // Number of threads to use
    
    // Exploration parameters
    float dirichletAlpha_;
    float dirichletEpsilon_;
    float initialTemperature_;
    int temperatureDropMove_;
    float finalTemperature_;
    
    // Game saving
    bool saveGames_;
    std::string outputDir_;
    
    // Progress tracking
    std::function<void(int, int, int, int)> progressCallback_;  // (game, move, totalGames, totalMoves)
    std::atomic<bool> abort_;
    std::atomic<bool> running_;
    
    // Thread-safe game record collection
    std::vector<GameRecord> gameRecords_;
    std::mutex gameRecordsMutex_;
    
    /**
     * @brief Play a single game
     * 
     * @param gameId ID of the game (for progress tracking)
     * @param gameType Type of game to play
     * @param boardSize Board size
     * @param useVariantRules Whether to use variant rules
     * @return Game record
     */
    GameRecord playSingleGame(
        int gameId,
        core::GameType gameType, 
        int boardSize,
        bool useVariantRules);
    
    /**
     * @brief Get temperature for the current move
     * 
     * @param moveNum Current move number
     * @return Temperature value
     */
    float getTemperature(int moveNum) const;
};

} // namespace selfplay
} // namespace alphazero

#endif // SELF_PLAY_MANAGER_H