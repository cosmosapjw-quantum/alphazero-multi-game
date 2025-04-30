// parallel_mcts.h
#ifndef PARALLEL_MCTS_H
#define PARALLEL_MCTS_H

// Include our types header first to prevent pthread conflicts
#include "alphazero/types.h"

#include <vector>
#include <deque>
#include <atomic>
#include <mutex>
#include <random>
#include <future>
#include <thread>
#include <memory>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include "alphazero/core/igamestate.h"
#include "alphazero/mcts/mcts_node.h"
#include "alphazero/mcts/transposition_table.h"
#include "alphazero/nn/batch_queue.h"

namespace alphazero {

// Forward declarations
namespace nn {
    class NeuralNetwork;
    class BatchQueue;
}

namespace mcts {

/**
 * @brief Node selection strategies for MCTS
 */
enum class MCTSNodeSelection {
    UCB,             // Standard UCB formula
    PUCT,            // AlphaZero's PUCT formula
    PROGRESSIVE_BIAS, // Progressive bias with visit count
    RAVE            // Rapid Action Value Estimation
};

/**
 * @brief Thread pool for parallel execution
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();
    
    // Execute task asynchronously
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Get the number of threads
    size_t size() const { return workers.size(); }
    
private:
    // Worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::deque<std::function<void()>> tasks;
    
    // Synchronization
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

/**
 * @brief Statistics for MCTS search
 */
struct MCTSStats {
    std::atomic<size_t> nodesCreated{0};
    std::atomic<size_t> nodesExpanded{0};
    std::atomic<size_t> nodesTotalVisits{0};
    std::atomic<size_t> cacheHits{0};
    std::atomic<size_t> cacheMisses{0};
    std::atomic<size_t> evaluationCalls{0};
    std::atomic<size_t> simulationCount{0};
    std::atomic<size_t> batchCount{0};
    std::atomic<size_t> batchSize{0};
    
    void reset() {
        nodesCreated = 0;
        nodesExpanded = 0;
        nodesTotalVisits = 0;
        cacheHits = 0;
        cacheMisses = 0;
        evaluationCalls = 0;
        simulationCount = 0;
        batchCount = 0;
        batchSize = 0;
    }
};

/**
 * @brief Configuration for MCTS search
 */
struct MCTSConfig {
    int numThreads = 4;
    int numSimulations = 800;
    float cPuct = 1.5f;
    float fpuReduction = 0.2f;
    int virtualLoss = 3;
    MCTSNodeSelection selectionStrategy = MCTSNodeSelection::PUCT;
    bool useProgressiveWidening = true;
    int progressiveWideningBase = 4;
    float progressiveWideningExponent = 0.5f;
    int minVisitsForWidening = 2;
    bool useDirichletNoise = false;
    float dirichletAlpha = 0.03f;
    float dirichletEpsilon = 0.25f;
    float firstPlayUrgency = 1.0f;
    bool useBatchInference = true;
    int batchSize = 8;
    int transpositionTableSize = 1048576;
    int cacheEntryMaxAge = 50000; // milliseconds
    bool useValueBounds = true;
    bool useTemporalDifference = false;
    float tdLambda = 0.8f;
    int maxSearchDepth = 1000;
    int visitThresholdForPruning = 1;
    bool useFmapCache = true;
};

/**
 * @brief Parallel Monte Carlo Tree Search implementation
 * 
 * This class implements the MCTS algorithm with parallel search
 * and integration with neural network evaluation.
 */
class ParallelMCTS {
public:
    /**
     * @brief Constructor
     * 
     * @param rootState The initial game state
     * @param nn Neural network for evaluation (can be nullptr)
     * @param tt Transposition table (can be nullptr)
     * @param numThreads Number of search threads
     * @param numSimulations Number of simulations to run
     * @param cPuct Exploration constant
     * @param fpuReduction First play urgency reduction
     * @param virtualLoss Virtual loss amount
     */
    ParallelMCTS(
        const core::IGameState& rootState,
        alphazero::nn::NeuralNetwork* nn = nullptr,
        TranspositionTable* tt = nullptr,
        int numThreads = 1,
        int numSimulations = 800,
        float cPuct = 1.5f,
        float fpuReduction = 0.0f,
        int virtualLoss = 3
    );
    
    /**
     * @brief Constructor with configuration
     * 
     * @param rootState The initial game state
     * @param config MCTS configuration
     * @param nn Neural network for evaluation (can be nullptr)
     * @param tt Transposition table (can be nullptr)
     */
    ParallelMCTS(
        const core::IGameState& rootState,
        const MCTSConfig& config,
        alphazero::nn::NeuralNetwork* nn = nullptr,
        TranspositionTable* tt = nullptr
    );
    
    /**
     * @brief Destructor
     */
    ~ParallelMCTS();
    
    // Non-copyable but movable
    ParallelMCTS(const ParallelMCTS&) = delete;
    ParallelMCTS& operator=(const ParallelMCTS&) = delete;
    ParallelMCTS(ParallelMCTS&&) noexcept = default;
    ParallelMCTS& operator=(ParallelMCTS&&) noexcept = default;
    
    /**
     * @brief Run MCTS search
     */
    void search();
    
    /**
     * @brief Select best action based on visit counts
     * 
     * @param isTraining Whether in training mode (affects temperature)
     * @param temperature Temperature parameter for exploration
     * @return Selected action
     */
    int selectAction(bool isTraining = false, float temperature = 1.0f);
    
    /**
     * @brief Get action probabilities based on visit counts
     * 
     * @param temperature Temperature parameter for exploration
     * @return Vector of action probabilities
     */
    std::vector<float> getActionProbabilities(float temperature = 1.0f) const;
    
    /**
     * @brief Get value estimate of the root node
     * 
     * @return Value estimate [-1,1]
     */
    float getRootValue() const;
    
    /**
     * @brief Update tree with a move, keeping relevant subtree
     * 
     * @param action The action to make
     */
    void updateWithMove(int action);
    
    /**
     * @brief Add Dirichlet noise to root for exploration
     * 
     * @param alpha Concentration parameter
     * @param epsilon Weight of noise
     */
    void addDirichletNoise(float alpha = 0.03f, float epsilon = 0.25f);
    
    /**
     * @brief Set number of search threads
     * 
     * @param numThreads Number of threads
     */
    void setNumThreads(int numThreads);
    
    /**
     * @brief Set number of simulations
     * 
     * @param numSimulations Number of simulations
     */
    void setNumSimulations(int numSimulations);
    
    /**
     * @brief Set exploration constant
     * 
     * @param cPuct Exploration constant
     */
    void setCPuct(float cPuct) { config_.cPuct = cPuct; }
    
    /**
     * @brief Set first play urgency reduction
     * 
     * @param fpuReduction FPU reduction value
     */
    void setFpuReduction(float fpuReduction) { config_.fpuReduction = fpuReduction; }
    
    /**
     * @brief Set virtual loss amount
     * 
     * @param virtualLoss Virtual loss amount
     */
    void setVirtualLoss(int virtualLoss) { config_.virtualLoss = virtualLoss; }
    
    /**
     * @brief Set neural network for evaluation
     * 
     * @param nn Neural network pointer
     */
    void setNeuralNetwork(alphazero::nn::NeuralNetwork* nn);
    
    /**
     * @brief Set transposition table
     * 
     * @param tt Transposition table pointer
     */
    void setTranspositionTable(TranspositionTable* tt);
    
    /**
     * @brief Set node selection strategy
     * 
     * @param strategy Selection strategy enum
     */
    void setSelectionStrategy(MCTSNodeSelection strategy) { config_.selectionStrategy = strategy; }
    
    /**
     * @brief Set MCTS configuration
     * 
     * @param config New configuration
     */
    void setConfig(const MCTSConfig& config);
    
    /**
     * @brief Get current MCTS configuration
     * 
     * @return Current configuration
     */
    const MCTSConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set deterministic mode for reproducible results
     * 
     * @param enable Whether to enable deterministic mode
     */
    void setDeterministicMode(bool enable);
    
    /**
     * @brief Enable debug mode for extra logging
     * 
     * @param debug Whether to enable debug mode
     */
    void setDebugMode(bool debug) { debugMode_ = debug; }
    
    /**
     * @brief Print search statistics to stdout
     */
    void printSearchStats() const;
    
    /**
     * @brief Print search path for an action
     * 
     * @param action The action to show path for
     */
    void printSearchPath(int action) const;
    
    /**
     * @brief Get search information as string
     * 
     * @return Search information string
     */
    std::string getSearchInfo() const;
    
    /**
     * @brief Set progress callback function
     * 
     * @param callback Function to call with progress updates
     */
    void setProgressCallback(std::function<void(int, int)> callback) {
        progressCallback_ = std::move(callback);
    }
    
    /**
     * @brief Get estimated memory usage
     * 
     * @return Memory usage in bytes
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Release memory by pruning unused nodes
     * 
     * @param visitThreshold Minimum visit count to keep a node
     * @return Number of nodes pruned
     */
    size_t releaseMemory(int visitThreshold = 1);
    
    /**
     * @brief Get the search statistics
     * 
     * @return Search statistics
     */
    const MCTSStats& getStats() const { return stats_; }
    
    /**
     * @brief Reset search statistics
     */
    void resetStats() { stats_.reset(); }
    
    /**
     * @brief Get the root node
     * 
     * @return Pointer to root node
     */
    MCTSNode* getRootNode() const { return rootNode_.get(); }
    
    /**
     * @brief Get the current root state
     * 
     * @return Reference to current root state
     */
    const core::IGameState& getRootState() const { return *rootState_; }
    
    /**
     * @brief Analyze the position and return top N moves with info
     * 
     * @param topN Number of top moves to return
     * @return Vector of (action, visits, value, prior) tuples
     */
    std::vector<std::tuple<int, int, float, float>> analyzePosition(int topN = 5) const;
    
    /**
     * @brief Enable or disable progressive widening
     * 
     * @param enable Whether to enable progressive widening
     */
    void enableProgressiveWidening(bool enable) { config_.useProgressiveWidening = enable; }
    
    /**
     * @brief Get the number of nodes created in the search tree
     * 
     * @return Number of nodes
     */
    size_t getNodeCount() const { return stats_.nodesCreated.load(); }
    
    /**
     * @brief Get the number of evaluations performed
     * 
     * @return Number of evaluations
     */
    size_t getEvaluationCount() const { return stats_.evaluationCalls.load(); }
    
private:
    // Member variables
    std::unique_ptr<core::IGameState> rootState_;   // Current root state
    std::unique_ptr<MCTSNode> rootNode_;            // Root of the search tree
    alphazero::nn::NeuralNetwork* nn_;              // Neural network for evaluation
    TranspositionTable* tt_;                        // Optional transposition table
    std::unique_ptr<nn::BatchQueue> batchQueue_;    // Queue for batching network requests
    std::unique_ptr<ThreadPool> threadPool_;        // Thread pool for parallel search
    std::atomic<int> pendingSimulations_{0};        // Counter for remaining simulations
    
    // MCTS configuration
    MCTSConfig config_;
    
    // Statistics
    MCTSStats stats_;
    
    // Feature map cache for avoiding redundant tensor computation
    std::unordered_map<uint64_t, std::vector<std::vector<std::vector<float>>>> featureMapCache_;
    mutable std::mutex featureMapMutex_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniformDist_{0.0f, 1.0f};
    
    // Search control
    std::atomic<bool> searchInProgress_{false};
    std::condition_variable searchCondVar_;
    std::function<void(int, int)> progressCallback_;
    bool debugMode_{false};
    
    // Helper methods
    void initialize(const core::IGameState& rootState);
    void runSingleSimulation();
    MCTSNode* selectLeaf(core::IGameState& state);
    void expandNode(MCTSNode* node, const core::IGameState& state);
    void backpropagate(MCTSNode* node, float value);
    std::pair<std::vector<float>, float> evaluateState(const core::IGameState& state);
    std::pair<std::vector<float>, float> evaluateStateBatch(const std::vector<std::reference_wrapper<const core::IGameState>>& states);
    float getTemperatureVisitWeight(int visitCount, float temperature) const;
    MCTSNode* selectChildUcb(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildPuct(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildProgressiveBias(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildRave(MCTSNode* node, const core::IGameState& state);
    float convertToValue(core::GameResult result, int currentPlayer);
    float getDirichletAlpha() const;
    
    // Progressive widening
    int getProgressiveWideningCount(int parentVisits, int totalChildren) const;
    
    // Memory management helpers
    void pruneAllExcept(MCTSNode* nodeToKeep);
    
    // Feature map caching
    std::vector<std::vector<std::vector<float>>> getCachedFeatureMap(const core::IGameState& state);
    void cacheFeatureMap(uint64_t hash, const std::vector<std::vector<std::vector<float>>>& featureMap);
};

// Template implementation for ThreadPool::enqueue
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        
        // Don't allow enqueueing after stopping the pool
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        // Add task to queue
        tasks.push_back([task](){ (*task)(); });
    }
    
    // Wake up one worker thread
    condition.notify_one();
    return result;
}

} // namespace mcts
} // namespace alphazero

#endif // PARALLEL_MCTS_H