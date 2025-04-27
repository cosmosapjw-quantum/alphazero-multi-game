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
#include "alphazero/core/igamestate.h"
#include "alphazero/mcts/mcts_node.h"
#include "alphazero/mcts/transposition_table.h"

namespace alphazero {

// Forward declarations
namespace nn {
    class NeuralNetwork;
}

namespace mcts {

/**
 * @brief Node selection strategies for MCTS
 */
enum class MCTSNodeSelection {
    UCB,             // Standard UCB formula
    PUCT,            // AlphaZero's PUCT formula
    PROGRESSIVE_BIAS // Progressive bias with visit count
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
    void setCPuct(float cPuct) { cPuct_ = cPuct; }
    
    /**
     * @brief Set first play urgency reduction
     * 
     * @param fpuReduction FPU reduction value
     */
    void setFpuReduction(float fpuReduction) { fpuReduction_ = fpuReduction; }
    
    /**
     * @brief Set virtual loss amount
     * 
     * @param virtualLoss Virtual loss amount
     */
    void setVirtualLoss(int virtualLoss) { virtualLoss_ = virtualLoss; }
    
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
    void setSelectionStrategy(MCTSNodeSelection strategy) { selectionStrategy_ = strategy; }
    
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
     */
    void releaseMemory();
    
private:
    // Member variables
    std::unique_ptr<core::IGameState> rootState_;   // Current root state
    std::unique_ptr<MCTSNode> rootNode_;            // Root of the search tree
    alphazero::nn::NeuralNetwork* nn_;              // Neural network for evaluation
    TranspositionTable* tt_;                       // Optional transposition table
    std::unique_ptr<ThreadPool> threadPool_;       // Thread pool for parallel search
    std::atomic<int> pendingSimulations_{0};       // Counter for remaining simulations
    
    // MCTS parameters
    int numSimulations_;
    float cPuct_;
    float fpuReduction_;
    int virtualLoss_;
    bool deterministicMode_{false};
    MCTSNodeSelection selectionStrategy_{MCTSNodeSelection::PUCT};
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniformDist_{0.0f, 1.0f};
    
    // Search control
    std::atomic<bool> searchInProgress_{false};
    std::condition_variable searchCondVar_;
    std::function<void(int, int)> progressCallback_;
    bool debugMode_{false};
    
    // Helper methods
    void runSingleSimulation();
    MCTSNode* selectLeaf(core::IGameState& state);
    void expandNode(MCTSNode* node, const core::IGameState& state);
    void backpropagate(MCTSNode* node, float value);
    std::pair<std::vector<float>, float> evaluateState(const core::IGameState& state);
    float getTemperatureVisitWeight(int visitCount, float temperature) const;
    MCTSNode* selectChildUcb(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildPuct(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildProgressiveBias(MCTSNode* node, const core::IGameState& state);
    float convertToValue(core::GameResult result, int currentPlayer);
    float getDirichletAlpha() const;
    
    // Memory management helpers
    void pruneAllExcept(MCTSNode* nodeToKeep);
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