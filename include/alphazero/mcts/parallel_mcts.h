#ifndef ALPHAZERO_MCTS_PARALLEL_MCTS_H
#define ALPHAZERO_MCTS_PARALLEL_MCTS_H

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <random>
#include <functional>
#include <future>
#include <condition_variable>
#include <deque>
#include <optional>

#include "alphazero/mcts/mcts_node.h"
#include "alphazero/mcts/thread_pool.h"
#include "alphazero/mcts/transposition_table.h"
#include "alphazero/core/igamestate.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/batch_queue.h"

namespace alphazero {
namespace mcts {

// Node selection strategies
enum class MCTSNodeSelection {
    UCB,            // Upper Confidence Bound
    PUCT,           // Predictor + UCT (AlphaZero default)
    PROGRESSIVE_BIAS, // Progressive bias
    RAVE            // Rapid Action Value Estimation
};

// Search modes for MCTS
enum class MCTSSearchMode {
    SERIAL,    // Each simulation runs from start to finish
    PARALLEL,  // Multiple simulations run in parallel but independently
    BATCHED    // Simulations coordinate for batch evaluation
};

// Configuration for MCTS
struct MCTSConfig {
    int numThreads = 1;
    int numSimulations = 800;
    float cPuct = 1.5f;
    float fpuReduction = 0.0f;
    int virtualLoss = 3;
    int maxSearchDepth = 1000;
    bool useDirichletNoise = false;
    float dirichletAlpha = 0.03f;
    float dirichletEpsilon = 0.25f;
    bool useBatchInference = false;
    bool useTemporalDifference = false;
    float tdLambda = 0.8f;
    bool useProgressiveWidening = false;
    int minVisitsForWidening = 10;
    float progressiveWideningBase = 2.0f;
    float progressiveWideningExponent = 0.5f;
    MCTSNodeSelection selectionStrategy = MCTSNodeSelection::PUCT;
    int maxRetries = 3;
    int transpositionTableSize = 1048576;  // 1M entries
    uint64_t cacheEntryMaxAge = 60000;     // 60 seconds in ms
    bool useFmapCache = false;
    int batchSize = 16;
    
    // Batched MCTS parameters
    bool useBatchedMCTS = false;
    int batchTimeoutMs = 5;
    MCTSSearchMode searchMode = MCTSSearchMode::PARALLEL;
};

// Statistics for MCTS
struct MCTSStats {
    std::atomic<size_t> nodesCreated{0};
    std::atomic<size_t> nodesExpanded{0};
    std::atomic<size_t> nodesTotalVisits{0};
    std::atomic<size_t> simulationCount{0};
    std::atomic<size_t> evaluationCalls{0};
    std::atomic<size_t> cacheHits{0};
    std::atomic<size_t> cacheMisses{0};
    std::atomic<size_t> batchedEvaluations{0};
    std::atomic<size_t> totalBatches{0};
    
    void reset() {
        nodesCreated = 0;
        nodesExpanded = 0;
        nodesTotalVisits = 0;
        simulationCount = 0;
        evaluationCalls = 0;
        cacheHits = 0;
        cacheMisses = 0;
        batchedEvaluations = 0;
        totalBatches = 0;
    }
};

// Structure to hold information about a pending simulation in batched mode
struct PendingSimulation {
    std::unique_ptr<core::IGameState> state;
    MCTSNode* node;
    std::vector<MCTSNode*> searchPath;
    
    // For virtual loss handling
    bool virtualLossApplied = false;
    
    // For result handling
    std::future<std::pair<std::vector<float>, float>> evalFuture;
    bool isComplete = false;
    std::vector<float> policy;
    float value = 0.0f;
};

/**
 * @brief Parallel MCTS implementation
 * 
 * This class implements a parallel Monte Carlo Tree Search algorithm that can
 * run multiple simulations concurrently and batch neural network evaluations.
 * 
 * WARNING: Using a Python-backed neural network with this class will limit parallelism
 * due to the Python Global Interpreter Lock (GIL). For best performance, use a
 * C++-based neural network implementation like TorchNeuralNetwork that directly
 * uses LibTorch without Python bindings.
 * 
 * If you need to use a Python model, export it to LibTorch format first to achieve
 * proper parallelism.
 */
class ParallelMCTS {
public:
    // Constructor
    ParallelMCTS(
        const core::IGameState& rootState,
        nn::NeuralNetwork* nn = nullptr,
        TranspositionTable* tt = nullptr,
        int numThreads = 1,
        int numSimulations = 800,
        float cPuct = 1.5f,
        float fpuReduction = 0.0f,
        int virtualLoss = 3
    );
    
    // Constructor with config
    ParallelMCTS(
        const core::IGameState& rootState,
        const MCTSConfig& config,
        nn::NeuralNetwork* nn = nullptr,
        TranspositionTable* tt = nullptr
    );
    
    // Destructor
    ~ParallelMCTS();
    
    // Search methods
    void search();
    void runSingleSimulation();
    void runBatchedSearch();
    
    // Action selection
    int selectAction(bool isTraining = false, float temperature = 1.0f);
    std::vector<float> getActionProbabilities(float temperature = 1.0f) const;
    float getRootValue() const;
    
    // Tree navigation
    void updateWithMove(int action);
    
    // Exploration
    void addDirichletNoise(float alpha = 0.03f, float epsilon = 0.25f);
    
    // Configuration
    void setNumThreads(int numThreads);
    void setNumSimulations(int numSimulations);
    void setCPuct(float cPuct) { config_.cPuct = cPuct; }
    void setFpuReduction(float fpuReduction) { config_.fpuReduction = fpuReduction; }
    void setVirtualLoss(int virtualLoss) { config_.virtualLoss = virtualLoss; }
    void setNeuralNetwork(nn::NeuralNetwork* nn);
    void setTranspositionTable(TranspositionTable* tt);
    void setSelectionStrategy(MCTSNodeSelection strategy) { config_.selectionStrategy = strategy; }
    void setConfig(const MCTSConfig& config);
    
    // Batched MCTS configuration
    void enableBatchedMCTS(bool enable);
    void setBatchSize(int batchSize);
    void setBatchTimeout(int timeoutMs);
    
    // Debug methods
    void setDeterministicMode(bool enable);
    void setDebugMode(bool enable) { debugMode_ = enable; }
    void setProgressCallback(std::function<void(int, int)> callback) { progressCallback_ = callback; }
    void printSearchStats() const;
    std::string getSearchInfo() const;
    void printSearchPath(int action) const;
    
    // Memory management
    size_t getMemoryUsage() const;
    size_t releaseMemory(int visitThreshold = 10);
    
    // Analysis
    std::vector<std::tuple<int, int, float, float>> analyzePosition(int topN = 10) const;
    
private:
    // Node selection
    MCTSNode* selectLeaf(core::IGameState& state);
    MCTSNode* selectLeafWithPath(core::IGameState& state, std::vector<MCTSNode*>& searchPath);
    MCTSNode* selectChildUcb(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildPuct(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildProgressiveBias(MCTSNode* node, const core::IGameState& state);
    MCTSNode* selectChildRave(MCTSNode* node, const core::IGameState& state);
    
    // Node expansion
    void expandNode(MCTSNode* node, const core::IGameState& state);
    void expandNodeWithPolicy(MCTSNode* node, const core::IGameState& state, const std::vector<float>& policy);
    
    // Evaluation
    std::pair<std::vector<float>, float> evaluateState(const core::IGameState& state);
    std::pair<std::vector<float>, float> evaluateStateBatch(
        const std::vector<std::reference_wrapper<const core::IGameState>>& states);
    
    // Backpropagation
    void backpropagate(MCTSNode* node, float value);
    void backpropagate(MCTSNode* node, float value, const std::vector<MCTSNode*>& searchPath);
    
    // Batch processing
    void processCompletedEvaluations(
        std::vector<PendingSimulation>& pendingSimulations,
        std::mutex& simulationsMutex,
        std::atomic<int>& completedSimulations,
        bool processAll = false);
    
    // Helper methods
    float convertToValue(core::GameResult result, int currentPlayer);
    float getDirichletAlpha() const;
    float getTemperatureVisitWeight(int visitCount, float temperature) const;
    int getProgressiveWideningCount(int parentVisits, int totalChildren) const;
    void initialize(const core::IGameState& rootState);
    
    // FeatureMap cache
    std::vector<std::vector<std::vector<float>>> getCachedFeatureMap(const core::IGameState& state);
    void cacheFeatureMap(uint64_t hash, const std::vector<std::vector<std::vector<float>>>& featureMap);

// Private data
    MCTSConfig config_;
    std::unique_ptr<MCTSNode> rootNode_;
    std::unique_ptr<core::IGameState> rootState_;
    nn::NeuralNetwork* nn_;
    TranspositionTable* tt_;
    bool debugMode_;
    
    // Thread pool
    std::unique_ptr<ThreadPool> threadPool_;
    
    // Batch queue for neural network inference
    std::unique_ptr<nn::BatchQueue> batchQueue_;
    
    // Search control
    std::atomic<bool> searchInProgress_{false};
    std::atomic<int> pendingSimulations_{0};
    std::condition_variable searchCondVar_;
    std::mutex searchMutex_;
    
    // Feature map cache
    std::unordered_map<uint64_t, std::vector<std::vector<std::vector<float>>>> featureMapCache_;
    mutable std::mutex featureMapMutex_;
    
    // Random number generator
    mutable std::mt19937 rng_;
    
    // Progress reporting
    std::function<void(int, int)> progressCallback_;
    
    // Stats
    MCTSStats stats_;
    
    // Constants
    static constexpr int ESTIMATED_NODE_SIZE = 128;  // Estimated size of MCTSNode in bytes
};

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_PARALLEL_MCTS_H