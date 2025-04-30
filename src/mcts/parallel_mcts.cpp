// src/mcts/parallel_mcts.cpp
#include "alphazero/types.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/batch_queue.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <random>
#include <map>
#include <deque>

namespace alphazero {
namespace mcts {

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    
                    // Wait for task or stop signal
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    
                    // Exit if stop signal and no tasks
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    
                    // Get task from front
                    task = std::move(this->tasks.front());
                    this->tasks.pop_front();
                }
                
                // Execute task
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    
    // Wake up all threads
    condition.notify_all();
    
    // Join all threads
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

ParallelMCTS::ParallelMCTS(
    const core::IGameState& rootState,
    nn::NeuralNetwork* nn,
    TranspositionTable* tt,
    int numThreads,
    int numSimulations,
    float cPuct,
    float fpuReduction,
    int virtualLoss
) : nn_(nn),
    tt_(tt),
    debugMode_(false),
    searchInProgress_(false)
{
    // Set configuration
    config_.numThreads = numThreads;
    config_.numSimulations = numSimulations;
    config_.cPuct = cPuct;
    config_.fpuReduction = fpuReduction;
    config_.virtualLoss = virtualLoss;
    
    // Initialize with root state
    initialize(rootState);
}

ParallelMCTS::ParallelMCTS(
    const core::IGameState& rootState,
    const MCTSConfig& config,
    nn::NeuralNetwork* nn,
    TranspositionTable* tt
) : config_(config),
    nn_(nn),
    tt_(tt),
    debugMode_(false),
    searchInProgress_(false)
{
    // Initialize with root state
    initialize(rootState);
}

void ParallelMCTS::initialize(const core::IGameState& rootState) {
    // Initialize random number generator
    std::random_device rd;
    rng_.seed(rd());
    
    // Clone root state
    rootState_ = rootState.clone();
    
    // Create root node
    rootNode_ = std::make_unique<MCTSNode>(rootState_.get(), nullptr, 0.0f, -1);
    
    // Create thread pool
    threadPool_ = std::make_unique<ThreadPool>(config_.numThreads);
    
    // Create batch queue if using batched inference
    if (config_.useBatchInference && nn_) {
        batchQueue_ = std::make_unique<nn::BatchQueue>(nn_, config_.batchSize, 5);
    }
    
    // Create transposition table if not provided
    if (!tt_ && nn_) {
        tt_ = new TranspositionTable(config_.transpositionTableSize, 
                                   config_.transpositionTableSize / 1024);
        tt_->setReplacementPolicy(config_.cacheEntryMaxAge, 5);
    }
    
    // Reset statistics
    stats_.reset();
    stats_.nodesCreated.fetch_add(1, std::memory_order_relaxed);  // Count root node
}

ParallelMCTS::~ParallelMCTS() {
    // Stop any ongoing search
    if (searchInProgress_.load()) {
        searchInProgress_.store(false);
        searchCondVar_.notify_all();
    }
    
    // Delete transposition table if we created it
    if (tt_ && !config_.useBatchInference) {
        delete tt_;
        tt_ = nullptr;
    }
}

void ParallelMCTS::search() {
    // Don't start a new search if one is already in progress
    if (searchInProgress_.exchange(true)) {
        return;
    }
    
    // Set pending simulations counter
    pendingSimulations_.store(config_.numSimulations, std::memory_order_relaxed);
    
    // Make sure root node is expanded
    if (!rootNode_->isExpanded && !rootState_->isTerminal()) {
        expandNode(rootNode_.get(), *rootState_);
    }
    
    // Apply Dirichlet noise if in training mode
    if (config_.useDirichletNoise) {
        addDirichletNoise(config_.dirichletAlpha, config_.dirichletEpsilon);
    }
    
    // Launch worker threads
    int numThreads = threadPool_->size();
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < numThreads; ++i) {
        futures.push_back(threadPool_->enqueue([this] {
            while (pendingSimulations_.load(std::memory_order_relaxed) > 0 && 
                   searchInProgress_.load()) {
                runSingleSimulation();
            }
        }));
    }
    
    // Wait for all simulations to complete or search to be stopped
    for (auto& future : futures) {
        future.wait();
    }
    
    // Report final progress
    if (progressCallback_ && searchInProgress_.load()) {
        progressCallback_(config_.numSimulations, config_.numSimulations);
    }
    
    searchInProgress_.store(false);
}

void ParallelMCTS::runSingleSimulation() {
    // Exit if search was stopped
    if (!searchInProgress_.load()) {
        return;
    }
    
    // Skip if no more simulations needed
    if (pendingSimulations_.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        pendingSimulations_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    
    // Update simulation count
    stats_.simulationCount.fetch_add(1, std::memory_order_relaxed);
    
    // Clone the root state for this simulation
    std::unique_ptr<core::IGameState> state = rootState_->clone();
    
    // Select a leaf node
    MCTSNode* node = selectLeaf(*state);
    
    // If null node returned, simulation failed
    if (!node) {
        if (debugMode_) {
            std::cerr << "Warning: Null node selected in simulation" << std::endl;
        }
        return;
    }
    
    float value = 0.0f;
    
    // Expand the leaf node if it's not terminal
    if (!node->isTerminal) {
        expandNode(node, *state);
        
        // Evaluate the leaf node
        auto [policy, val] = evaluateState(*state);
        
        // Store in transposition table if available
        if (tt_) {
            tt_->store(state->getHash(), state->getGameType(), policy, val);
        }
        
        // For non-terminal leaf nodes, use neural network evaluation
        value = val;
    } else {
        // For terminal nodes, use game result
        value = node->getTerminalValue(state->getCurrentPlayer());
    }
    
    // Backpropagate the result
    backpropagate(node, value);
    
    // Report progress
    if (progressCallback_) {
        int completed = config_.numSimulations - pendingSimulations_.load(std::memory_order_relaxed);
        if (completed % (config_.numSimulations / 100 + 1) == 0 || completed == config_.numSimulations) {
            progressCallback_(completed, config_.numSimulations);
        }
    }
}

MCTSNode* ParallelMCTS::selectLeaf(core::IGameState& state) {
    MCTSNode* node = rootNode_.get();
    if (!node) return nullptr;
    
    // Add virtual loss to root node
    node->addVirtualLoss(config_.virtualLoss);
    
    // Keep track of search path for backpropagation
    std::deque<MCTSNode*> searchPath;
    searchPath.push_back(node);
    
    int currentDepth = 0;
    
    // Continue until we reach a leaf node
    while (node->isExpanded && !node->isTerminal && currentDepth < config_.maxSearchDepth) {
        // Apply progressive widening if configured
        if (config_.useProgressiveWidening && node->children.size() > 1) {
            int numVisits = node->visitCount.load(std::memory_order_relaxed);
            int wideningCount = getProgressiveWideningCount(numVisits, node->children.size());
            
            // If we're already at max width, continue as normal
            if (wideningCount < static_cast<int>(node->children.size())) {
                // First, remove virtual loss from nodes that will be pruned
                for (size_t i = wideningCount; i < node->children.size(); ++i) {
                    if (node->children[i]->visitCount.load(std::memory_order_relaxed) < 1) {
                        // Only "prune" nodes that haven't been visited yet
                        node->children[i]->addVirtualLoss(-config_.virtualLoss);
                    }
                }
            }
        }
        
        // Select child according to strategy
        MCTSNode* childNode = nullptr;
        switch (config_.selectionStrategy) {
            case MCTSNodeSelection::UCB:
                childNode = selectChildUcb(node, state);
                break;
            case MCTSNodeSelection::PROGRESSIVE_BIAS:
                childNode = selectChildProgressiveBias(node, state);
                break;
            case MCTSNodeSelection::RAVE:
                childNode = selectChildRave(node, state);
                break;
            case MCTSNodeSelection::PUCT:
            default:
                childNode = selectChildPuct(node, state);
                break;
        }
        
        if (!childNode) {
            break;
        }
        
        // Add virtual loss to selected child
        childNode->addVirtualLoss(config_.virtualLoss);
        
        // Find action index
        int actionIdx = -1;
        for (size_t i = 0; i < node->children.size(); ++i) {
            if (node->children[i].get() == childNode) {
                actionIdx = static_cast<int>(i);
                break;
            }
        }
        
        if (actionIdx < 0 || actionIdx >= static_cast<int>(node->actions.size())) {
            // Error finding action, break and remove virtual loss
            node->removeVirtualLoss(config_.virtualLoss);
            break;
        }
        
        // Make the move in the state
        int action = node->actions[actionIdx];
        try {
            state.makeMove(action);
        } catch (const std::exception& e) {
            // Error making move, break
            node->removeVirtualLoss(config_.virtualLoss);
            if (debugMode_) {
                std::cerr << "Error making move: " << e.what() << std::endl;
            }
            break;
        }
        
        // Add to search path, update node
        searchPath.push_back(childNode);
        node = childNode;
        currentDepth++;
    }
    
    // Calculate total visits for statistics
    stats_.nodesTotalVisits.fetch_add(searchPath.size(), std::memory_order_relaxed);
    
    return node;
}

MCTSNode* ParallelMCTS::selectChildPuct(MCTSNode* node, const core::IGameState& state) {
    int currentPlayer = state.getCurrentPlayer();
    int parentVisits = node->visitCount.load(std::memory_order_relaxed);
    
    // Find child with highest PUCT score
    float bestScore = -std::numeric_limits<float>::max();
    MCTSNode* bestChild = nullptr;
    
    // Apply progressive widening if configured
    int maxChildren = node->children.size();
    if (config_.useProgressiveWidening && parentVisits >= config_.minVisitsForWidening) {
        maxChildren = getProgressiveWideningCount(parentVisits, node->children.size());
    }
    
    // Only consider the top N children (widening limited)
    for (size_t i = 0; i < std::min(maxChildren, static_cast<int>(node->children.size())); ++i) {
        MCTSNode* child = node->children[i].get();
        float score = child->getPuctScore(config_.cPuct, currentPlayer, config_.fpuReduction, parentVisits);
        
        if (score > bestScore) {
            bestScore = score;
            bestChild = child;
        }
    }
    
    return bestChild;
}

MCTSNode* ParallelMCTS::selectChildUcb(MCTSNode* node, const core::IGameState& state) {
    int currentPlayer = state.getCurrentPlayer();
    int parentVisits = node->visitCount.load(std::memory_order_relaxed);
    
    // Find child with highest UCB score
    float bestScore = -std::numeric_limits<float>::max();
    MCTSNode* bestChild = nullptr;
    
    // Apply progressive widening if configured
    int maxChildren = node->children.size();
    if (config_.useProgressiveWidening && parentVisits >= config_.minVisitsForWidening) {
        maxChildren = getProgressiveWideningCount(parentVisits, node->children.size());
    }
    
    for (size_t i = 0; i < std::min(maxChildren, static_cast<int>(node->children.size())); ++i) {
        MCTSNode* child = node->children[i].get();
        float score = child->getUcbScore(config_.cPuct, currentPlayer, config_.fpuReduction, parentVisits);
        
        if (score > bestScore) {
            bestScore = score;
            bestChild = child;
        }
    }
    
    // If all children are unvisited, pick one randomly
    if (!bestChild && !node->children.empty()) {
        size_t randomIdx = 0;
        if (config_.useBatchInference) {
            randomIdx = node->visitCount.load() % node->children.size();
        } else {
            std::uniform_int_distribution<size_t> dist(0, node->children.size() - 1);
            randomIdx = dist(rng_);
        }
        bestChild = node->children[randomIdx].get();
    }
    
    return bestChild;
}

MCTSNode* ParallelMCTS::selectChildProgressiveBias(MCTSNode* node, const core::IGameState& state) {
    int currentPlayer = state.getCurrentPlayer();
    int parentVisits = node->visitCount.load(std::memory_order_relaxed);
    
    // Find child with highest progressive bias score
    float bestScore = -std::numeric_limits<float>::max();
    MCTSNode* bestChild = nullptr;
    
    // Apply progressive widening if configured
    int maxChildren = node->children.size();
    if (config_.useProgressiveWidening && parentVisits >= config_.minVisitsForWidening) {
        maxChildren = getProgressiveWideningCount(parentVisits, node->children.size());
    }
    
    for (size_t i = 0; i < std::min(maxChildren, static_cast<int>(node->children.size())); ++i) {
        MCTSNode* child = node->children[i].get();
        float score = child->getProgressiveBiasScore(config_.cPuct, currentPlayer, parentVisits);
        
        if (score > bestScore) {
            bestScore = score;
            bestChild = child;
        }
    }
    
    return bestChild;
}

MCTSNode* ParallelMCTS::selectChildRave(MCTSNode* node, const core::IGameState& state) {
    // Not yet implemented - fallback to PUCT
    return selectChildPuct(node, state);
}

void ParallelMCTS::expandNode(MCTSNode* node, const core::IGameState& state) {
    // Lock to prevent multiple threads from expanding the same node
    std::lock_guard<std::mutex> lock(node->expansionMutex);
    
    // Check if node is already expanded or terminal
    if (node->isExpanded || node->isTerminal) {
        return;
    }
    
    // Get legal moves
    std::vector<int> legalMoves = state.getLegalMoves();
    
    // If no legal moves, mark as terminal
    if (legalMoves.empty()) {
        node->isTerminal = true;
        node->gameResult = state.getGameResult();
        node->isExpanded = true;
        return;
    }
    
    // Get policy from neural network or transposition table
    std::vector<float> policy;
    float value;
    
    if (tt_) {
        // Try to get from transposition table first
        TranspositionTable::Entry entry;
        if (tt_->lookup(state.getHash(), state.getGameType(), entry)) {
            policy = entry.policy;
            value = entry.value;
            stats_.cacheHits.fetch_add(1, std::memory_order_relaxed);
        } else {
            stats_.cacheMisses.fetch_add(1, std::memory_order_relaxed);
            std::tie(policy, value) = evaluateState(state);
            tt_->store(state.getHash(), state.getGameType(), policy, value);
        }
    } else {
        // Evaluate with neural network
        stats_.cacheMisses.fetch_add(1, std::memory_order_relaxed);
        std::tie(policy, value) = evaluateState(state);
    }
    
    // Normalize policy for legal moves
    float policySum = 0.0f;
    std::vector<float> legalPolicy(legalMoves.size(), 0.0f);
    
    for (size_t i = 0; i < legalMoves.size(); ++i) {
        int action = legalMoves[i];
        if (action >= 0 && action < static_cast<int>(policy.size())) {
            legalPolicy[i] = policy[action];
            policySum += legalPolicy[i];
        }
    }
    
    // Normalize if sum is not zero
    if (policySum > 0.0f) {
        for (size_t i = 0; i < legalPolicy.size(); ++i) {
            legalPolicy[i] /= policySum;
        }
    } else {
        // Uniform policy if all probabilities are zero
        float uniformProb = 1.0f / static_cast<float>(legalMoves.size());
        for (size_t i = 0; i < legalPolicy.size(); ++i) {
            legalPolicy[i] = uniformProb;
        }
    }
    
    // Create child nodes
    for (size_t i = 0; i < legalMoves.size(); ++i) {
        int action = legalMoves[i];
        float prior = legalPolicy[i];
        
        // Create child node
        auto childNode = std::make_unique<MCTSNode>(nullptr, node, prior, action);
        
        // Update statistics
        stats_.nodesCreated.fetch_add(1, std::memory_order_relaxed);
        
        // Add child to parent
        node->addChild(action, prior, std::move(childNode));
    }
    
    // Mark node as expanded
    node->isExpanded = true;
    stats_.nodesExpanded.fetch_add(1, std::memory_order_relaxed);
}

void ParallelMCTS::backpropagate(MCTSNode* node, float value) {
    MCTSNode* current = node;
    
    while (current) {
        // Remove virtual loss first
        current->removeVirtualLoss(config_.virtualLoss);
        
        // Update statistics
        current->visitCount.fetch_add(1, std::memory_order_relaxed);
        
        // For valueSum, use a compare-exchange loop since it's a float atomic
        float oldValue = current->valueSum.load(std::memory_order_relaxed);
        float nodeValue = value;
        float newValue = oldValue + nodeValue;
        
        while (!current->valueSum.compare_exchange_weak(oldValue, newValue,
                                                      std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
            // If the exchange failed, oldValue has been updated with the current value
            newValue = oldValue + nodeValue;
        }
        
        // Flip value sign for opponent's perspective
        value = -value;
        
        // Implement temporal difference if configured
        if (config_.useTemporalDifference && current->parent) {
            float parentValue = current->parent->getValue();
            // TD(λ) formula: value = (1-λ) * next_state_value + λ * terminal_value
            value = (1.0f - config_.tdLambda) * (-parentValue) + config_.tdLambda * value;
        }
        
        // Move to parent
        current = current->parent;
    }
}

std::pair<std::vector<float>, float> ParallelMCTS::evaluateState(const core::IGameState& state) {
    // Increment evaluation count
    stats_.evaluationCalls.fetch_add(1, std::memory_order_relaxed);
    
    // Handle terminal states
    if (state.isTerminal()) {
        float value = convertToValue(state.getGameResult(), state.getCurrentPlayer());
        
        // Create a uniform policy for terminal states
        std::vector<float> policy(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
        return {policy, value};
    }
    
    // Check transposition table first
    if (tt_) {
        TranspositionTable::Entry entry;
        if (tt_->lookup(state.getHash(), state.getGameType(), entry)) {
            stats_.cacheHits.fetch_add(1, std::memory_order_relaxed);
            return {entry.policy, entry.value};
        }
        stats_.cacheMisses.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Use neural network if available
    if (nn_) {
        return nn_->predict(state);
    }
    
    // Fallback: use uniform random policy
    std::vector<float> policy(state.getActionSpaceSize(), 0.0f);
    std::vector<int> legalActions = state.getLegalMoves();
    
    if (!legalActions.empty()) {
        float prob = 1.0f / legalActions.size();
        for (int action : legalActions) {
            if (action >= 0 && action < static_cast<int>(policy.size())) {
                policy[action] = prob;
            }
        }
    }
    
    return {policy, 0.0f};
}

std::pair<std::vector<float>, float> ParallelMCTS::evaluateStateBatch(
    const std::vector<std::reference_wrapper<const core::IGameState>>& states) {
    
    // Not implemented yet - just evaluate first state as fallback
    if (states.empty()) {
        std::vector<float> emptyPolicy;
        return {emptyPolicy, 0.0f};
    }
    
    return evaluateState(states[0].get());
}

std::vector<std::vector<std::vector<float>>> ParallelMCTS::getCachedFeatureMap(const core::IGameState& state) {
    if (!config_.useFmapCache) {
        return state.getEnhancedTensorRepresentation();
    }
    
    uint64_t hash = state.getHash();
    
    // Check cache
    {
        std::lock_guard<std::mutex> lock(featureMapMutex_);
        auto it = featureMapCache_.find(hash);
        if (it != featureMapCache_.end()) {
            return it->second;
        }
    }
    
    // Cache miss, compute and cache
    auto featureMap = state.getEnhancedTensorRepresentation();
    
    {
        std::lock_guard<std::mutex> lock(featureMapMutex_);
        // Check size before inserting to avoid excessive memory usage
        if (featureMapCache_.size() < 1000) {  // Limit cache size
            featureMapCache_[hash] = featureMap;
        }
    }
    
    return featureMap;
}

void ParallelMCTS::cacheFeatureMap(uint64_t hash, const std::vector<std::vector<std::vector<float>>>& featureMap) {
    if (!config_.useFmapCache) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(featureMapMutex_);
    // Check size before inserting to avoid excessive memory usage
    if (featureMapCache_.size() < 1000) {  // Limit cache size
        featureMapCache_[hash] = featureMap;
    }
}

float ParallelMCTS::convertToValue(core::GameResult result, int currentPlayer) {
    switch (result) {
        case core::GameResult::WIN_PLAYER1:
            return currentPlayer == 1 ? 1.0f : -1.0f;
        case core::GameResult::WIN_PLAYER2:
            return currentPlayer == 2 ? 1.0f : -1.0f;
        case core::GameResult::DRAW:
            return 0.0f;
        case core::GameResult::ONGOING:
        default:
            return 0.0f;
    }
}

int ParallelMCTS::selectAction(bool isTraining, float temperature) {
    if (!rootNode_->isExpanded) {
        // If root is not expanded, expand it first
        search();
    }
    
    // Handle edge cases
    if (rootNode_->isTerminal || !rootNode_->hasChildren()) {
        // Find a random legal move from the root state rather than returning -1
        std::vector<int> legalMoves = rootState_->getLegalMoves();
        if (legalMoves.empty()) {
            return -1;  // Still return -1 if no legal moves exist
        }
        
        if (config_.useBatchInference) {
            // In deterministic mode, return the first legal move
            return legalMoves[0];
        } else {
            // Random selection in non-deterministic mode
            std::uniform_int_distribution<size_t> dist(0, legalMoves.size() - 1);
            size_t idx = dist(rng_);
            return legalMoves[idx];
        }
    }
    
    // In training mode with temperature > 0, use stochastic selection
    if (isTraining && temperature > 0.0f) {
        // Get visit count distribution
        std::vector<float> distribution = getActionProbabilities(temperature);
        
        // Sample action from distribution
        if (config_.useBatchInference) {
            // Deterministic version: pick highest probability
            auto maxIt = std::max_element(distribution.begin(), distribution.end());
            return rootNode_->actions[std::distance(distribution.begin(), maxIt)];
        } else {
            // Stochastic version: sample according to distribution
            std::discrete_distribution<int> dist(distribution.begin(), distribution.end());
            int idx = dist(rng_);
            return rootNode_->actions[idx];
        }
    } else {
        // In evaluation mode or temperature=0
        // Get all actions with highest visit count
        std::vector<int> bestActions = rootNode_->getBestActions();
        
        if (bestActions.empty()) {
            return -1;
        }
        
        if (bestActions.size() == 1 || config_.useBatchInference) {
            // Only one best action or in deterministic mode, return the first
            return bestActions[0];
        } else {
            // Randomly select one of the best actions
            std::uniform_int_distribution<size_t> dist(0, bestActions.size() - 1);
            size_t idx = dist(rng_);
            return bestActions[idx];
        }
    }
}

std::vector<float> ParallelMCTS::getActionProbabilities(float temperature) const {
    if (!rootNode_->isExpanded || !rootNode_->hasChildren()) {
        return std::vector<float>();
    }
    
    return rootNode_->getVisitCountDistribution(temperature);
}

float ParallelMCTS::getRootValue() const {
    if (!rootNode_->hasChildren()) {
        return 0.0f;
    }
    
    return rootNode_->getValue();
}

void ParallelMCTS::updateWithMove(int action) {
    // Find child node corresponding to action
    MCTSNode* childNode = nullptr;
    size_t childIdx = 0;
    
    for (size_t i = 0; i < rootNode_->actions.size(); ++i) {
        if (rootNode_->actions[i] == action) {
            childNode = rootNode_->children[i].get();
            childIdx = i;
            break;
        }
    }
    
    // Update root state
    try {
        rootState_->makeMove(action);
    } catch (const std::exception& e) {
        // Invalid move, keep current state
        if (debugMode_) {
            std::cerr << "Invalid move in updateWithMove: " << e.what() << std::endl;
        }
        return;
    }
    
    // If child exists, make it the new root
    if (childNode) {
        // Detach the child from parent
        std::unique_ptr<MCTSNode> newRoot = std::move(rootNode_->children[childIdx]);
        newRoot->parent = nullptr;
        
        // Replace root
        rootNode_ = std::move(newRoot);
    } else {
        // Child doesn't exist, create new root
        rootNode_ = std::make_unique<MCTSNode>(rootState_.get(), nullptr, 0.0f, -1);
        stats_.nodesCreated.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Clear feature map cache when position changes
    if (config_.useFmapCache) {
        std::lock_guard<std::mutex> lock(featureMapMutex_);
        featureMapCache_.clear();
    }
}

void ParallelMCTS::addDirichletNoise(float alpha, float epsilon) {
    if (!rootNode_) {
        return;
    }

    // Make sure the root node is expanded before adding noise
    if (!rootNode_->isExpanded) {
        // Expand root node first
        try {
            expandNode(rootNode_.get(), *rootState_);
        } catch (const std::exception& e) {
            // Failed to expand, don't add noise
            return;
        }
    }
    
    // Check if we have children after expansion
    if (rootNode_->children.empty()) {
        return;
    }
    
    // Get Dirichlet alpha based on game type
    float dirichletAlpha = alpha > 0.0f ? alpha : getDirichletAlpha();

    try {
        // Generate Dirichlet noise safely
        std::vector<float> noise(rootNode_->children.size());
        std::gamma_distribution<float> gamma(dirichletAlpha, 1.0f);
        float noiseSum = 0.0f;
        
        for (size_t i = 0; i < noise.size(); ++i) {
            noise[i] = std::max(1e-10f, gamma(rng_));  // Ensure positive values
            noiseSum += noise[i];
        }
        
        // Guard against zero sum
        if (noiseSum <= 0.0f) {
            noiseSum = 1.0f;
            for (size_t i = 0; i < noise.size(); ++i) {
                noise[i] = 1.0f / noise.size();
            }
        }
        
        // Normalize noise
        for (size_t i = 0; i < noise.size(); ++i) {
            noise[i] /= noiseSum;
        }
        
        // Mix noise with prior probabilities
        for (size_t i = 0; i < rootNode_->children.size(); ++i) {
            auto child = rootNode_->children[i].get();
            if (child) {
                float oldPrior = child->prior;
                float newPrior = (1.0f - epsilon) * oldPrior + epsilon * noise[i];
                child->prior = newPrior;
            }
        }
    } catch (const std::exception& e) {
        // Handle any potential errors during noise generation
        // Just continue without adding noise
    }
}

void ParallelMCTS::setNumThreads(int numThreads) {
    // Wait for any ongoing search to complete
    while (searchInProgress_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    config_.numThreads = numThreads;
    
    // Create new thread pool
    threadPool_ = std::make_unique<ThreadPool>(numThreads);
}

void ParallelMCTS::setNumSimulations(int numSimulations) {
    // Wait for any ongoing search to complete
    while (searchInProgress_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    config_.numSimulations = numSimulations;
}

void ParallelMCTS::setNeuralNetwork(nn::NeuralNetwork* nn) {
    nn_ = nn;
    
    // Re-create batch queue if needed
    if (config_.useBatchInference && nn_) {
        batchQueue_ = std::make_unique<nn::BatchQueue>(nn_, config_.batchSize, 5);
    } else {
        batchQueue_.reset();
    }
}

void ParallelMCTS::setTranspositionTable(TranspositionTable* tt) {
    // Delete old table if we created it
    if (tt_ && !config_.useBatchInference) {
        delete tt_;
    }
    
    tt_ = tt;
    
    // Create new table if none provided
    if (!tt_ && nn_) {
        tt_ = new TranspositionTable(config_.transpositionTableSize, 
                                   config_.transpositionTableSize / 1024);
        tt_->setReplacementPolicy(config_.cacheEntryMaxAge, 5);
    }
}

void ParallelMCTS::setConfig(const MCTSConfig& config) {
    // Wait for any ongoing search to complete
    while (searchInProgress_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    config_ = config;
    
    // Update thread pool if needed
    if (threadPool_->size() != static_cast<size_t>(config.numThreads)) {
        threadPool_ = std::make_unique<ThreadPool>(config.numThreads);
    }
    
    // Update batch queue if needed
    if (config.useBatchInference && nn_) {
        batchQueue_ = std::make_unique<nn::BatchQueue>(nn_, config.batchSize, 5);
    } else {
        batchQueue_.reset();
    }
    
    // Update transposition table if needed
    if (tt_ && tt_->getSize() != static_cast<size_t>(config.transpositionTableSize)) {
        if (!config_.useBatchInference) {
            delete tt_;
        }
        
        tt_ = new TranspositionTable(config.transpositionTableSize, 
                                   config.transpositionTableSize / 1024);
        tt_->setReplacementPolicy(config.cacheEntryMaxAge, 5);
    }
}

void ParallelMCTS::setDeterministicMode(bool enable) {
    config_.useBatchInference = enable;
    
    if (enable) {
        // Use fixed seed for deterministic results
        rng_.seed(42);
    } else {
        // Use random seed
        std::random_device rd;
        rng_.seed(rd());
    }
}

float ParallelMCTS::getDirichletAlpha() const {
    // Alpha parameter for Dirichlet noise depends on game
    switch (rootNode_->gameType) {
        case core::GameType::CHESS:
            return 0.3f;  // Chess: smaller alpha, more concentrated
        case core::GameType::GO:
            return 0.03f;  // Go: much smaller alpha
        case core::GameType::GOMOKU:
        default:
            return 0.03f;  // Gomoku: similar to Go
    }
}

float ParallelMCTS::getTemperatureVisitWeight(int visitCount, float temperature) const {
    // Apply temperature to visit count
    if (temperature > 0.0f) {
        return std::pow(visitCount, 1.0f / temperature);
    } else {
        // For temperature = 0, use deterministic selection
        return static_cast<float>(visitCount);
    }
}

int ParallelMCTS::getProgressiveWideningCount(int parentVisits, int totalChildren) const {
    // Formula: k * N^alpha where:
    // - k is the base constant
    // - N is the parent visit count
    // - alpha is the exponent (typically 0.5-0.8)
    
    float base = static_cast<float>(config_.progressiveWideningBase);
    float exponent = config_.progressiveWideningExponent;
    
    // Calculate widening count
    int wideningCount = static_cast<int>(base * std::pow(parentVisits, exponent));
    
    // Ensure at least 1 child and at most totalChildren
    return std::max(1, std::min(wideningCount, totalChildren));
}

void ParallelMCTS::printSearchStats() const {
    std::cout << getSearchInfo() << std::endl;
}

std::string ParallelMCTS::getSearchInfo() const {
    std::stringstream ss;
    if (!rootNode_->isExpanded) {
        ss << "Root node not expanded";
        return ss.str();
    }
    
    ss << "Search stats:" << std::endl;
    ss << "  Total visits: " << rootNode_->visitCount.load() << std::endl;
    ss << "  Root value: " << std::fixed << std::setprecision(3) << rootNode_->getValue() << std::endl;
    ss << "  Nodes created: " << stats_.nodesCreated.load() << std::endl;
    ss << "  Nodes expanded: " << stats_.nodesExpanded.load() << std::endl;
    ss << "  Evaluations: " << stats_.evaluationCalls.load() << std::endl;
    ss << "  Cache hits/misses: " << stats_.cacheHits.load() << "/" << stats_.cacheMisses.load() << std::endl;
    
    if (stats_.cacheHits.load() + stats_.cacheMisses.load() > 0) {
        float hitRate = static_cast<float>(stats_.cacheHits.load()) / 
                      (stats_.cacheHits.load() + stats_.cacheMisses.load());
        ss << "  Cache hit rate: " << std::fixed << std::setprecision(2) << (hitRate * 100.0f) << "%" << std::endl;
    }
    
    ss << "  Child visits:" << std::endl;
    
    // Sort children by visit count
    std::vector<std::tuple<int, int, float, float>> visitCounts;
    for (size_t i = 0; i < rootNode_->children.size(); ++i) {
        visitCounts.emplace_back(
            rootNode_->actions[i],
            rootNode_->children[i]->visitCount.load(),
            rootNode_->children[i]->getValue(),
            rootNode_->children[i]->prior
        );
    }
    
    std::sort(visitCounts.begin(), visitCounts.end(),
              [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });
    
    // Display top 10 moves
    int numToShow = std::min(10, static_cast<int>(visitCounts.size()));
    for (int i = 0; i < numToShow; ++i) {
        auto [action, visits, value, prior] = visitCounts[i];
        
        float visitPercent = 100.0f * visits / rootNode_->visitCount.load();
        ss << "    Action " << std::setw(3) << action 
           << ": visits=" << std::setw(5) << visits 
           << " (" << std::fixed << std::setprecision(1) << visitPercent << "%)"
           << ", value=" << std::fixed << std::setprecision(3) << value
           << ", prior=" << std::fixed << std::setprecision(3) << prior;
        
        // Add UCB score if available
        auto child = rootNode_->getChildForAction(action);
        if (child) {
            float ucbScore = child->getPuctScore(config_.cPuct, rootState_->getCurrentPlayer(), 
                                              config_.fpuReduction, rootNode_->visitCount.load());
            ss << ", UCB=" << std::fixed << std::setprecision(3) << ucbScore;
        }
        
        ss << std::endl;
    }
    
    return ss.str();
}

void ParallelMCTS::printSearchPath(int action) const {
    // Find child
    MCTSNode* child = nullptr;
    for (size_t i = 0; i < rootNode_->actions.size(); ++i) {
        if (rootNode_->actions[i] == action) {
            child = rootNode_->children[i].get();
            break;
        }
    }
    
    if (!child) {
        std::cout << "Action " << action << " not found in children" << std::endl;
        return;
    }
    
    std::cout << "Path for action " << action << ":" << std::endl;
    std::cout << "  Root: V=" << rootNode_->visitCount.load() 
              << ", Q=" << std::fixed << std::setprecision(3) << rootNode_->getValue() 
              << std::endl;
    
    std::cout << "  Child: V=" << child->visitCount.load() 
              << ", Q=" << std::fixed << std::setprecision(3) << child->getValue()
              << ", P=" << std::fixed << std::setprecision(3) << child->prior
              << std::endl;
    
    // Print top grandchildren
    if (child->isExpanded && !child->children.empty()) {
        std::cout << "  Grandchildren:" << std::endl;
        
        std::vector<std::tuple<int, int, float, float>> visitCounts;
        for (size_t i = 0; i < child->children.size(); ++i) {
            visitCounts.emplace_back(
                child->actions[i],
                child->children[i]->visitCount.load(),
                child->children[i]->getValue(),
                child->children[i]->prior
            );
        }
        
        std::sort(visitCounts.begin(), visitCounts.end(),
                [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });
        
        int numToShow = std::min(5, static_cast<int>(visitCounts.size()));
        for (int i = 0; i < numToShow; ++i) {
            auto [gcAction, visits, value, prior] = visitCounts[i];
            
            float visitPercent = 100.0f * visits / child->visitCount.load();
            std::cout << "    Action " << std::setw(3) << gcAction 
                     << ": visits=" << std::setw(5) << visits 
                     << " (" << std::fixed << std::setprecision(1) << visitPercent << "%)"
                     << ", value=" << std::fixed << std::setprecision(3) << value
                     << ", prior=" << std::fixed << std::setprecision(3) << prior
                     << std::endl;
        }
    }
}

size_t ParallelMCTS::getMemoryUsage() const {
    size_t totalSize = 0;
    
    // Size of this object
    totalSize += sizeof(*this);
    
    // Size of root state
    totalSize += sizeof(*rootState_);
    
    // Size of tree
    if (rootNode_) {
        totalSize += rootNode_->getTreeMemoryUsage();
    }
    
    // Size of transposition table
    if (tt_) {
        totalSize += tt_->getMemoryUsageBytes();
    }
    
    // Size of feature map cache
    {
        std::lock_guard<std::mutex> lock(featureMapMutex_);
        totalSize += sizeof(decltype(featureMapCache_)) + featureMapCache_.size() * 
                   (sizeof(uint64_t) + sizeof(std::vector<std::vector<std::vector<float>>>));
        
        // Rough estimation of feature map sizes
        for (const auto& [hash, featureMap] : featureMapCache_) {
            if (!featureMap.empty() && !featureMap[0].empty() && !featureMap[0][0].empty()) {
                size_t singleMapSize = featureMap.size() * featureMap[0].size() * featureMap[0][0].size() * sizeof(float);
                totalSize += singleMapSize;
            }
        }
    }
    
    return totalSize;
}

size_t ParallelMCTS::releaseMemory(int visitThreshold) {
    if (!rootNode_) {
        return 0;
    }
    
    // Prune tree
    size_t prunedNodes = rootNode_->pruneTree(visitThreshold);
    
    // Clear feature map cache
    {
        std::lock_guard<std::mutex> lock(featureMapMutex_);
        featureMapCache_.clear();
    }
    
    return prunedNodes;
}

std::vector<std::tuple<int, int, float, float>> ParallelMCTS::analyzePosition(int topN) const {
    std::vector<std::tuple<int, int, float, float>> result;
    
    if (!rootNode_ || !rootNode_->isExpanded) {
        return result;
    }
    
    // Sort children by visit count
    std::vector<std::tuple<int, int, float, float>> visitCounts;
    for (size_t i = 0; i < rootNode_->children.size(); ++i) {
        visitCounts.emplace_back(
            rootNode_->actions[i],
            rootNode_->children[i]->visitCount.load(),
            rootNode_->children[i]->getValue(),
            rootNode_->children[i]->prior
        );
    }
    
    std::sort(visitCounts.begin(), visitCounts.end(),
              [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });
    
    // Return top N moves
    int numToReturn = std::min(topN, static_cast<int>(visitCounts.size()));
    return std::vector<std::tuple<int, int, float, float>>(
        visitCounts.begin(), visitCounts.begin() + numToReturn);
}

} // namespace mcts
} // namespace alphazero