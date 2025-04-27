// parallel_mcts.cpp
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <random>

namespace alphazero {
namespace mcts {

using nn::NeuralNetwork;

ParallelMCTS::ParallelMCTS(
    const core::IGameState& rootState,
    NeuralNetwork* nn,
    TranspositionTable* tt,
    int numThreads,
    int numSimulations,
    float cPuct,
    float fpuReduction,
    int virtualLoss
) : rootState_(rootState.clone()),
    nn_(nn),
    tt_(tt),
    threadPool_(std::make_unique<ThreadPool>(numThreads)),
    numSimulations_(numSimulations),
    cPuct_(cPuct),
    fpuReduction_(fpuReduction),
    virtualLoss_(virtualLoss),
    deterministicMode_(false),
    selectionStrategy_(MCTSNodeSelection::PUCT),
    debugMode_(false)
{
    // Initialize random number generator
    std::random_device rd;
    rng_.seed(rd());
    
    // Create root node
    rootNode_ = std::make_unique<MCTSNode>(rootState_.get(), nullptr, 0.0f);
}

ParallelMCTS::~ParallelMCTS() {
    // Stop any ongoing search
    if (searchInProgress_.load()) {
        searchInProgress_.store(false);
        searchCondVar_.notify_all();
    }
}

void ParallelMCTS::search() {
    // Don't start a new search if one is already in progress
    if (searchInProgress_.exchange(true)) {
        return;
    }
    
    // Set pending simulations counter
    pendingSimulations_.store(numSimulations_, std::memory_order_relaxed);
    
    // Make sure root node is expanded
    if (!rootNode_->isExpanded && !rootState_->isTerminal()) {
        expandNode(rootNode_.get(), *rootState_);
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
        progressCallback_(numSimulations_, numSimulations_);
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
    
    // Clone the root state for this simulation
    std::unique_ptr<core::IGameState> state = rootState_->clone();
    
    // Select a leaf node
    MCTSNode* node = selectLeaf(*state);
    
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
        int completed = numSimulations_ - pendingSimulations_.load(std::memory_order_relaxed);
        if (completed % 100 == 0 || completed == numSimulations_) {
            progressCallback_(completed, numSimulations_);
        }
    }
}

MCTSNode* ParallelMCTS::selectLeaf(core::IGameState& state) {
    MCTSNode* node = rootNode_.get();
    
    // Add virtual loss to root node
    node->addVirtualLoss(virtualLoss_);
    
    // Continue until we reach a leaf node
    while (node->isExpanded && !node->isTerminal) {
        // Select child according to strategy
        MCTSNode* childNode = nullptr;
        switch (selectionStrategy_) {
            case MCTSNodeSelection::UCB:
                childNode = selectChildUcb(node, state);
                break;
            case MCTSNodeSelection::PROGRESSIVE_BIAS:
                childNode = selectChildProgressiveBias(node, state);
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
        childNode->addVirtualLoss(virtualLoss_);
        
        // Find action index
        int actionIdx = -1;
        for (size_t i = 0; i < node->children.size(); ++i) {
            if (node->children[i].get() == childNode) {
                actionIdx = static_cast<int>(i);
                break;
            }
        }
        
        if (actionIdx < 0 || actionIdx >= static_cast<int>(node->actions.size())) {
            // Error finding action, break
            node->removeVirtualLoss(virtualLoss_);
            break;
        }
        
        // Make the move in the state
        int action = node->actions[actionIdx];
        try {
            state.makeMove(action);
        } catch (const std::exception& e) {
            // Error making move, break
            node->removeVirtualLoss(virtualLoss_);
            if (debugMode_) {
                std::cerr << "Error making move: " << e.what() << std::endl;
            }
            break;
        }
        
        // Remove virtual loss from parent, move to child
        node->removeVirtualLoss(virtualLoss_);
        node = childNode;
    }
    
    return node;
}

MCTSNode* ParallelMCTS::selectChildPuct(MCTSNode* node, const core::IGameState& state) {
    int currentPlayer = state.getCurrentPlayer();
    
    // Find child with highest UCB score
    float bestScore = -std::numeric_limits<float>::max();
    MCTSNode* bestChild = nullptr;
    
    // Apply different selection strategies based on node's visit count
    for (size_t i = 0; i < node->children.size(); ++i) {
        MCTSNode* child = node->children[i].get();
        float score = child->getUcbScore(cPuct_, currentPlayer, fpuReduction_);
        
        if (score > bestScore) {
            bestScore = score;
            bestChild = child;
        }
    }
    
    return bestChild;
}

MCTSNode* ParallelMCTS::selectChildUcb(MCTSNode* node, const core::IGameState& state) {
    int currentPlayer = state.getCurrentPlayer();
    float totalVisits = static_cast<float>(node->visitCount.load());
    
    if (totalVisits == 0) {
        return nullptr;
    }
    
    // Find child with highest UCB score
    float bestScore = -std::numeric_limits<float>::max();
    MCTSNode* bestChild = nullptr;
    
    for (size_t i = 0; i < node->children.size(); ++i) {
        MCTSNode* child = node->children[i].get();
        float childVisits = static_cast<float>(child->visitCount.load());
        
        // Skip unvisited nodes
        if (childVisits == 0) {
            continue;
        }
        
        // UCB1 formula: exploitation + exploration
        float exploit = -child->getValue();  // Negated for current player
        if (currentPlayer == 1) {
            exploit = -exploit;  // Adjust sign based on player
        }
        
        float explore = std::sqrt(2.0f * std::log(totalVisits) / childVisits);
        float score = exploit + cPuct_ * explore;
        
        if (score > bestScore) {
            bestScore = score;
            bestChild = child;
        }
    }
    
    // If all children are unvisited, pick one randomly
    if (!bestChild && !node->children.empty()) {
        size_t randomIdx = 0;
        if (deterministicMode_) {
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
    float totalVisits = static_cast<float>(node->visitCount.load());
    
    if (totalVisits == 0) {
        return nullptr;
    }
    
    // Find child with highest score
    float bestScore = -std::numeric_limits<float>::max();
    MCTSNode* bestChild = nullptr;
    
    for (size_t i = 0; i < node->children.size(); ++i) {
        MCTSNode* child = node->children[i].get();
        float childVisits = static_cast<float>(child->visitCount.load());
        
        // Exploration term decreases with visit count
        float explorationFactor = childVisits > 0 ? cPuct_ / std::sqrt(childVisits) : cPuct_;
        
        // Progressive bias formula
        float exploitation = childVisits > 0 ? -child->getValue() : 0.0f;
        if (currentPlayer == 1) {
            exploitation = -exploitation;
        }
        
        float exploration = child->prior * explorationFactor;
        float score = exploitation + exploration;
        
        if (score > bestScore) {
            bestScore = score;
            bestChild = child;
        }
    }
    
    return bestChild;
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
        } else {
            std::tie(policy, value) = evaluateState(state);
            tt_->store(state.getHash(), state.getGameType(), policy, value);
        }
    } else {
        // Evaluate with neural network
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
        
        // Clone state and make move
        std::unique_ptr<core::IGameState> childState = state.clone();
        try {
            childState->makeMove(action);
        } catch (const std::exception& e) {
            if (debugMode_) {
                std::cerr << "Error creating child state: " << e.what() << std::endl;
            }
            continue;
        }
        
        // Create child node
        auto childNode = std::make_unique<MCTSNode>(nullptr, node, prior);
        
        // Check if terminal
        childNode->isTerminal = childState->isTerminal();
        if (childNode->isTerminal) {
            childNode->gameResult = childState->getGameResult();
        }
        
        // Store hash
        childNode->stateHash = childState->getHash();
        childNode->gameType = childState->getGameType();
        
        // Add child to parent
        node->addChild(action, prior, std::move(childNode));
    }
    
    // Mark node as expanded
    node->isExpanded = true;
}

void ParallelMCTS::backpropagate(MCTSNode* node, float value) {
    // Convert value to [-1, 1] for minimax
    float nodeValue = value;
    
    // Update statistics up the tree
    MCTSNode* current = node;
    
    while (current != nullptr) {
        // Update visit count and value sum
        current->visitCount.fetch_add(1, std::memory_order_relaxed);
        current->valueSum.fetch_add(nodeValue, std::memory_order_relaxed);
        
        // Flip value for parent node (minimax)
        nodeValue = -nodeValue;
        
        // Move to parent
        current = current->parent;
    }
}

std::pair<std::vector<float>, float> ParallelMCTS::evaluateState(const core::IGameState& state) {
    // Check if state is terminal
    if (state.isTerminal()) {
        // Create uniform policy for terminal states
        std::vector<float> uniformPolicy(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
        float value = convertToValue(state.getGameResult(), state.getCurrentPlayer());
        return {uniformPolicy, value};
    }
    
    // Use neural network for evaluation
    if (nn_) {
        try {
            return nn_->predict(state);
        } catch (const std::exception& e) {
            if (debugMode_) {
                std::cerr << "Neural network error: " << e.what() << std::endl;
            }
        }
    }
    
    // Fallback to random policy if neural network fails
    std::vector<float> randomPolicy(state.getActionSpaceSize(), 1.0f / state.getActionSpaceSize());
    float randomValue = 0.0f;  // Neutral value
    
    return {randomPolicy, randomValue};
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
        return -1;
    }
    
    // In training mode with temperature > 0, use stochastic selection
    if (isTraining && temperature > 0.0f) {
        // Get visit count distribution
        std::vector<float> distribution = getActionProbabilities(temperature);
        
        // Sample action from distribution
        if (deterministicMode_) {
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
        // In evaluation mode or temperature=0, pick move with highest visit count
        return rootNode_->getBestAction();
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
        rootNode_ = std::make_unique<MCTSNode>(rootState_.get(), nullptr, 0.0f);
    }
}

void ParallelMCTS::addDirichletNoise(float alpha, float epsilon) {
    if (!rootNode_->isExpanded) {
        // Expand root node first
        expandNode(rootNode_.get(), *rootState_);
    }
    
    if (rootNode_->children.empty()) {
        return;
    }
    
    // Get Dirichlet alpha based on game type
    float dirichletAlpha = alpha > 0.0f ? alpha : getDirichletAlpha();
    
    // Generate Dirichlet noise
    std::vector<float> noise(rootNode_->children.size());
    std::gamma_distribution<float> gamma(dirichletAlpha, 1.0f);
    float noiseSum = 0.0f;
    
    for (size_t i = 0; i < noise.size(); ++i) {
        noise[i] = gamma(rng_);
        noiseSum += noise[i];
    }
    
    // Normalize noise
    for (size_t i = 0; i < noise.size(); ++i) {
        noise[i] /= noiseSum;
    }
    
    // Mix noise with prior probabilities
    for (size_t i = 0; i < rootNode_->children.size(); ++i) {
        float oldPrior = rootNode_->children[i]->prior;
        float newPrior = (1.0f - epsilon) * oldPrior + epsilon * noise[i];
        rootNode_->children[i]->prior = newPrior;
    }
}

void ParallelMCTS::setNumThreads(int numThreads) {
    // Wait for any ongoing search to complete
    while (searchInProgress_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Create new thread pool
    threadPool_ = std::make_unique<ThreadPool>(numThreads);
}

void ParallelMCTS::setNeuralNetwork(NeuralNetwork* nn) {
    nn_ = nn;
}

void ParallelMCTS::setTranspositionTable(TranspositionTable* tt) {
    tt_ = tt;
}

void ParallelMCTS::setDeterministicMode(bool enable) {
    deterministicMode_ = enable;
    
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
    ss << "  Child visits:" << std::endl;
    
    // Sort children by visit count
    std::vector<std::pair<int, int>> visitCounts;
    for (size_t i = 0; i < rootNode_->children.size(); ++i) {
        visitCounts.emplace_back(rootNode_->actions[i], rootNode_->children[i]->visitCount.load());
    }
    
    std::sort(visitCounts.begin(), visitCounts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Display top 10 moves
    int numToShow = std::min(10, static_cast<int>(visitCounts.size()));
    for (int i = 0; i < numToShow; ++i) {
        int action = visitCounts[i].first;
        int visits = visitCounts[i].second;
        float value = 0.0f;
        float prior = 0.0f;
        
        // Find child
        for (size_t j = 0; j < rootNode_->actions.size(); ++j) {
            if (rootNode_->actions[j] == action) {
                value = rootNode_->children[j]->getValue();
                prior = rootNode_->children[j]->prior;
                break;
            }
        }
        
        float visitPercent = 100.0f * visits / rootNode_->visitCount.load();
        ss << "    Action " << std::setw(3) << action 
           << ": visits=" << std::setw(5) << visits 
           << " (" << std::fixed << std::setprecision(1) << visitPercent << "%)"
           << ", value=" << std::fixed << std::setprecision(3) << value
           << ", prior=" << std::fixed << std::setprecision(3) << prior
           << std::endl;
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
        
        std::vector<std::pair<int, int>> visitCounts;
        for (size_t i = 0; i < child->children.size(); ++i) {
            visitCounts.emplace_back(child->actions[i], child->children[i]->visitCount.load());
        }
        
        std::sort(visitCounts.begin(), visitCounts.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
        
        int numToShow = std::min(5, static_cast<int>(visitCounts.size()));
        for (int i = 0; i < numToShow; ++i) {
            int gcAction = visitCounts[i].first;
            int visits = visitCounts[i].second;
            float value = 0.0f;
            float prior = 0.0f;
            
            // Find grandchild
            for (size_t j = 0; j < child->actions.size(); ++j) {
                if (child->actions[j] == gcAction) {
                    value = child->children[j]->getValue();
                    prior = child->children[j]->prior;
                    break;
                }
            }
            
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
    
    // Recursively count nodes
    std::function<size_t(const MCTSNode*)> countNodeSize = 
        [&countNodeSize](const MCTSNode* node) -> size_t {
            if (!node) return 0;
            
            size_t size = sizeof(*node);
            
            // Add children sizes
            for (const auto& child : node->children) {
                size += countNodeSize(child.get());
            }
            
            // Add actions vector size
            size += node->actions.size() * sizeof(int);
            
            return size;
        };
    
    totalSize += countNodeSize(rootNode_.get());
    
    // Add thread pool size
    totalSize += sizeof(*threadPool_);
    totalSize += threadPool_->size() * sizeof(std::thread);
    
    return totalSize;
}

void ParallelMCTS::releaseMemory() {
    // Prune all nodes except the current root and its direct children
    pruneAllExcept(rootNode_.get());
}

void ParallelMCTS::pruneAllExcept(MCTSNode* nodeToKeep) {
    // This is a no-op as memory management is handled by unique_ptr
    // and the tree structure automatically cleans up unused branches
    // when updateWithMove is called.
    //
    // This method is included for API completeness, but it doesn't
    // need to do anything in this implementation.
}

} // namespace mcts
} // namespace alphazero