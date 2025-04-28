// mcts_node.cpp
#include "alphazero/mcts/mcts_node.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

namespace alphazero {
namespace mcts {

MCTSNode::MCTSNode(const core::IGameState* state, MCTSNode* parent, float prior)
    : visitCount(0), 
      valueSum(0.0f), 
      prior(prior), 
      parent(parent),
      children(),
      actions(),
      expansionMutex(),
      stateHash(state ? state->getHash() : 0),
      isTerminal(state ? state->isTerminal() : false),
      gameResult(state ? state->getGameResult() : GameResult::ONGOING),
      gameType(state ? state->getGameType() : GameType::GOMOKU),
      isExpanded(false) {
}

MCTSNode::~MCTSNode() {
    // Children are automatically cleaned up by unique_ptr
}

float MCTSNode::getTerminalValue(int currentPlayer) const {
    return convertToValue(gameResult, currentPlayer);
}

float MCTSNode::getUcbScore(float cPuct, int currentPlayer, float fpuReduction) const {
    // If no visits, return max score to ensure exploration
    if (visitCount.load() == 0) {
        return std::numeric_limits<float>::max();
    }
    
    // PUCT formula: Q + U
    // Q = value estimate
    // U = cPuct * P * sqrt(N_parent) / (1 + N)
    
    // Value term (Q)
    float qValue = getValue();
    
    // For opponent nodes, negate the value to get the relative value
    if (parent && parent->parent) {
        int nodePlayer = parent->parent->parent ? currentPlayer : 3 - currentPlayer;
        if (nodePlayer != currentPlayer) {
            qValue = -qValue;
        }
    }
    
    // If node is never visited or FPU reduction is applied
    if (parent && visitCount.load() == 0 && fpuReduction > 0.0f) {
        // FPU: Use parent's value with reduction to encourage exploration of unvisited nodes
        qValue = parent->getValue() - fpuReduction;
    }
    
    // Exploration term (U)
    float parentVisitSqrt = parent ? sqrtf(static_cast<float>(parent->visitCount.load())) : 0.0f;
    float explorationTerm = cPuct * prior * parentVisitSqrt / (1.0f + visitCount.load());
    
    return qValue + explorationTerm;
}

void MCTSNode::addVirtualLoss(int virtualLoss) {
    // Add negative virtual loss to discourage other threads from exploring this path
    visitCount.fetch_add(virtualLoss, std::memory_order_relaxed);
    
    // Since std::atomic<float> doesn't have fetch_sub, use atomic load-modify-store pattern
    float oldValue = valueSum.load(std::memory_order_relaxed);
    float newValue = oldValue - static_cast<float>(virtualLoss);
    while (!valueSum.compare_exchange_weak(oldValue, newValue, 
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
        newValue = oldValue - static_cast<float>(virtualLoss);
    }
}

void MCTSNode::removeVirtualLoss(int virtualLoss) {
    // Remove the virtual loss
    visitCount.fetch_sub(virtualLoss, std::memory_order_relaxed);
    
    // Since std::atomic<float> doesn't have fetch_add, use atomic load-modify-store pattern
    float oldValue = valueSum.load(std::memory_order_relaxed);
    float newValue = oldValue + static_cast<float>(virtualLoss);
    while (!valueSum.compare_exchange_weak(oldValue, newValue, 
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
        newValue = oldValue + static_cast<float>(virtualLoss);
    }
}

MCTSNode* MCTSNode::getChild(int actionIndex) const {
    if (actionIndex < 0 || actionIndex >= static_cast<int>(children.size())) {
        return nullptr;
    }
    return children[actionIndex].get();
}

int MCTSNode::getActionIndex(int action) const {
    for (size_t i = 0; i < actions.size(); ++i) {
        if (actions[i] == action) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MCTSNode::addChild(int action, float prior, std::unique_ptr<MCTSNode> child) {
    actions.push_back(action);
    children.push_back(std::move(child));
}

int MCTSNode::getBestAction() const {
    if (children.empty()) {
        return -1;
    }
    
    // Find child with highest visit count
    int bestActionIdx = 0;
    int maxVisits = children[0]->visitCount.load();
    
    for (size_t i = 1; i < children.size(); ++i) {
        int visits = children[i]->visitCount.load();
        if (visits > maxVisits) {
            maxVisits = visits;
            bestActionIdx = static_cast<int>(i);
        }
    }
    
    return actions[bestActionIdx];
}

std::vector<int> MCTSNode::getBestActions() const {
    if (children.empty()) {
        return {};
    }
    
    // Find max visit count
    int maxVisits = 0;
    for (const auto& child : children) {
        int visits = child->visitCount.load();
        maxVisits = std::max(maxVisits, visits);
    }
    
    // Collect all actions with this visit count
    std::vector<int> bestActions;
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]->visitCount.load() == maxVisits) {
            bestActions.push_back(actions[i]);
        }
    }
    
    return bestActions;
}

MCTSNode* MCTSNode::getBestChild() const {
    if (children.empty()) {
        return nullptr;
    }
    
    // Find child with highest visit count
    MCTSNode* bestChild = children[0].get();
    int maxVisits = bestChild->visitCount.load();
    
    for (size_t i = 1; i < children.size(); ++i) {
        MCTSNode* child = children[i].get();
        int visits = child->visitCount.load();
        if (visits > maxVisits) {
            maxVisits = visits;
            bestChild = child;
        }
    }
    
    return bestChild;
}

std::vector<float> MCTSNode::getVisitCountDistribution(float temperature) const {
    std::vector<float> distribution(actions.size(), 0.0f);
    
    if (children.empty()) {
        return distribution;
    }
    
    // Get total count to normalize
    float totalCount = 0.0f;
    std::vector<float> counts(children.size());
    
    for (size_t i = 0; i < children.size(); ++i) {
        // Apply temperature to visit counts
        float count = powf(static_cast<float>(children[i]->visitCount.load()), 1.0f / temperature);
        counts[i] = count;
        totalCount += count;
    }
    
    // Normalize to get probability distribution
    if (totalCount > 0.0f) {
        for (size_t i = 0; i < children.size(); ++i) {
            distribution[i] = counts[i] / totalCount;
        }
    } else {
        // If no visits, use uniform distribution
        float uniformProb = 1.0f / static_cast<float>(children.size());
        for (size_t i = 0; i < children.size(); ++i) {
            distribution[i] = uniformProb;
        }
    }
    
    return distribution;
}

std::string MCTSNode::toString(int maxDepth) const {
    std::stringstream ss;
    
    // Node statistics
    ss << "Node: V=" << visitCount.load() 
       << ", Q=" << std::fixed << std::setprecision(3) << getValue()
       << ", P=" << std::fixed << std::setprecision(3) << prior
       << (isTerminal ? " (Terminal)" : "");
    
    // Children information if depth allows
    if (maxDepth > 0 && !children.empty()) {
        ss << "\nChildren: " << children.size() << std::endl;
        
        for (size_t i = 0; i < children.size(); ++i) {
            ss << indentString(1) << "Action " << actions[i] << ": "
               << "V=" << children[i]->visitCount.load()
               << ", Q=" << std::fixed << std::setprecision(3) << children[i]->getValue()
               << ", P=" << std::fixed << std::setprecision(3) << children[i]->prior;
            
            // Recursively print children with reduced depth
            if (maxDepth > 1) {
                ss << "\n" << indentString(2) << children[i]->toString(maxDepth - 1);
            }
            
            if (i < children.size() - 1) {
                ss << std::endl;
            }
        }
    }
    
    return ss.str();
}

void MCTSNode::printTree(int maxDepth) const {
    std::cout << toString(maxDepth) << std::endl;
}

std::string MCTSNode::indentString(int depth) const {
    return std::string(depth * 4, ' ');
}

float MCTSNode::convertToValue(GameResult result, int perspectivePlayer) const {
    switch (result) {
        case GameResult::WIN_PLAYER1:
            return perspectivePlayer == 1 ? 1.0f : -1.0f;
        case GameResult::WIN_PLAYER2:
            return perspectivePlayer == 2 ? 1.0f : -1.0f;
        case GameResult::DRAW:
            return 0.0f;
        case GameResult::ONGOING:
        default:
            return 0.0f;
    }
}

} // namespace mcts
} // namespace alphazero