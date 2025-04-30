// mcts_node.h
#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include "alphazero/core/igamestate.h"

namespace alphazero {
namespace mcts {

using core::GameType;
using core::GameResult;

/**
 * @brief Node in the Monte Carlo Tree Search tree
 * 
 * This class represents a node in the MCTS tree, storing statistics
 * and children for the search algorithm.
 */
class MCTSNode {
public:
    /**
     * @brief Constructor
     * 
     * @param state Game state this node represents (not owned)
     * @param parent Parent node (nullptr for root)
     * @param prior Prior probability from policy network
     * @param action Action that led to this node (-1 for root)
     */
    MCTSNode(const core::IGameState* state, MCTSNode* parent = nullptr, 
             float prior = 0.0f, int action = -1);
    
    /**
     * @brief Destructor
     */
    ~MCTSNode();
    
    // Non-copyable but movable
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) noexcept = default;
    MCTSNode& operator=(MCTSNode&&) noexcept = default;
    
    // Core MCTS statistics - atomic for thread safety
    std::atomic<int> visitCount{0};       // Number of visits to this node
    std::atomic<float> valueSum{0.0f};    // Sum of values from this node
    std::atomic<int> virtualLoss{0};      // Virtual loss for parallel search
    float prior;                           // Prior probability from policy network
    int action;                            // Action that led to this node (-1 for root)
    
    // Tree structure
    MCTSNode* parent;                                    // Parent node
    std::vector<std::unique_ptr<MCTSNode>> children;     // Child nodes (owned)
    std::vector<int> actions;                            // Actions leading to each child
    std::mutex expansionMutex;                           // Mutex for thread-safe expansion
    
    // State information
    uint64_t stateHash;                   // Zobrist hash of the state
    bool isTerminal;                       // Whether node represents a terminal state
    GameResult gameResult;                 // Game outcome if terminal
    GameType gameType;                     // Type of game this node represents
    
    // Expansion status
    bool isExpanded;                       // Whether node has been expanded
    
    /**
     * @brief Get the value estimate for this node
     * 
     * @return The value estimate [-1,1]
     */
    float getValue() const {
        int visits = visitCount.load(std::memory_order_relaxed);
        if (visits == 0) return 0.0f;
        
        return valueSum.load(std::memory_order_relaxed) / visits;
    }
    
    /**
     * @brief Get the adjusted value that accounts for virtual loss
     * 
     * @return The adjusted value estimate [-1,1]
     */
    float getAdjustedValue() const;
    
    /**
     * @brief Convert terminal game result to value
     * 
     * @param currentPlayer Current player perspective
     * @return Value [-1,1] from perspective player's view
     */
    float getTerminalValue(int currentPlayer) const;
    
    /**
     * @brief Get the UCB score for this node
     * 
     * @param cPuct Exploration constant
     * @param currentPlayer Current player perspective
     * @param fpuReduction First play urgency reduction
     * @param parentVisits Parent visit count
     * @return The UCB score
     */
    float getUcbScore(float cPuct, int currentPlayer, float fpuReduction = 0.0f, int parentVisits = 0) const;
    
    /**
     * @brief Get the PUCT score for this node
     * 
     * @param cPuct Exploration constant
     * @param currentPlayer Current player perspective
     * @param fpuReduction First play urgency reduction
     * @param parentVisits Parent visit count
     * @return The PUCT score
     */
    float getPuctScore(float cPuct, int currentPlayer, float fpuReduction = 0.0f, int parentVisits = 0) const;
    
    /**
     * @brief Get the Progressive Bias score for this node
     * 
     * @param cPuct Exploration constant
     * @param currentPlayer Current player perspective
     * @param parentVisits Parent visit count
     * @return The Progressive Bias score
     */
    float getProgressiveBiasScore(float cPuct, int currentPlayer, int parentVisits = 0) const;
    
    /**
     * @brief Add virtual loss for parallel search
     * 
     * @param virtualLossAmount Amount of virtual loss to add
     */
    void addVirtualLoss(int virtualLossAmount);
    
    /**
     * @brief Remove virtual loss after search completes
     * 
     * @param virtualLossAmount Amount of virtual loss to remove
     */
    void removeVirtualLoss(int virtualLossAmount);
    
    /**
     * @brief Get child node at specific index
     * 
     * @param actionIndex Index of the action
     * @return Pointer to child node (nullptr if not exists)
     */
    MCTSNode* getChild(int actionIndex) const;
    
    /**
     * @brief Get child node for a specific action
     * 
     * @param action The action to find the child for
     * @return Pointer to child node (nullptr if not exists)
     */
    MCTSNode* getChildForAction(int action) const;
    
    /**
     * @brief Get index of action in children array
     * 
     * @param action The action to find
     * @return Index of action (-1 if not found)
     */
    int getActionIndex(int action) const;
    
    /**
     * @brief Add a child node
     * 
     * @param action Action leading to child
     * @param prior Prior probability for this action
     * @param child Child node to add
     */
    void addChild(int action, float prior, std::unique_ptr<MCTSNode> child);
    
    /**
     * @brief Check if node has children
     * 
     * @return true if node has children, false otherwise
     */
    bool hasChildren() const { return !children.empty(); }
    
    /**
     * @brief Get the number of children
     * 
     * @return The number of child nodes
     */
    size_t getChildCount() const { return children.size(); }
    
    /**
     * @brief Get the best action based on visit counts
     * 
     * @return The action with highest visit count
     */
    int getBestAction() const;
    
    /**
     * @brief Get all actions with the highest visit count
     * 
     * @return Vector of actions with the highest visit count
     */
    std::vector<int> getBestActions() const;
    
    /**
     * @brief Get the best child based on visit counts
     * 
     * @return Pointer to best child (nullptr if no children)
     */
    MCTSNode* getBestChild() const;
    
    /**
     * @brief Get action visit count distribution
     * 
     * @param temperature Temperature parameter for exploration
     * @return Vector of probabilities for each action
     */
    std::vector<float> getVisitCountDistribution(float temperature = 1.0f) const;
    
    /**
     * @brief Get debug string representation
     * 
     * @param maxDepth Maximum depth to print
     * @return String representation of the node and children
     */
    std::string toString(int maxDepth = 1) const;
    
    /**
     * @brief Print tree to standard output
     * 
     * @param maxDepth Maximum depth to print
     */
    void printTree(int maxDepth = 1) const;
    
    /**
     * @brief Check if node has unexpanded children
     * 
     * @return true if node has at least one unexpanded child
     */
    bool hasUnexpandedChildren() const;
    
    /**
     * @brief Get the total size of the tree rooted at this node
     * 
     * @return The total number of nodes in the tree
     */
    size_t getTreeSize() const;
    
    /**
     * @brief Get the estimated memory usage of the tree rooted at this node
     * 
     * @return The estimated memory usage in bytes
     */
    size_t getTreeMemoryUsage() const;
    
    /**
     * @brief Prune the tree to reduce memory usage
     * 
     * @param visitThreshold Minimum visit count to keep a node
     * @return Number of nodes pruned
     */
    size_t pruneTree(int visitThreshold);
    
private:
    // Helper methods
    std::string indentString(int depth) const;
    float convertToValue(GameResult result, int perspectivePlayer) const;
    
    // Node metadata for memory optimization
    static constexpr size_t ESTIMATED_NODE_SIZE = 128; // Bytes per node (estimated)
};

} // namespace mcts
} // namespace alphazero

#endif // MCTS_NODE_H