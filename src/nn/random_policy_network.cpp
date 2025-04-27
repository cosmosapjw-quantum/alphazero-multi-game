// random_policy_network.cpp
#include "alphazero/nn/random_policy_network.h"
#include <chrono>
#include <thread>

namespace alphazero {
namespace nn {

RandomPolicyNetwork::RandomPolicyNetwork(core::GameType gameType, int boardSize, unsigned int seed)
    : gameType_(gameType) {
    
    // Set board size
    boardSize_ = (boardSize > 0) ? boardSize : getDefaultBoardSize(gameType);
    
    // Initialize random number generator
    unsigned int actualSeed = seed;
    if (actualSeed == 0) {
        // Use current time as seed if none provided
        actualSeed = static_cast<unsigned int>(
            std::chrono::system_clock::now().time_since_epoch().count());
    }
    
    rng_.seed(actualSeed);
}

std::pair<std::vector<float>, float> RandomPolicyNetwork::predict(const core::IGameState& state) {
    // Generate random policy
    std::vector<float> policy = generateRandomPolicy(state);
    
    // Generate random value
    float value = generateRandomValue();
    
    // Simulate some computation time
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    
    return {policy, value};
}

void RandomPolicyNetwork::predictBatch(
    const std::vector<std::reference_wrapper<const core::IGameState>>& states,
    std::vector<std::vector<float>>& policies,
    std::vector<float>& values) {
    
    // Clear output vectors
    policies.clear();
    values.clear();
    
    // Process each state
    for (const auto& stateRef : states) {
        const core::IGameState& state = stateRef.get();
        
        // Generate random policy
        std::vector<float> policy = generateRandomPolicy(state);
        
        // Generate random value
        float value = generateRandomValue();
        
        // Add to output vectors
        policies.push_back(policy);
        values.push_back(value);
    }
    
    // Simulate some computation time
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

std::future<std::pair<std::vector<float>, float>> RandomPolicyNetwork::predictAsync(
    const core::IGameState& state) {
    
    // Use async to run prediction in a separate thread
    return std::async(std::launch::async, [this, state = state.clone()]() mutable {
        return this->predict(*state);
    });
}

int RandomPolicyNetwork::getDefaultBoardSize(core::GameType gameType) const {
    switch (gameType) {
        case core::GameType::GOMOKU:
            return 15;
        case core::GameType::CHESS:
            return 8;
        case core::GameType::GO:
            return 19;
        default:
            return 15;
    }
}

std::vector<float> RandomPolicyNetwork::generateRandomPolicy(const core::IGameState& state) const {
    // Get action space size
    int actionSpaceSize = state.getActionSpaceSize();
    
    // Create uniform random distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Generate random values
    std::vector<float> policy(actionSpaceSize);
    float sum = 0.0f;
    
    // Get legal moves
    std::vector<int> legalMoves = state.getLegalMoves();
    
    // Set to small value for all actions
    for (int i = 0; i < actionSpaceSize; ++i) {
        policy[i] = 0.001f;
    }
    
    // Set higher probabilities for legal moves
    for (int move : legalMoves) {
        if (move >= 0 && move < actionSpaceSize) {
            policy[move] = dist(const_cast<std::mt19937&>(rng_));
            sum += policy[move];
        }
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < actionSpaceSize; ++i) {
            policy[i] /= sum;
        }
    } else if (!legalMoves.empty()) {
        // If sum is zero but legal moves exist, use uniform
        float uniformProb = 1.0f / static_cast<float>(legalMoves.size());
        for (int move : legalMoves) {
            if (move >= 0 && move < actionSpaceSize) {
                policy[move] = uniformProb;
            }
        }
    }
    
    return policy;
}

float RandomPolicyNetwork::generateRandomValue() const {
    // Generate small random value to avoid strong bias
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    return dist(const_cast<std::mt19937&>(rng_));
}

} // namespace nn
} // namespace alphazero