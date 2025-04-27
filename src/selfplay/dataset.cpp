// src/selfplay/dataset.cpp
#include "alphazero/selfplay/dataset.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "alphazero/core/game_factory.h"

namespace alphazero {
namespace selfplay {

using json = nlohmann::json;

std::string TrainingExample::toJson() const {
    json j;
    
    // Serialize state tensor
    json state_json = json::array();
    for (const auto& plane : state) {
        json plane_json = json::array();
        for (const auto& row : plane) {
            plane_json.push_back(row);
        }
        state_json.push_back(plane_json);
    }
    j["state"] = state_json;
    
    j["policy"] = policy;
    j["value"] = value;
    
    return j.dump();
}

TrainingExample TrainingExample::fromJson(const std::string& jsonStr) {
    json j = json::parse(jsonStr);
    
    TrainingExample example;
    
    // Deserialize state tensor
    const auto& state_json = j["state"];
    example.state.resize(state_json.size());
    for (size_t i = 0; i < state_json.size(); ++i) {
        const auto& plane_json = state_json[i];
        example.state[i].resize(plane_json.size());
        for (size_t r = 0; r < plane_json.size(); ++r) {
            example.state[i][r] = plane_json[r].get<std::vector<float>>();
        }
    }
    
    example.policy = j["policy"].get<std::vector<float>>();
    example.value = j["value"];
    
    return example;
}

Dataset::Dataset() : rng_(std::random_device{}()) {
}

void Dataset::addGameRecord(const GameRecord& record, bool useEnhancedFeatures) {
    gameRecords_.push_back(record);
}

void Dataset::extractExamples(bool includeAugmentations) {
    examples_.clear();
    
    for (const auto& record : gameRecords_) {
        auto [gameType, boardSize, useVariantRules] = record.getMetadata();
        
        // Create a game state for tensor conversion
        auto state = core::createGameState(gameType, boardSize, useVariantRules);
        
        // Extract examples from each position in the game
        const auto& moves = record.getMoves();
        for (size_t i = 0; i < moves.size(); ++i) {
            if (i > 0) {
                // Make the previous move to update the state
                state->makeMove(moves[i-1].action);
            }
            
            // Create training example
            TrainingExample example;
            example.state = state->getEnhancedTensorRepresentation();
            example.policy = moves[i].policy;
            
            // Value target: final game result from this player's perspective
            float gameValue = 0.0f;
            if (record.getResult() == core::GameResult::WIN_PLAYER1) {
                gameValue = 1.0f;
            } else if (record.getResult() == core::GameResult::WIN_PLAYER2) {
                gameValue = -1.0f;
            }
            
            // Flip value based on current player
            if (state->getCurrentPlayer() == 2) {
                gameValue = -gameValue;
            }
            
            example.value = gameValue;
            
            // Add the example
            examples_.push_back(example);
            
            // Add augmented examples if requested
            if (includeAugmentations) {
                auto augmented = augmentExample(example, gameType);
                examples_.insert(examples_.end(), augmented.begin(), augmented.end());
            }
        }
    }
    
    // Shuffle examples
    shuffle();
}

size_t Dataset::size() const {
    return examples_.size();
}

std::tuple<std::vector<std::vector<std::vector<std::vector<float>>>>,
           std::vector<std::vector<float>>,
           std::vector<float>> Dataset::getBatch(size_t batchSize) const {
    
    batchSize = std::min(batchSize, examples_.size());
    
    // Prepare batch containers
    std::vector<std::vector<std::vector<std::vector<float>>>> states(batchSize);
    std::vector<std::vector<float>> policies(batchSize);
    std::vector<float> values(batchSize);
    
    // Random indices for batch
    std::vector<size_t> indices(examples_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    
    // Fill batch
    for (size_t i = 0; i < batchSize; ++i) {
        size_t idx = indices[i];
        states[i] = examples_[idx].state;
        policies[i] = examples_[idx].policy;
        values[i] = examples_[idx].value;
    }
    
    return {states, policies, values};
}

void Dataset::shuffle() {
    std::shuffle(examples_.begin(), examples_.end(), rng_);
}

bool Dataset::saveToFile(const std::string& filename) const {
    try {
        json j;
        json examples_json = json::array();
        
        for (const auto& example : examples_) {
            json example_json;
            
            // Serialize state tensor
            json state_json = json::array();
            for (const auto& plane : example.state) {
                json plane_json = json::array();
                for (const auto& row : plane) {
                    plane_json.push_back(row);
                }
                state_json.push_back(plane_json);
            }
            example_json["state"] = state_json;
            
            example_json["policy"] = example.policy;
            example_json["value"] = example.value;
            
            examples_json.push_back(example_json);
        }
        
        j["examples"] = examples_json;
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << j.dump();
        return true;
    } catch (...) {
        return false;
    }
}

bool Dataset::loadFromFile(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        json j;
        file >> j;
        
        examples_.clear();
        
        for (const auto& example_json : j["examples"]) {
            TrainingExample example;
            
            // Deserialize state tensor
            const auto& state_json = example_json["state"];
            example.state.resize(state_json.size());
            for (size_t i = 0; i < state_json.size(); ++i) {
                const auto& plane_json = state_json[i];
                example.state[i].resize(plane_json.size());
                for (size_t r = 0; r < plane_json.size(); ++r) {
                    example.state[i][r] = plane_json[r].get<std::vector<float>>();
                }
            }
            
            example.policy = example_json["policy"].get<std::vector<float>>();
            example.value = example_json["value"];
            
            examples_.push_back(example);
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<TrainingExample> Dataset::getRandomSubset(size_t count) const {
    count = std::min(count, examples_.size());
    
    std::vector<size_t> indices(examples_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    
    std::vector<TrainingExample> subset;
    subset.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        subset.push_back(examples_[indices[i]]);
    }
    
    return subset;
}

std::vector<TrainingExample> Dataset::augmentExample(
    const TrainingExample& example, core::GameType gameType) const {
    
    std::vector<TrainingExample> augmented;
    
    // For Chess, no augmentation
    if (gameType == core::GameType::CHESS) {
        return augmented; // Return empty vector
    }
    
    // For Gomoku and Go, apply rotations and reflections
    // We can do up to 8 augmentations: 4 rotations x 2 reflections
    
    const size_t numPlanes = example.state.size();
    const size_t boardSize = example.state[0].size();
    
    // Original example already in the dataset, so start with 7 augmentations
    augmented.reserve(7);
    
    // 90-degree rotation
    TrainingExample rot90 = example;
    rot90.state.resize(numPlanes);
    for (size_t p = 0; p < numPlanes; ++p) {
        rot90.state[p].resize(boardSize, std::vector<float>(boardSize, 0.0f));
        for (size_t i = 0; i < boardSize; ++i) {
            for (size_t j = 0; j < boardSize; ++j) {
                rot90.state[p][j][boardSize-1-i] = example.state[p][i][j];
            }
        }
    }
    // Also need to rotate the policy
    rot90.policy.resize(example.policy.size());
    for (size_t i = 0; i < boardSize; ++i) {
        for (size_t j = 0; j < boardSize; ++j) {
            size_t oldIdx = i * boardSize + j;
            size_t newIdx = j * boardSize + (boardSize-1-i);
            if (oldIdx < example.policy.size() && newIdx < rot90.policy.size()) {
                rot90.policy[newIdx] = example.policy[oldIdx];
            }
        }
    }
    augmented.push_back(rot90);
    
    // 180-degree rotation
    TrainingExample rot180 = example;
    rot180.state.resize(numPlanes);
    for (size_t p = 0; p < numPlanes; ++p) {
        rot180.state[p].resize(boardSize, std::vector<float>(boardSize, 0.0f));
        for (size_t i = 0; i < boardSize; ++i) {
            for (size_t j = 0; j < boardSize; ++j) {
                rot180.state[p][boardSize-1-i][boardSize-1-j] = example.state[p][i][j];
            }
        }
    }
    // Rotate policy
    rot180.policy.resize(example.policy.size());
    for (size_t i = 0; i < boardSize; ++i) {
        for (size_t j = 0; j < boardSize; ++j) {
            size_t oldIdx = i * boardSize + j;
            size_t newIdx = (boardSize-1-i) * boardSize + (boardSize-1-j);
            if (oldIdx < example.policy.size() && newIdx < rot180.policy.size()) {
                rot180.policy[newIdx] = example.policy[oldIdx];
            }
        }
    }
    augmented.push_back(rot180);
    
    // 270-degree rotation
    TrainingExample rot270 = example;
    rot270.state.resize(numPlanes);
    for (size_t p = 0; p < numPlanes; ++p) {
        rot270.state[p].resize(boardSize, std::vector<float>(boardSize, 0.0f));
        for (size_t i = 0; i < boardSize; ++i) {
            for (size_t j = 0; j < boardSize; ++j) {
                rot270.state[p][boardSize-1-j][i] = example.state[p][i][j];
            }
        }
    }
    // Rotate policy
    rot270.policy.resize(example.policy.size());
    for (size_t i = 0; i < boardSize; ++i) {
        for (size_t j = 0; j < boardSize; ++j) {
            size_t oldIdx = i * boardSize + j;
            size_t newIdx = (boardSize-1-j) * boardSize + i;
            if (oldIdx < example.policy.size() && newIdx < rot270.policy.size()) {
                rot270.policy[newIdx] = example.policy[oldIdx];
            }
        }
    }
    augmented.push_back(rot270);
    
    // Horizontal flip
    TrainingExample flipH = example;
    flipH.state.resize(numPlanes);
    for (size_t p = 0; p < numPlanes; ++p) {
        flipH.state[p].resize(boardSize, std::vector<float>(boardSize, 0.0f));
        for (size_t i = 0; i < boardSize; ++i) {
            for (size_t j = 0; j < boardSize; ++j) {
                flipH.state[p][i][boardSize-1-j] = example.state[p][i][j];
            }
        }
    }
    // Flip policy
    flipH.policy.resize(example.policy.size());
    for (size_t i = 0; i < boardSize; ++i) {
        for (size_t j = 0; j < boardSize; ++j) {
            size_t oldIdx = i * boardSize + j;
            size_t newIdx = i * boardSize + (boardSize-1-j);
            if (oldIdx < example.policy.size() && newIdx < flipH.policy.size()) {
                flipH.policy[newIdx] = example.policy[oldIdx];
            }
        }
    }
    augmented.push_back(flipH);
    
    // Add the other flips and rotations
    // Horizontal flip + 90-degree rotation
    TrainingExample flipH_rot90 = rot90;
    flipH_rot90.state.resize(numPlanes);
    for (size_t p = 0; p < numPlanes; ++p) {
        flipH_rot90.state[p].resize(boardSize, std::vector<float>(boardSize, 0.0f));
        for (size_t i = 0; i < boardSize; ++i) {
            for (size_t j = 0; j < boardSize; ++j) {
                flipH_rot90.state[p][i][boardSize-1-j] = rot90.state[p][i][j];
            }
        }
    }
    // Flip policy
    flipH_rot90.policy.resize(rot90.policy.size());
    for (size_t i = 0; i < boardSize; ++i) {
        for (size_t j = 0; j < boardSize; ++j) {
            size_t oldIdx = i * boardSize + j;
            size_t newIdx = i * boardSize + (boardSize-1-j);
            if (oldIdx < rot90.policy.size() && newIdx < flipH_rot90.policy.size()) {
                flipH_rot90.policy[newIdx] = rot90.policy[oldIdx];
            }
        }
    }
    augmented.push_back(flipH_rot90);
    
    // Horizontal flip + 180-degree rotation
    TrainingExample flipH_rot180 = rot180;
    flipH_rot180.state.resize(numPlanes);
    for (size_t p = 0; p < numPlanes; ++p) {
        flipH_rot180.state[p].resize(boardSize, std::vector<float>(boardSize, 0.0f));
        for (size_t i = 0; i < boardSize; ++i) {
            for (size_t j = 0; j < boardSize; ++j) {
                flipH_rot180.state[p][i][boardSize-1-j] = rot180.state[p][i][j];
            }
        }
    }
    // Flip policy
    flipH_rot180.policy.resize(rot180.policy.size());
    for (size_t i = 0; i < boardSize; ++i) {
        for (size_t j = 0; j < boardSize; ++j) {
            size_t oldIdx = i * boardSize + j;
            size_t newIdx = i * boardSize + (boardSize-1-j);
            if (oldIdx < rot180.policy.size() && newIdx < flipH_rot180.policy.size()) {
                flipH_rot180.policy[newIdx] = rot180.policy[oldIdx];
            }
        }
    }
    augmented.push_back(flipH_rot180);
    
    // Horizontal flip + 270-degree rotation
    TrainingExample flipH_rot270 = rot270;
    flipH_rot270.state.resize(numPlanes);
    for (size_t p = 0; p < numPlanes; ++p) {
        flipH_rot270.state[p].resize(boardSize, std::vector<float>(boardSize, 0.0f));
        for (size_t i = 0; i < boardSize; ++i) {
            for (size_t j = 0; j < boardSize; ++j) {
                flipH_rot270.state[p][i][boardSize-1-j] = rot270.state[p][i][j];
            }
        }
    }
    // Flip policy
    flipH_rot270.policy.resize(rot270.policy.size());
    for (size_t i = 0; i < boardSize; ++i) {
        for (size_t j = 0; j < boardSize; ++j) {
            size_t oldIdx = i * boardSize + j;
            size_t newIdx = i * boardSize + (boardSize-1-j);
            if (oldIdx < rot270.policy.size() && newIdx < flipH_rot270.policy.size()) {
                flipH_rot270.policy[newIdx] = rot270.policy[oldIdx];
            }
        }
    }
    augmented.push_back(flipH_rot270);
    
    return augmented;
}

} // namespace selfplay
} // namespace alphazero