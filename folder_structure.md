# AlphaZero Multi-Game AI Engine: Folder and File Structure

Based on the PRD, here's a comprehensive folder structure for the AlphaZero Multi-Game AI Engine project:

```
alphazero-multi-game/
├── CMakeLists.txt                 # Main CMake configuration
├── LICENSE                        # Dual license (AGPL-3.0/Commercial)
├── README.md                      # Project overview and documentation
├── .gitignore                     # Git ignore file
├── .github/                       # GitHub CI/CD configuration
│   └── workflows/
│       ├── build.yml              # Build workflow
│       ├── test.yml               # Test workflow
│       └── release.yml            # Release workflow
│
├── include/                       # Public header files
│   └── alphazero/
│       ├── core/                  # Core interfaces and abstractions
│       │   ├── igamestate.h       # Game state interface
│       │   ├── zobrist_hash.h     # Position hashing
│       │   └── game_factory.h     # Game factory function
│       │
│       ├── games/                 # Game-specific implementations
│       │   ├── gomoku/
│       │   │   ├── gomoku_state.h # Gomoku implementation
│       │   │   └── gomoku_rules.h # Renju rules implementation
│       │   ├── chess/
│       │   │   ├── chess_state.h  # Chess implementation
│       │   │   └── chess960.h     # Chess960 variant
│       │   └── go/
│       │       ├── go_state.h     # Go implementation
│       │       └── go_rules.h     # Japanese/Chinese rules
│       │
│       ├── mcts/                  # Monte Carlo Tree Search
│       │   ├── mcts_node.h        # MCTS node definition
│       │   ├── parallel_mcts.h    # Parallel search implementation
│       │   └── transposition_table.h # Position caching
│       │
│       ├── nn/                    # Neural network components
│       │   ├── neural_network.h   # Neural network interface
│       │   ├── torch_neural_network.h # PyTorch implementation
│       │   ├── batch_queue.h      # Batched inference queue
│       │   └── attack_defense_module.h # Pattern evaluation
│       │
│       ├── selfplay/              # Self-play system
│       │   ├── self_play_manager.h # Self-play orchestration
│       │   ├── game_record.h      # Game recording
│       │   └── dataset.h          # Training dataset
│       │
│       ├── elo/                   # ELO rating system
│       │   └── elo_tracker.h      # Rating calculation and tracking
│       │
│       ├── ui/                    # User interface components
│       │   ├── game_ui.h          # Game visualization
│       │   └── renderer.h         # Rendering utilities
│       │
│       ├── cli/                   # Command-line interface
│       │   ├── cli_interface.h    # CLI implementation
│       │   └── command_parser.h   # Command parsing
│       │
│       └── api/                   # REST API interface
│           ├── rest_api.h         # API endpoints
│           └── http_server.h      # Server implementation
│
├── src/                           # Implementation files
│   ├── core/                      # Core implementation
│   │   ├── igamestate.cpp
│   │   ├── zobrist_hash.cpp
│   │   └── game_factory.cpp
│   │
│   ├── games/                     # Game implementations
│   │   ├── gomoku/
│   │   │   ├── gomoku_state.cpp
│   │   │   └── gomoku_rules.cpp
│   │   ├── chess/
│   │   │   ├── chess_state.cpp
│   │   │   └── chess960.cpp
│   │   └── go/
│   │       ├── go_state.cpp
│   │       └── go_rules.cpp
│   │
│   ├── mcts/                      # MCTS implementation
│   │   ├── mcts_node.cpp
│   │   ├── parallel_mcts.cpp
│   │   └── transposition_table.cpp
│   │
│   ├── nn/                        # Neural network implementation
│   │   ├── neural_network.cpp
│   │   ├── torch_neural_network.cpp
│   │   ├── batch_queue.cpp
│   │   └── attack_defense_module.cpp
│   │
│   ├── selfplay/                  # Self-play implementation
│   │   ├── self_play_manager.cpp
│   │   ├── game_record.cpp
│   │   └── dataset.cpp
│   │
│   ├── elo/                       # ELO implementation
│   │   └── elo_tracker.cpp
│   │
│   ├── ui/                        # UI implementation
│   │   ├── game_ui.cpp
│   │   └── renderer.cpp
│   │
│   ├── cli/                       # CLI implementation
│   │   ├── cli_interface.cpp
│   │   ├── command_parser.cpp
│   │   └── cli_main.cpp           # CLI executable
│   │
│   ├── api/                       # API implementation
│   │   ├── rest_api.cpp
│   │   ├── http_server.cpp
│   │   └── server_main.cpp        # Server executable
│   │
│   ├── gui/                       # GUI application
│   │   └── gui_main.cpp           # GUI executable
│   │
│   └── pybind/                    # Python bindings
│       └── python_bindings.cpp    # PyBind11 implementation
│
├── tests/                         # Test files
│   ├── CMakeLists.txt             # Test build configuration
│   ├── core/                      # Core tests
│   │   ├── igamestate_test.cpp
│   │   ├── zobrist_test.cpp
│   │   └── game_factory_test.cpp
│   │
│   ├── games/                     # Game-specific tests
│   │   ├── gomoku/
│   │   │   ├── gomoku_state_test.cpp
│   │   │   └── gomoku_rules_test.cpp
│   │   ├── chess/
│   │   │   ├── chess_state_test.cpp
│   │   │   └── chess960_test.cpp
│   │   └── go/
│   │       ├── go_state_test.cpp
│   │       └── go_rules_test.cpp
│   │
│   ├── mcts/                      # MCTS tests
│   │   ├── mcts_node_test.cpp
│   │   ├── parallel_mcts_test.cpp
│   │   └── transposition_table_test.cpp
│   │
│   ├── nn/                        # Neural network tests
│   │   ├── neural_network_test.cpp
│   │   ├── torch_neural_network_test.cpp
│   │   ├── batch_queue_test.cpp
│   │   └── attack_defense_module_test.cpp
│   │
│   ├── selfplay/                  # Self-play tests
│   │   ├── self_play_manager_test.cpp
│   │   ├── game_record_test.cpp
│   │   └── dataset_test.cpp
│   │
│   ├── elo/                       # ELO rating tests
│   │   └── elo_tracker_test.cpp
│   │
│   ├── ui/                        # UI tests
│   │   ├── game_ui_test.cpp
│   │   └── renderer_test.cpp
│   │
│   ├── cli/                       # CLI tests
│   │   ├── cli_interface_test.cpp
│   │   └── command_parser_test.cpp
│   │
│   ├── api/                       # API tests
│   │   ├── rest_api_test.cpp
│   │   └── http_server_test.cpp
│   │
│   ├── integration/               # Integration tests
│   │   ├── gomoku_integration_test.cpp
│   │   ├── chess_integration_test.cpp
│   │   ├── go_integration_test.cpp
│   │   └── multi_game_test.cpp
│   │
│   └── performance/               # Performance tests
│       ├── gomoku_performance_test.cpp
│       ├── chess_performance_test.cpp
│       └── go_performance_test.cpp
│
├── python/                        # Python code
│   ├── alphazero/                 # Python package
│   │   ├── __init__.py
│   │   ├── models/                # Neural network models
│   │   │   ├── __init__.py
│   │   │   └── ddw_randwire.py    # DDW-RandWire-ResNet model
│   │   ├── training/              # Training code
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py         # Dataset handling
│   │   │   ├── loss.py            # Training loss functions
│   │   │   └── scheduler.py       # Learning rate scheduling
│   │   └── utils/                 # Utilities
│   │       ├── __init__.py
│   │       ├── visualization.py   # Result visualization
│   │       └── elo.py             # ELO calculation
│   │
│   ├── scripts/                   # Training scripts
│   │   ├── train.py               # Main training script
│   │   ├── self_play.py           # Self-play generation
│   │   └── evaluate.py            # Model evaluation
│   │
│   ├── examples/                  # Example scripts
│   │   ├── play_gomoku.py
│   │   ├── play_chess.py
│   │   └── play_go.py
│   │
│   ├── tests/                     # Python tests
│   │   ├── test_models.py
│   │   ├── test_training.py
│   │   └── test_binding.py
│   │
│   ├── setup.py                   # Package setup
│   └── requirements.txt           # Python dependencies
│
├── config/                        # Configuration files
│   ├── defaults/                  # Default configurations
│   │   ├── gomoku_config.json     # Gomoku settings
│   │   ├── chess_config.json      # Chess settings
│   │   └── go_config.json         # Go settings
│   │
│   ├── nn/                        # Neural network configs
│   │   ├── gomoku_nn_config.json
│   │   ├── chess_nn_config.json
│   │   └── go_nn_config.json
│   │
│   ├── mcts/                      # MCTS configs
│   │   ├── gomoku_mcts_config.json
│   │   ├── chess_mcts_config.json
│   │   └── go_mcts_config.json
│   │
│   └── self_play/                 # Self-play configs
│       ├── gomoku_selfplay_config.json
│       ├── chess_selfplay_config.json
│       └── go_selfplay_config.json
│
├── models/                        # Pre-trained models
│   ├── gomoku/                    # Gomoku models
│   │   ├── baseline.pt            # Baseline model
│   │   └── 2000_elo.pt            # 2000 ELO model
│   │
│   ├── chess/                     # Chess models
│   │   ├── baseline.pt            # Baseline model
│   │   └── 2200_elo.pt            # 2200 ELO model
│   │
│   └── go/                        # Go models
│       ├── 9x9/                   # 9x9 board models
│       │   ├── baseline.pt
│       │   └── 2000_elo.pt
│       └── 19x19/                 # 19x19 board models
│           ├── baseline.pt
│           └── 2000_elo.pt
│
├── docs/                          # Documentation
│   ├── index.md                   # Main documentation
│   ├── architecture/              # Architecture docs
│   │   ├── overview.md            # System overview
│   │   ├── game_abstraction.md    # Game abstraction layer
│   │   ├── mcts.md                # MCTS implementation
│   │   └── neural_network.md      # Neural network architecture
│   │
│   ├── games/                     # Game-specific docs
│   │   ├── gomoku.md              # Gomoku implementation
│   │   ├── chess.md               # Chess implementation
│   │   └── go.md                  # Go implementation
│   │
│   ├── training/                  # Training docs
│   │   ├── self_play.md           # Self-play process
│   │   ├── training.md            # Training process
│   │   └── evaluation.md          # Evaluation process
│   │
│   ├── api/                       # API docs
│   │   ├── rest_api.md            # REST API
│   │   └── python_api.md          # Python API
│   │
│   ├── user_guide/                # User docs
│   │   ├── installation.md        # Installation guide
│   │   ├── playing.md             # Playing guide
│   │   └── training.md            # Training guide
│   │
│   └── developer_guide/           # Developer docs
│       ├── contributing.md        # Contributing guide
│       ├── coding_style.md        # Coding standards
│       └── testing.md             # Testing guide
│
├── examples/                      # Example code
│   ├── CMakeLists.txt             # Examples build config
│   ├── gomoku_example.cpp         # Gomoku example
│   ├── chess_example.cpp          # Chess example
│   └── go_example.cpp             # Go example
│
└── scripts/                       # Build/utility scripts
    ├── build.sh                   # Build script
    ├── test.sh                    # Test runner
    ├── format.sh                  # Code formatter
    └── install_dependencies.sh    # Dependency installer
```

## Key File Contents

### Core Interface (include/alphazero/core/igamestate.h)

This file would define the IGameState interface that all game implementations must implement:

```cpp
#ifndef IGAMESTATE_H
#define IGAMESTATE_H

#include <vector>
#include <string>
#include <memory>
#include <optional>

class ZobristHash;

enum class GameType {
    GOMOKU,
    CHESS,
    GO
};

enum class GameResult {
    ONGOING,
    DRAW,
    WIN_PLAYER1,
    WIN_PLAYER2
};

class IGameState {
public:
    explicit IGameState(GameType type);
    virtual ~IGameState() = default;
    
    // Core methods that all games must implement
    virtual std::vector<int> getLegalMoves() const = 0;
    virtual bool isLegalMove(int action) const = 0;
    virtual void makeMove(int action) = 0;
    virtual bool undoMove() = 0;
    virtual bool isTerminal() const = 0;
    virtual GameResult getGameResult() const = 0;
    virtual int getCurrentPlayer() const = 0;
    
    // Board information
    virtual int getBoardSize() const = 0;
    virtual int getActionSpaceSize() const = 0;
    
    // Neural network representation
    virtual std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const = 0;
    virtual std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const = 0;
    
    // Hash for transposition table
    virtual uint64_t getHash() const = 0;
    
    // Clone the current state
    virtual std::unique_ptr<IGameState> clone() const = 0;
    
    // String conversion
    virtual std::string actionToString(int action) const = 0;
    virtual std::optional<int> stringToAction(const std::string& moveStr) const = 0;
    virtual std::string toString() const = 0;
    
    // Additional methods
    virtual bool equals(const IGameState& other) const = 0;
    virtual std::vector<int> getMoveHistory() const = 0;
    virtual bool validate() const = 0;
    
    GameType getGameType() const { return gameType_; }
    
protected:
    GameType gameType_;
};

// Factory function to create game states
std::unique_ptr<IGameState> createGameState(GameType type, 
                                           int boardSize = 0, 
                                           bool variantRules = false);

#endif // IGAMESTATE_H
```

### MCTS Node (include/alphazero/mcts/mcts_node.h)

This file would define the Monte Carlo Tree Search node:

```cpp
#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include "../core/igamestate.h"

class MCTSNode {
public:
    MCTSNode(const IGameState* state, MCTSNode* parent = nullptr, float prior = 0.0f);
    ~MCTSNode();
    
    // Thread-safe statistics
    std::atomic<int> visitCount{0};
    std::atomic<float> valueSum{0.0f};
    float prior;  // Prior probability from neural network
    
    // Tree structure
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> actions;
    std::mutex expansionMutex;
    
    // State information
    uint64_t stateHash;
    bool isTerminal;
    GameResult gameResult;
    
    // Node methods
    float getValue() const;
    float getUcbScore(float cPuct, int currentPlayer, float fpuReduction = 0.0f) const;
    void addVirtualLoss(int virtualLoss);
    void removeVirtualLoss(int virtualLoss);
    int getBestAction() const;
    std::vector<float> getVisitCountDistribution(float temperature = 1.0f) const;
    
    // Debug utilities
    std::string toString(int maxDepth = 1) const;
};

#endif // MCTS_NODE_H
```

### Main CMake Configuration (CMakeLists.txt)

This file would contain the primary build configuration:

```cmake
cmake_minimum_required(VERSION 3.14)
project(AlphaZeroMultiGame VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(AZ_BUILD_TESTS "Build tests" ON)
option(AZ_USE_TORCH "Build with LibTorch support" ON)
option(AZ_USE_PYTHON "Build Python bindings" ON)
option(AZ_BUILD_GOMOKU "Build Gomoku game support" ON)
option(AZ_BUILD_CHESS "Build Chess game support" ON)
option(AZ_BUILD_GO "Build Go game support" ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

if(AZ_USE_TORCH)
    find_package(Torch REQUIRED)
    add_definitions(-DHAS_TORCH)
endif()

if(AZ_USE_PYTHON)
    find_package(Python 3.8 COMPONENTS Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    add_definitions(-DHAS_PYTHON)
endif()

if(AZ_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
endif()

# Define include directories
include_directories(include)

# Define libraries
add_subdirectory(src)

# Build tests
if(AZ_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Build examples
add_subdirectory(examples)

# Python bindings
if(AZ_USE_PYTHON)
    add_subdirectory(python)
endif()
```

### Python Training Script (python/scripts/train.py)

```python
#!/usr/bin/env python3
"""
Training script for the AlphaZero Multi-Game AI Engine neural network.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from alphazero.models.ddw_randwire import DDWRandWireNetwork
from alphazero.training.dataset import AlphaZeroDataset
from alphazero.training.scheduler import WarmupCosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero training script")
    parser.add_argument("--game", type=str, required=True, choices=["gomoku", "chess", "go"],
                        help="Game type to train")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--model", type=str, help="Path to load/save model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    return parser.parse_args()

def train(model, dataloader, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    
    for states, policies, values in dataloader:
        # Training logic
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        optimizer.zero_grad()
        policy_logits, value_preds = model(states)
        
        policy_loss = nn.functional.cross_entropy(policy_logits, policies)
        value_loss = nn.functional.mse_loss(value_preds.squeeze(-1), values)
        regularization_loss = model.get_l2_regularization_loss()
        
        loss = policy_loss + value_loss + regularization_loss
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * states.size(0)
        policy_loss_sum += policy_loss.item() * states.size(0)
        value_loss_sum += value_loss.item() * states.size(0)
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader.dataset)
    avg_policy_loss = policy_loss_sum / len(dataloader.dataset)
    avg_value_loss = value_loss_sum / len(dataloader.dataset)
    
    return {
        'loss': avg_loss,
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss
    }

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    if args.game == "gomoku":
        model = DDWRandWireNetwork(game_type="gomoku", board_size=15)
    elif args.game == "chess":
        model = DDWRandWireNetwork(game_type="chess", board_size=8)
    elif args.game == "go":
        model = DDWRandWireNetwork(game_type="go", board_size=19)
    
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=args.epochs)
    
    # Load training data
    dataset = AlphaZeroDataset(args.data, augment=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    
    # Training loop
    for epoch in range(args.epochs):
        # Train one epoch
        train_stats = train(model, dataloader, optimizer, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_stats['loss']:.4f}, "
              f"Policy Loss: {train_stats['policy_loss']:.4f}, "
              f"Value Loss: {train_stats['value_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.model}_epoch{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), f"{args.model}_final.pt")

if __name__ == "__main__":
    main()
```

This folder structure organizes the AlphaZero Multi-Game AI Engine according to the PRD requirements, with clear separation of components and a modular design that supports the three game types (Gomoku, Chess, and Go) through the common interfaces.