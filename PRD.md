# AlphaZero-style Multi-Game AI Engine: Product Requirements Document

## 1. Executive Summary and Success Metrics

This project aims to develop a **production-grade, AlphaZero-style Multi-Game AI engine** that can play Gomoku, Chess, and Go at expert levels without human-provided strategies. The engine uses a modular architecture implemented in C++ for high performance with Python bindings for training and integration. It combines **Monte Carlo Tree Search (MCTS)** with a **deep neural network** for evaluation, following DeepMind's AlphaZero approach.

The development will follow a phased approach, first implementing Gomoku, then Chess, and finally Go to manage complexity and ensure quality. Each game will be fully developed, tested, and optimized before moving to the next.

**Success Metrics:** 
- **Elo Rating / Win Rate:** Achieve a rating equivalent to a strong human player in each supported game. Target win rates of **≥ 90%** against benchmark AIs, or Elo ratings of **2000+ for Gomoku**, **2200+ for Chess**, and **2000+ for Go** in self-play evaluation.
- **Move Decision Latency:** Return a move decision for Gomoku within **150 ms on average** on a modern 8-core CPU, **300 ms for Chess**, and **500 ms for Go**. On GPU-accelerated setups, aim for **< 100 ms** for Gomoku, **< 200 ms** for Chess, and **< 300 ms** for Go.
- **Throughput for Self-Play:** Generate at least **50 Gomoku, 25 Chess, or 15 Go self-play games per minute** (using ~8 CPU threads and 1 GPU).
- **Resource Utilization:** Efficiently utilize hardware with **≥ 90% GPU utilization** during batched inference and **≥ 70%** CPU utilization across cores during parallel MCTS.
- **Training Efficiency:** Train to competitive models in **< 48 hours** for Gomoku, **< 96 hours** for Chess, and **< 144 hours** for Go on a single NVIDIA RTX 3090 GPU or equivalent (24GB VRAM, 10,000+ CUDA cores).
- **Stability and Robustness:** Zero critical crashes in at least **1000 consecutive games** of automated play-testing for each supported game. All games must adhere to official rules with 100% correctness.
- **Model Evolution:** Demonstrate consistent ELO rating growth during training with **improvement of at least 100 ELO** between consecutive training iterations until plateauing.
- **Memory Efficiency:** The engine should use < **500 MB** RAM for Gomoku, < **1 GB** for Chess, and < **2 GB** for Go (excluding neural network weights), verified through automated memory profiling.
- **Code Quality:** Maintain > 90% test coverage, zero critical static analysis warnings, and adherence to defined style guides for both C++ and Python code.

## 2. In-Scope vs. Out-of-Scope

### In Scope:
- **Multi-Game AI Engine Core** – Core algorithms for Gomoku (15x15 default with optional Renju rules), Chess, and Go (9x9 and 19x19), including game state management, MCTS search, and neural network inference.
- **Game Abstraction Layer** – An abstraction layer to allow different games to interface with the same core engine components.
- **Neural Network Architecture** – Implementation of a DDW-RandWire-ResNet architecture with dynamic graph topology, enhanced input representation, and attack/defense scoring.
- **AlphaZero-Style Self-Play Learning** – A reinforcement learning pipeline where the engine improves by playing against itself, with different configurations for each supported game.
- **ELO Rating System** – A comprehensive system for tracking model strength progression through training.
- **High-Performance C++ Implementation** – Core logic implemented in C++ for speed, with multi-threading, batch processing, SIMD optimization, and memory optimization.
- **Python Bindings** – PyBind11-based interfaces for orchestrating training, game-playing, and visualization, with proper memory management and GIL handling.
- **Single-Machine Training Pipeline** – System designed for single machine with multi-core CPU and single GPU.
- **Cross-Platform Support** – Building and running on **Linux** and **Windows** (64-bit).
- **Game Interface** – A modular interface allowing human players to play against the AI and run matches, with visualization support.
- **Game Record Storage** – Standardized formats for storing game records, training data, and model checkpoints.
- **CI/CD Pipeline** – Continuous integration and deployment workflow for automated building, testing, and quality assurance.
- **Documentation** – Comprehensive documentation for code, APIs, and user guides.

### Out of Scope:
- **Other Board Games:** Games beyond Gomoku, Chess, and Go are not in scope for the initial implementation.
- **Distributed Training:** No multi-machine distributed self-play or training in this phase.
- **Mobile/Web Deployment:** No optimization for mobile devices or web service provision, though the architecture will be designed to facilitate future extensions.
- **Full Game Suite:** No implementation of additional game features beyond the core gameplay (e.g., timers, undo/redo, game analysis tools).
- **Manually-Crafted Knowledge:** No human-crafted opening books or endgame databases. Learning purely from self-play.
- **External Tournament Support:** No support for participating in online game servers or tournaments.

## 3. System Architecture and Component Diagram

The architecture introduces modularity for multi-game support and includes the ELO tracking system:

```
┌───────────────────────────────┐          
│ Game Abstraction Layer        │          
│ ┌─────────────────────────┐   │          
│ │ IGameState              │   │          
│ │- Abstract Interface     │   │          
│ └─────────────────────────┘   │          
│                               │          
│ ┌─────────────────┐ ┌───────────────┐ ┌───────────┐
│ │ GomokuState     │ │ ChessState    │ │ GoState   │
│ │- Implementation │ │- Implementation│ │- Impl.    │
│ └─────────────────┘ └───────────────┘ └───────────┘
└───────────────────────────────┘          
           ▲                        
           │                        
           │                        
┌──────────▼─────────────┐          ┌──────────────────────────┐          ┌─────────────────────────────┐
│ Core MCTS Engine       │          │ Neural Network Engine     │          │ ELO Rating System           │
│ ┌────────────────────┐ │          │ ┌───────────────────────┐ │          │ ┌───────────────────────┐   │
│ │ MCTSNode           │ │          │ │ TorchNeuralNetwork    │ │          │ │ EloTracker            │   │
│ │- Statistics        │ │          │ │- ResNet Architecture  │ │          │ │- Rating Calculation   │   │
│ │- Children          │◄├──────────┼─┤- Enhanced Input Rep.  │ │          │ │- Match Management     │   │
│ │- Thread Safety     │ │          │ │- Attack/Defense Module│ │          │ │- History Tracking     │   │
│ └────────────────────┘ │          │ └───────────────────────┘ │          │ └───────────────────────┘   │
│ ┌────────────────────┐ │          │ ┌───────────────────────┐ │          │ ┌───────────────────────┐   │
│ │ ParallelMCTS       │ │          │ │ BatchQueue            │ │          │ │ RandomPolicyNetwork   │   │
│ │- Thread Pool       │◄├──────────┼─┤- Asynchronous Batching│ │          │ │- MCTS Baseline        │   │
│ │- Tree Management   │ │          │ │- GPU Utilization      │ │          │ │- Benchmark Opponent   │   │
│ └────────────────────┘ │          │ └───────────────────────┘ │          │ └───────────────────────┘   │
└──────────────────────┘          └──────────────────────────┘          └─────────────────────────────┘
        ▲                                      ▲                                       ▲
        │                                      │                                       │
        │                                      │                                       │
        ▼                                      ▼                                       ▼
┌─────────────────────────┐          ┌──────────────────────────┐          ┌─────────────────────────────┐
│ Support Systems         │          │ Self-Play System         │          │ Training System              │
│ ┌─────────────────────┐ │          │ ┌────────────────────┐   │          │ ┌───────────────────────┐   │
│ │ TranspositionTable  │ │          │ │ SelfPlayManager    │◄──┼───────────┤►│ Model Training        │   │
│ │- Thread-Safe Access │ │          │ │- Game Generation   │   │          │ │- Data Loading         │   │
│ │- Entry Management   │ │          │ │- Data Collection   │   │          │ │- Optimization         │   │
│ └─────────────────────┘ │          │ └────────────────────┘   │          │ └───────────────────────┘   │
│ ┌─────────────────────┐ │          │ ┌────────────────────┐   │          │ ┌───────────────────────┐   │
│ │ ZobristHash         │ │          │ │ GameRecord         │   │          │ │ ELO History Plotting  │   │
│ │- Hash Calculation   │ │          │ │- Game Storage      │   │          │ │- Training Visualization│   │
│ │- Game-Specific Impl.│ │          │ │- Serialization     │   │          │ │- Performance Analysis │   │
│ └─────────────────────┘ │          │ └────────────────────┘   │          │ └───────────────────────┘   │
└─────────────────────────┘          └──────────────────────────┘          └─────────────────────────────┘
                                              ▲                                        ▲
                                              │                                        │
                                              │                                        │
                                              ▼                                        ▼
┌────────────────────────────┐      ┌─────────────────────────────┐          ┌─────────────────────────────┐
│ Human Interface            │      │ Python Interface            │          │ CI/CD Pipeline               │
│ ┌──────────────────────┐   │      │ ┌───────────────────────┐   │          │ ┌───────────────────────┐   │
│ │ GameUI               │   │      │ │ PyBind11 Wrappers     │   │          │ │ Automated Building    │   │
│ │- Rendering           │◄──┼──────┼─┤- Game Control         │   │          │ │- Unit/Integration Tests│   │
│ │- Input Handling      │   │      │ │- Training Interface   │   │          │ │- Static Analysis      │   │
│ │- Game Controls       │   │      │ │- ELO Interface        │   │          │ │- Performance Testing  │   │
│ └──────────────────────┘   │      │ └───────────────────────┘   │          │ └───────────────────────┘   │
└────────────────────────────┘      └─────────────────────────────┘          └─────────────────────────────┘
```

*Figure: High-level architecture of the AlphaZero Multi-Game AI Engine showing modular components for Gomoku, Chess, and Go with enhanced neural network architecture and ELO tracking system.*

## 4. Non-Functional Requirements

### Performance
- **Game-Specific Inference Latency:** 
  - **Gomoku**: < 150 ms per move on CPU, < 50 ms on GPU
  - **Chess**: < 300 ms per move on CPU, < 200 ms on GPU
  - **Go (19x19)**: < 500 ms per move on CPU, < 300 ms on GPU

- **Throughput:** For self-play data generation:
  - **Gomoku**: At least 5,000 node expansions per second using 8 threads
  - **Chess**: At least 3,000 node expansions per second using 8 threads
  - **Go**: At least 2,000 node expansions per second using 8 threads

- **Scalability (CPU):** MCTS search performance should scale near-linearly with CPU threads, achieving ~80% or better of linear speedup for all supported games. Performance must be verified through automated benchmark tests that run as part of the CI pipeline.

- **Scalability (GPU Batch):** Neural network inference should efficiently utilize batch processing on the GPU, maintaining **> 90% GPU utilization** during large batch inference for all game types. This utilization will be measured and logged during training runs.

- **Memory Footprint:** The engine should use < **500 MB** RAM for Gomoku, < **1 GB** for Chess, and < **2 GB** for Go (excluding neural network weights). Memory profiling must be performed to verify these constraints, with tests implemented in the CI pipeline.

### Scalability and Concurrency
- **Game Abstraction Overhead:** The abstraction layer for supporting multiple games should add no more than 5% overhead compared to game-specific implementations. This will be measured through comparative benchmarks.

- **Interface Consistency:** All games must present a consistent interface to the MCTS and neural network components to ensure modularity. Interface compliance will be verified through static analysis and unit tests.

- **Configuration Adaptability:** The system should handle changes in game type, board size, and rule variations without requiring code changes. Configuration will be loaded from standardized JSON files.

- **Model Size Scalability:** The neural network architecture should support different model sizes optimized for each game without architectural changes. Model configurations will be defined in JSON format.

- **Multi-Threading:** The system must handle concurrent threads safely and efficiently across all game types. All shared data structures must be thread-safe with clear documentation of synchronization mechanisms. Thread sanitizers will be used to verify thread safety.

- **GIL Handling:** The Python Global Interpreter Lock should not bottleneck the engine. All heavy computations must run in C++ threads with the GIL released. Explicit `py::gil_scoped_release` will be used in all computation-intensive interfaces.

- **Resource Contention:** The design must avoid contention between CPU and GPU tasks, overlapping computation where possible. Thread and task scheduling will be optimized to maximize parallelism.

### Reliability and Robustness
- **Cross-Game Stability:** The engine must maintain stability across all supported games with no game-specific crashes or hangs. Each game must pass a suite of at least 1,000 automated test games.

- **Game Rule Compliance:** Each game implementation must strictly adhere to standard rules, verified through comprehensive test suites with at least 95% branch coverage.

- **Error Handling:** Game-specific errors must be properly encapsulated and handled without affecting the core engine. A consistent error handling strategy will be implemented, with proper exception propagation between C++ and Python layers.

- **Thread Safety Verification:** Thread sanitizers and other analysis tools should be used to catch data races or undefined behavior in multithreaded code. All multi-threaded code must pass thread sanitizer verification.

- **Reproducibility:** Given a fixed random seed and identical conditions, the engine's behavior should be deterministic for all games. Reproducibility tests will be part of the CI pipeline.

- **Graceful Termination:** If the engine is running a long search or self-play loop and needs to terminate, it should do so cleanly, releasing all resources and saving state where appropriate. Signal handlers will be implemented for proper shutdown.

- **Input Validation:** All external inputs (configurations, moves, etc.) must be validated before processing to prevent crashes and security issues. Comprehensive input validation will be applied at all interface boundaries.

### Maintainability and Extensibility
- **Game Addition Simplicity:** Adding a new board game should require implementing only the game-specific logic without modifying the core engine. A comprehensive guide will document this process.

- **Neural Network Configurability:** The neural network architecture should be configurable via hyperparameters without code changes. All neural network parameters will be defined in configuration files.

- **ELO System Flexibility:** The ELO rating system should work consistently across all game types with appropriate benchmarks for each. ELO calculation parameters will be configurable per game type.

- **Modular Codebase:** The code must be organized into clear modules with well-defined interfaces between components. Each module will have its own set of unit tests and documentation.

- **Coding Standards:** The code should adhere to modern C++ best practices (C++20 standard) and consistent style conventions. Python code should follow PEP8. Automated style checking will be part of the CI pipeline.

- **Memory Management:** A consistent memory management strategy must be used throughout the codebase, with clear ownership semantics. Smart pointers will be used for all dynamically allocated objects with documented ownership transfer at interfaces.

### Observability and Monitoring
- **Game-Specific Metrics:** The engine should provide game-specific metrics for monitoring performance and behavior, exposed through a consistent metrics interface.

- **Model Strength Tracking:** The ELO rating system should provide clear visibility into model strength progression over time, with automated visualization of progress.

- **Logging:** The engine should include comprehensive logging facilities for debugging and monitoring important events and statistics. Log levels (DEBUG, INFO, WARNING, ERROR) should be configurable.

- **Metrics Exposure:** The engine should provide ways to access internal metrics for performance monitoring. Metrics will be accessible through both C++ and Python interfaces.

- **Profiling Support:** The code should be instrumented with timing measurements around critical sections. In debug mode, detailed profiling information will be available.

- **Visualization Tools:** The system should include tools for visualizing game states, search trees, and training progress. These tools will be implemented in Python using standard visualization libraries.

### Portability
- **Game-Agnostic Containers:** Containerized environments should support all game types without game-specific configurations. Docker containers will be provided for both development and deployment.

- **Supported Platforms:** The engine must run on **Linux (Ubuntu 20.04+ or equivalent)** and **Windows 10/11** (64-bit). All platform-specific code must be properly abstracted.

- **Compiler Compatibility:** Minimum required compiler versions are GCC 11, Clang 12, and MSVC 2019 (which support C++20 features). Compiler-specific issues will be handled through conditional compilation.

- **Python Versions:** The Python API should target **Python 3.8 through 3.11** with proper version checks and compatibility handling. Type hints will be used throughout Python code following PEP 484.

## 5. Game Abstraction Layer

The Game Abstraction Layer provides a unified interface for different board games to interact with the MCTS and neural network components.

### 5.1 IGameState Interface

```cpp
// igamestate.h
#ifndef IGAMESTATE_H
#define IGAMESTATE_H

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <stdexcept>

// Forward declarations
class ZobristHash;

// Game type identifiers
enum class GameType {
    GOMOKU,
    CHESS,
    GO
};

// Result of a game
enum class GameResult {
    ONGOING,
    DRAW,
    WIN_PLAYER1,
    WIN_PLAYER2
};

// Game state interface
class IGameState {
public:
    // Constructor with game type
    explicit IGameState(GameType type) : game_type_(type) {}
    
    // Virtual destructor
    virtual ~IGameState() = default;
    
    // Core game state methods that all games must implement
    virtual std::vector<int> getLegalMoves() const = 0;
    virtual bool isLegalMove(int action) const = 0;
    virtual void makeMove(int action) = 0;
    virtual bool undoMove() = 0;  // Returns false if no moves to undo
    virtual bool isTerminal() const = 0;
    virtual GameResult getGameResult() const = 0;  // Returns game result enum
    virtual int getCurrentPlayer() const = 0;  // 1=player1, 2=player2
    
    // Board information
    virtual int getBoardSize() const = 0;
    virtual int getActionSpaceSize() const = 0;  // Total number of possible actions
    
    // Tensor representation for neural network
    virtual std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const = 0;
    
    // Enhanced tensor representation with attack/defense and history
    virtual std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const = 0;
    
    // Hash for transposition table
    virtual uint64_t getHash() const = 0;
    
    // Clone the current state
    virtual std::unique_ptr<IGameState> clone() const = 0;
    
    // Convert between actions and human-readable notation
    virtual std::string actionToString(int action) const = 0;
    virtual std::optional<int> stringToAction(const std::string& moveStr) const = 0;
    
    // Game type accessor
    GameType getGameType() const { return game_type_; }
    
    // String representation for display
    virtual std::string toString() const = 0;
    
    // Equality comparison
    virtual bool equals(const IGameState& other) const = 0;

    // Get the history of moves
    virtual std::vector<int> getMoveHistory() const = 0;
    
    // Check if state is valid
    virtual bool validate() const = 0;
    
protected:
    GameType game_type_;
};

// Factory function to create game states
std::unique_ptr<IGameState> createGameState(GameType type, 
                                           int boardSize = 0, 
                                           bool variantRules = false);

// Exception classes for game-related errors
class GameStateException : public std::runtime_error {
public:
    explicit GameStateException(const std::string& message) 
        : std::runtime_error(message) {}
};

class IllegalMoveException : public GameStateException {
public:
    IllegalMoveException(const std::string& message, int action) 
        : GameStateException(message), action_(action) {}
    int getAction() const { return action_; }
private:
    int action_;
};

#endif // IGAMESTATE_H
```

### 5.2 Implementation for Gomoku

```cpp
// gomoku_state.h
#ifndef GOMOKU_STATE_H
#define GOMOKU_STATE_H

#include "igamestate.h"
#include "zobrist_hash.h"
#include <vector>
#include <cstdint>
#include <array>
#include <bitset>
#include <mutex>

class GomokuState : public IGameState {
public:
    // Constants
    static constexpr int BLACK = 1;
    static constexpr int WHITE = 2;
    static constexpr int MAX_BOARD_SIZE = 19;
    static constexpr int DEFAULT_BOARD_SIZE = 15;

    // Constructors
    explicit GomokuState(int boardSize = DEFAULT_BOARD_SIZE, bool useRenjuRules = false);
    GomokuState(const GomokuState& other);
    GomokuState(GomokuState&& other) noexcept;
    GomokuState& operator=(const GomokuState& other);
    GomokuState& operator=(GomokuState&& other) noexcept;
    
    // IGameState interface implementation
    std::vector<int> getLegalMoves() const override;
    bool isLegalMove(int action) const override;
    void makeMove(int action) override;
    bool undoMove() override;
    bool isTerminal() const override;
    GameResult getGameResult() const override;
    int getCurrentPlayer() const override { return currentPlayer_; }
    int getBoardSize() const override { return boardSize_; }
    int getActionSpaceSize() const override { return boardSize_ * boardSize_; }
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override;
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override;
    uint64_t getHash() const override { return currentHash_; }
    std::unique_ptr<IGameState> clone() const override;
    std::string actionToString(int action) const override;
    std::optional<int> stringToAction(const std::string& moveStr) const override;
    std::string toString() const override;
    bool equals(const IGameState& other) const override;
    std::vector<int> getMoveHistory() const override { return moveHistory_; }
    bool validate() const override;
    
    // Gomoku-specific methods
    bool isColor(int position, int color) const;
    std::vector<std::vector<int>> getBoard() const;
    std::pair<int, int> positionToXY(int position) const;
    int xyToPosition(int x, int y) const;
    bool usingRenjuRules() const { return useRenjuRules_; }
    
    // Attack/Defense scoring for enhanced representation
    std::pair<std::vector<float>, std::vector<float>> computeAttackDefenseScores() const;
    std::vector<int> getPreviousMoves(int player, int count) const;
    
private:
    // Board representation
    int boardSize_;
    int currentPlayer_;
    std::vector<uint64_t> blackStones_;
    std::vector<uint64_t> whiteStones_;
    int numBitboardChunks_;
    
    // Game rules
    bool useRenjuRules_;
    
    // Move history
    std::vector<int> moveHistory_;
    
    // Zobrist hashing
    ZobristHash zobrist_;
    uint64_t currentHash_;
    
    // Helper methods
    bool checkFiveInARow(int position, int color) const;
    bool isForbiddenMove(int position) const;
    void updateHash(int position, int color);
    bool hasFiveInARow(int color) const;
    
    // Cache for optimization
    mutable bool terminalStateChecked_;
    mutable GameResult cachedResult_;
    mutable std::vector<int> cachedLegalMoves_;
    mutable bool legalMovesDirty_;
    
    // Thread safety for mutable caches
    mutable std::mutex cacheMutex_;
    
    // Bitboard operations
    void setBit(std::vector<uint64_t>& bitboard, int position);
    void clearBit(std::vector<uint64_t>& bitboard, int position);
    bool testBit(const std::vector<uint64_t>& bitboard, int position) const;
    
    // Constants for bit manipulation
    static constexpr int BITS_PER_CHUNK = 64;
};

// Inline implementations of simple methods
inline bool GomokuState::isColor(int position, int color) const {
    if (position < 0 || position >= boardSize_ * boardSize_) {
        return false;
    }
    
    if (color == BLACK) {
        return testBit(blackStones_, position);
    } else if (color == WHITE) {
        return testBit(whiteStones_, position);
    }
    
    return false;
}

inline std::pair<int, int> GomokuState::positionToXY(int position) const {
    if (position < 0 || position >= boardSize_ * boardSize_) {
        throw IllegalMoveException("Invalid position", position);
    }
    return {position % boardSize_, position / boardSize_};
}

inline int GomokuState::xyToPosition(int x, int y) const {
    if (x < 0 || x >= boardSize_ || y < 0 || y >= boardSize_) {
        throw GameStateException("Invalid coordinates");
    }
    return y * boardSize_ + x;
}

#endif // GOMOKU_STATE_H
```

### 5.3 Factory Implementation

```cpp
// game_factory.cpp
#include "igamestate.h"
#include "gomoku_state.h"
#include "chess_state.h"
#include "go_state.h"
#include <stdexcept>
#include <string>

std::unique_ptr<IGameState> createGameState(GameType type, int boardSize, bool variantRules) {
    try {
        switch (type) {
            case GameType::GOMOKU:
                return std::make_unique<GomokuState>(
                    boardSize > 0 ? boardSize : GomokuState::DEFAULT_BOARD_SIZE,
                    variantRules  // Renju rules if true
                );
            
            case GameType::CHESS:
                return std::make_unique<ChessState>(
                    variantRules  // Chess960 if true
                );
            
            case GameType::GO:
                return std::make_unique<GoState>(
                    boardSize > 0 ? boardSize : GoState::DEFAULT_BOARD_SIZE,
                    variantRules  // Chinese rules if true, Japanese if false
                );
            
            default:
                throw std::invalid_argument("Unsupported game type");
        }
    } catch (const std::exception& e) {
        throw GameStateException("Failed to create game state: " + std::string(e.what()));
    }
}
```

### 5.4 Enhanced Neural Network Input Representation

```cpp
// Enhanced tensor representation for Gomoku
std::vector<std::vector<std::vector<float>>> GomokuState::getEnhancedTensorRepresentation() const {
    // Create a tensor with multiple planes:
    // - Channel 0-1: Current player's stones and opponent's stones (base representation)
    // - Channel 2: Player flag (1.0 if current player is BLACK, 0.0 otherwise)
    // - Channel 3: Attack score for current player
    // - Channel 4: Defense score for current player
    // - Channel 5-14: Previous N moves (5 for each player, one hot encoded)
    // - Channel 15-19: CoordConv channels for position encoding
    
    // Calculate attack/defense scores
    auto [attackScores, defenseScores] = computeAttackDefenseScores();
    
    // Get previous moves for history channels
    std::vector<int> prevMovesP1 = getPreviousMoves(BLACK, 5);
    std::vector<int> prevMovesP2 = getPreviousMoves(WHITE, 5);
    
    // Base size expanded to accommodate all features
    const int numChannels = 20;
    std::vector<std::vector<std::vector<float>>> tensor(numChannels, 
        std::vector<std::vector<float>>(boardSize_, 
            std::vector<float>(boardSize_, 0.0f)));
    
    // 1. Current board state (channels 0-1)
    int pIdx = currentPlayer_ - 1;
    int oppIdx = 1 - pIdx;
    
    for (int y = 0; y < boardSize_; y++) {
        for (int x = 0; x < boardSize_; x++) {
            int pos = xyToPosition(x, y);
            if (isColor(pos, currentPlayer_)) {
                tensor[0][y][x] = 1.0f;
            } else if (isColor(pos, 3 - currentPlayer_)) {
                tensor[1][y][x] = 1.0f;
            }
        }
    }
    
    // 2. Player flag (channel 2)
    if (currentPlayer_ == BLACK) {
        for (int y = 0; y < boardSize_; y++) {
            for (int x = 0; x < boardSize_; x++) {
                tensor[2][y][x] = 1.0f;
            }
        }
    }
    
    // 3. Attack/Defense scores (channels 3-4)
    if (!attackScores.empty() && !defenseScores.empty()) {
        for (int y = 0; y < boardSize_; y++) {
            for (int x = 0; x < boardSize_; x++) {
                int pos = xyToPosition(x, y);
                if (pos < static_cast<int>(attackScores.size()) && 
                    pos < static_cast<int>(defenseScores.size())) {
                    // Normalize scores to [0, 1] range
                    float normAttack = std::min(1.0f, std::max(0.0f, attackScores[pos] / 5.0f));
                    float normDefense = std::min(1.0f, std::max(0.0f, defenseScores[pos] / 5.0f));
                    
                    tensor[3][y][x] = normAttack;
                    tensor[4][y][x] = normDefense;
                }
            }
        }
    }
    
    // 4. Previous N moves (channels 5-14)
    // 5-9: BLACK's previous 5 moves
    for (size_t i = 0; i < prevMovesP1.size() && i < 5; i++) {
        auto [x, y] = positionToXY(prevMovesP1[i]);
        tensor[5 + static_cast<int>(i)][y][x] = 1.0f;
    }
    
    // 10-14: WHITE's previous 5 moves
    for (size_t i = 0; i < prevMovesP2.size() && i < 5; i++) {
        auto [x, y] = positionToXY(prevMovesP2[i]);
        tensor[10 + static_cast<int>(i)][y][x] = 1.0f;
    }
    
    // 5. Add CoordConv channels (15-19)
    // Horizontal and vertical position encoding
    for (int y = 0; y < boardSize_; y++) {
        for (int x = 0; x < boardSize_; x++) {
            // Normalize coordinates to [0, 1]
            tensor[15][y][x] = static_cast<float>(x) / (boardSize_ - 1);
            tensor[16][y][x] = static_cast<float>(y) / (boardSize_ - 1);
            
            // Distance from center
            float centerX = boardSize_ / 2.0f;
            float centerY = boardSize_ / 2.0f;
            float distX = std::abs(x - centerX) / centerX;
            float distY = std::abs(y - centerY) / centerY;
            tensor[17][y][x] = distX;
            tensor[18][y][x] = distY;
            tensor[19][y][x] = std::sqrt(distX*distX + distY*distY) / std::sqrt(2.0f);
        }
    }
    
    return tensor;
}
```

## 6. Attack/Defense Module

The Attack/Defense Module evaluates game positions to identify tactical threats and defensive opportunities, enhancing the AI's ability to evaluate positions.

```cpp
// attack_defense_module.h
#ifndef ATTACK_DEFENSE_MODULE_H
#define ATTACK_DEFENSE_MODULE_H

#include <vector>
#include <utility>
#include <array>
#include <memory>
#include <unordered_map>
#include "igamestate.h"

// Forward declarations
class GomokuState;
class ChessState;
class GoState;

// Pattern type enums for different games
enum class GomokuPatternType {
    FIVE_IN_A_ROW,
    OPEN_FOUR,
    FOUR,
    OPEN_THREE,
    THREE,
    OPEN_TWO,
    NONE
};

class AttackDefenseModule {
public:
    explicit AttackDefenseModule(GameType gameType, int boardSize = 0);
    ~AttackDefenseModule() = default;
    
    // Non-copyable but movable
    AttackDefenseModule(const AttackDefenseModule&) = delete;
    AttackDefenseModule& operator=(const AttackDefenseModule&) = delete;
    AttackDefenseModule(AttackDefenseModule&&) noexcept = default;
    AttackDefenseModule& operator=(AttackDefenseModule&&) noexcept = default;
    
    // Compute attack and defense scores for a batch of positions
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    computeBatchScores(
        const std::vector<std::reference_wrapper<const IGameState>>& states
    );
    
    // Compute scores for a single position
    std::pair<std::vector<float>, std::vector<float>> 
    computeScores(const IGameState& state);
    
    // Game-specific scoring dispatchers
    std::pair<std::vector<float>, std::vector<float>> 
    computeGomokuScores(const GomokuState& state);
    
    std::pair<std::vector<float>, std::vector<float>> 
    computeChessScores(const ChessState& state);
    
    std::pair<std::vector<float>, std::vector<float>> 
    computeGoScores(const GoState& state);
    
    // Cache control
    void clearCache();
    size_t getCacheSize() const;
    
private:
    GameType gameType_;
    int boardSize_;
    
    // Cache for computed scores
    struct CacheEntry {
        std::vector<float> attackScores;
        std::vector<float> defenseScores;
        uint64_t timestamp;
    };
    
    mutable std::unordered_map<uint64_t, CacheEntry> scoreCache_;
    mutable std::mutex cacheMutex_;
    uint64_t cacheCounter_ = 0;
    static constexpr size_t MAX_CACHE_SIZE = 10000;
    
    // Pattern recognition for Gomoku
    GomokuPatternType identifyGomokuPattern(
        const std::vector<int>& line, 
        int position, 
        int player
    );
    
    float evaluateGomokuPattern(GomokuPatternType patternType, bool forAttack);
    std::vector<std::vector<int>> getGomokuLines(
        const GomokuState& state,
        int position
    );
    
    // Helper methods
    void pruneCache();
    std::vector<float> normalizeScores(const std::vector<float>& scores, float maxValue = 1.0f);
};

#endif // ATTACK_DEFENSE_MODULE_H
```

## 7. Neural Network Architecture

### 7.1 Neural Network Interface

```cpp
// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <future>
#include <functional>
#include "igamestate.h"

class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;
    
    // Single state prediction
    virtual std::pair<std::vector<float>, float> predict(const IGameState& state) = 0;
    
    // Batch prediction
    virtual void predictBatch(
        const std::vector<std::reference_wrapper<const IGameState>>& states,
        std::vector<std::vector<float>>& policies,
        std::vector<float>& values
    ) = 0;
    
    // Asynchronous prediction
    virtual std::future<std::pair<std::vector<float>, float>> predictAsync(
        const IGameState& state
    ) = 0;
    
    // Device information
    virtual bool isGpuAvailable() const = 0;
    virtual std::string getDeviceInfo() const = 0;
    virtual float getInferenceTimeMs() const = 0;
    virtual int getBatchSize() const = 0;
    
    // Model information
    virtual std::string getModelInfo() const = 0;
    virtual size_t getModelSizeBytes() const = 0;
    
    // Performance testing
    virtual void benchmark(int numIterations = 100, int batchSize = 16) = 0;
    
    // Debugging
    virtual void enableDebugMode(bool enable) = 0;
    virtual void printModelSummary() const = 0;
    
    // Create factory method
    static std::unique_ptr<NeuralNetwork> create(
        const std::string& modelPath,
        GameType gameType,
        int boardSize = 0,
        bool useGpu = true
    );
};

#endif // NEURAL_NETWORK_H
```

### 7.2 PyTorch Neural Network Implementation with DDW-RandWire Architecture

```cpp
// torch_neural_network.h
#ifndef TORCH_NEURAL_NETWORK_H
#define TORCH_NEURAL_NETWORK_H

#include <torch/torch.h>
#include <torch/script.h>
#include "neural_network.h"
#include "igamestate.h"
#include "attack_defense_module.h"
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <queue>
#include <random>

class TorchNeuralNetwork : public NeuralNetwork {
public:
    TorchNeuralNetwork(
        const std::string& modelPath, 
        GameType gameType, 
        int boardSize = 0, 
        bool useGpu = true
    );
    ~TorchNeuralNetwork() override;
    
    // NeuralNetwork interface implementation
    std::pair<std::vector<float>, float> predict(const IGameState& state) override;
    
    void predictBatch(
        const std::vector<std::reference_wrapper<const IGameState>>& states,
        std::vector<std::vector<float>>& policies,
        std::vector<float>& values
    ) override;
    
    std::future<std::pair<std::vector<float>, float>> predictAsync(
        const IGameState& state
    ) override;
    
    bool isGpuAvailable() const override;
    std::string getDeviceInfo() const override;
    float getInferenceTimeMs() const override { return avgInferenceTime_; }
    int getBatchSize() const override { return maxBatchSize_; }
    
    std::string getModelInfo() const override;
    size_t getModelSizeBytes() const override;
    
    void benchmark(int numIterations = 100, int batchSize = 16) override;
    
    void enableDebugMode(bool enable) override { debugMode_ = enable; }
    void printModelSummary() const override;
    
    // Advanced settings
    void setMaxBatchSize(int size);
    void setFp16Mode(bool enable);
    void optimizeForInference();
    
    // Handling background inference thread for async operations
    void startInferenceThread();
    void stopInferenceThread();
    
    // Create DDW-RandWire-ResNet architecture (for training)
    static torch::nn::Sequential createDDWRandWireNetwork(
        int inputChannels, 
        int numChannels = 144,
        int numNodes = 40,
        int avgOutDegree = 6
    );
    
    static torch::nn::Sequential createPolicyHead(
        int numChannels, 
        int actionSpace
    );
    
    static torch::nn::Sequential createValueHead(
        int numChannels
    );
    
private:
    // Model and device
    std::shared_ptr<torch::jit::Module> model_;
    torch::Device device_;
    GameType gameType_;
    int boardSize_;
    std::unordered_map<GameType, int> actionSpaces_;
    bool debugMode_;
    std::unique_ptr<AttackDefenseModule> adModule_;
    
    // Performance tracking
    std::atomic<float> avgInferenceTime_{0.0f};
    std::atomic<int> inferenceCount_{0};
    int maxBatchSize_;
    bool useFp16_;
    
    // Thread safety for inference
    std::mutex inferenceMutex_;
    
    // Background inference thread
    bool inferenceThreadRunning_ = false;
    std::thread inferenceThread_;
    std::mutex queueMutex_;
    std::condition_variable queueCondVar_;
    
    struct InferenceRequest {
        std::shared_ptr<IGameState> state;
        std::promise<std::pair<std::vector<float>, float>> promise;
    };
    
    std::queue<InferenceRequest> inferenceQueue_;
    
    // Batch processing
    void processBatchesInBackground();
    void processQueuedRequests();
    
    // Conversion helpers
    torch::Tensor stateToTensor(const IGameState& state);
    torch::Tensor batchStatesToTensor(
        const std::vector<std::reference_wrapper<const IGameState>>& states
    );
    std::vector<float> tensorToPolicy(
        const torch::Tensor& policyTensor, 
        int actionSpace
    );
    
    // Internal inference method with timing
    std::pair<torch::Tensor, torch::Tensor> forwardInternal(const torch::Tensor& input);
    
    // Measure and update timing
    void updateTiming(float ms);
    
    // Helper methods for DDW-RandWire network
    static std::vector<std::pair<int, int>> generateRandomGraph(
        int nodes, 
        float avgOutDegree,
        bool useSmallWorld = true
    );
    
    static torch::nn::Sequential createResidualBlock(int channels);
    static torch::nn::Sequential createSERouter(int channels);
};

#endif // TORCH_NEURAL_NETWORK_H
```

### 7.3 DDW-RandWire-ResNet Neural Network Architecture

```cpp
// torch_neural_network.cpp (partial implementation)

torch::nn::Sequential TorchNeuralNetwork::createDDWRandWireNetwork(
    int inputChannels, 
    int numChannels, 
    int numNodes,
    int avgOutDegree
) {
    // Create the stem layer (3x3 conv from inputChannels to numChannels)
    auto stem = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, numChannels, 3).padding(1).stride(1)),
        torch::nn::BatchNorm2d(numChannels),
        torch::nn::ReLU()
    );
    
    // Create the DDW-RandWire backbone
    std::vector<torch::nn::Sequential> nodes;
    for (int i = 0; i < numNodes; i++) {
        nodes.push_back(createResidualBlock(numChannels));
    }
    
    // Generate random graph topology (mix of small-world and scale-free)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    bool useSmallWorld = dist(gen) < 0.5; // 50% chance for each topology
    auto connections = generateRandomGraph(numNodes, static_cast<float>(avgOutDegree), useSmallWorld);
    
    // Create the SE-style router for each connection
    std::vector<torch::nn::Sequential> routers;
    for (size_t i = 0; i < connections.size(); i++) {
        routers.push_back(createSERouter(numChannels));
    }
    
    // Register modules
    auto backbone = torch::nn::Sequential();
    backbone->push_back(stem);
    
    // Register nodes and routers with metadata for connections
    for (size_t i = 0; i < nodes.size(); i++) {
        std::string nodeName = "node_" + std::to_string(i);
        backbone->push_back(nodes[i]);
    }
    
    for (size_t i = 0; i < routers.size(); i++) {
        std::string routerName = "router_" + std::to_string(i);
        backbone->push_back(routers[i]);
    }
    
    // Add module to handle the dynamic wiring
    backbone->push_back(torch::nn::Functional([numNodes, connections](torch::Tensor x) {
        std::vector<torch::Tensor> nodeOutputs;
        nodeOutputs.reserve(numNodes);
        
        // Initial node has x as input
        nodeOutputs.push_back(x);
        
        // Process all remaining nodes based on connections
        for (int nodeIdx = 1; nodeIdx < numNodes; nodeIdx++) {
            torch::Tensor nodeInput = torch::zeros_like(x);
            int inputCount = 0;
            
            // Collect inputs from all incoming connections
            for (const auto& conn : connections) {
                if (conn.second == nodeIdx) { // This node is the target
                    nodeInput += nodeOutputs[conn.first];
                    inputCount++;
                }
            }
            
            // Average the inputs
            if (inputCount > 0) {
                nodeInput = nodeInput / inputCount;
            }
            
            nodeOutputs.push_back(nodeInput);
        }
        
        // Final output is the average of all outputs from terminal nodes
        torch::Tensor output = torch::zeros_like(x);
        int outputCount = 0;
        
        // A node is terminal if it has no outgoing connections
        std::vector<bool> hasOutgoing(numNodes, false);
        for (const auto& conn : connections) {
            hasOutgoing[conn.first] = true;
        }
        
        for (int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
            if (!hasOutgoing[nodeIdx]) {
                output += nodeOutputs[nodeIdx];
                outputCount++;
            }
        }
        
        if (outputCount > 0) {
            output = output / outputCount;
        } else {
            // Fallback: use the last node's output
            output = nodeOutputs.back();
        }
        
        return output;
    }));
    
    return backbone;
}

torch::nn::Sequential TorchNeuralNetwork::createPolicyHead(int numChannels, int actionSpace) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(numChannels, 1, 1)),
        torch::nn::Flatten(),
        torch::nn::Linear(1 * 1, actionSpace) // Assumes board size is handled in the reshape
    );
}

torch::nn::Sequential TorchNeuralNetwork::createValueHead(int numChannels) {
    return torch::nn::Sequential(
        torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(numChannels, numChannels, 1)),
        torch::nn::ReLU(),
        torch::nn::Flatten(),
        torch::nn::Linear(numChannels, 1),
        torch::nn::Tanh()
    );
}

std::vector<std::pair<int, int>> TorchNeuralNetwork::generateRandomGraph(
    int nodes, 
    float avgOutDegree,
    bool useSmallWorld
) {
    std::vector<std::pair<int, int>> connections;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (useSmallWorld) {
        // Small-world network generation (Watts-Strogatz inspired)
        // Each node is connected to K nearest neighbors
        int K = static_cast<int>(avgOutDegree);
        float rewireProb = 0.1f;
        
        // Create regular ring lattice
        for (int i = 0; i < nodes; i++) {
            for (int j = 1; j <= K / 2; j++) {
                int target = (i + j) % nodes;
                connections.push_back({i, target});
                connections.push_back({i, (i - j + nodes) % nodes});
            }
        }
        
        // Rewire some connections randomly
        std::uniform_real_distribution<> dist(0.0, 1.0);
        std::uniform_int_distribution<> nodeDist(0, nodes - 1);
        
        for (auto& conn : connections) {
            if (dist(gen) < rewireProb) {
                int newTarget = nodeDist(gen);
                while (newTarget == conn.first || newTarget == conn.second) {
                    newTarget = nodeDist(gen);
                }
                conn.second = newTarget;
            }
        }
    } else {
        // Scale-free network (Barabási–Albert inspired)
        std::vector<int> degrees(nodes, 0);
        
        // Start with a small fully connected network
        int initialNodes = std::min(5, nodes);
        for (int i = 0; i < initialNodes; i++) {
            for (int j = i + 1; j < initialNodes; j++) {
                connections.push_back({i, j});
                degrees[i]++;
                degrees[j]++;
            }
        }
        
        // Add remaining nodes with preferential attachment
        for (int i = initialNodes; i < nodes; i++) {
            // Each new node creates 'avgOutDegree' connections
            int edgesToAdd = static_cast<int>(avgOutDegree);
            
            for (int e = 0; e < edgesToAdd; e++) {
                // Preferential attachment - nodes with higher degrees
                // are more likely to get new connections
                std::vector<double> weights(i);
                for (int j = 0; j < i; j++) {
                    weights[j] = degrees[j] + 1.0; // +1 to avoid zero probability
                }
                
                std::discrete_distribution<> weightedDist(weights.begin(), weights.end());
                int target = weightedDist(gen);
                
                connections.push_back({i, target});
                degrees[i]++;
                degrees[target]++;
            }
        }
    }
    
    return connections;
}

torch::nn::Sequential TorchNeuralNetwork::createResidualBlock(int channels) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
        torch::nn::BatchNorm2d(channels),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)),
        torch::nn::BatchNorm2d(channels),
        torch::nn::Functional([](torch::Tensor x) {
            auto identity = x;
            x = x + identity;
            return torch::relu(x);
        })
    );
}

torch::nn::Sequential TorchNeuralNetwork::createSERouter(int channels) {
    return torch::nn::Sequential(
        torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels / 16, 1)),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels / 16, channels, 1)),
        torch::nn::Sigmoid()
    );
}
```

## 8. Monte Carlo Tree Search Implementation

### 8.1 MCTS Node Structure

```cpp
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
#include "igamestate.h"

class MCTSNode {
public:
    // Constructor
    MCTSNode(const IGameState* state, MCTSNode* parent = nullptr, float prior = 0.0f);
    
    // Non-copyable but movable
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) noexcept = default;
    MCTSNode& operator=(MCTSNode&&) noexcept = default;
    
    // Destructor
    ~MCTSNode();
    
    // Core MCTS statistics - atomic for thread safety
    std::atomic<int> visitCount{0};
    std::atomic<float> valueSum{0.0f};
    float prior;  // Prior probability from policy network
    
    // Tree structure
    MCTSNode* parent;                  // Parent node
    std::vector<std::unique_ptr<MCTSNode>> children;  // Child nodes (owned)
    std::vector<int> actions;          // Actions leading to each child
    std::mutex expansionMutex;         // Mutex for thread-safe expansion
    
    // State information
    uint64_t stateHash;                // Zobrist hash of the state
    bool isTerminal;                   // Whether node represents a terminal state
    GameResult gameResult;             // Game outcome if terminal
    GameType gameType;                 // Type of game this node represents
    
    // Expansion status
    bool isExpanded;                   // Whether node has been expanded
    
    // Computed values and scoring
    float getValue() const {
        return visitCount.load() > 0 ? valueSum.load() / visitCount.load() : 0.0f;
    }
    
    // Convert game result to value (-1 to 1)
    float getTerminalValue(int currentPlayer) const;
    
    // UCB score calculation for selection phase
    float getUcbScore(float cPuct, int currentPlayer, float fpuReduction = 0.0f) const;
    
    // Virtual loss for thread-safe parallelization
    void addVirtualLoss(int virtualLoss);
    void removeVirtualLoss(int virtualLoss);
    
    // Child management
    MCTSNode* getChild(int actionIndex) const;
    int getActionIndex(int action) const;
    void addChild(int action, float prior, std::unique_ptr<MCTSNode> child);
    bool hasChildren() const { return !children.empty(); }
    
    // Best child selection
    int getBestAction() const;
    MCTSNode* getBestChild() const;
    std::vector<float> getVisitCountDistribution(float temperature = 1.0f) const;
    
    // Debug utilities
    std::string toString(int maxDepth = 1) const;
    void printTree(int maxDepth = 1) const;
    
private:
    // Helper methods
    std::string indentString(int depth) const;
    float convertToValue(GameResult result, int perspectivePlayer) const;
};

#endif // MCTS_NODE_H
```

### 8.2 Parallel MCTS Implementation

```cpp
// parallel_mcts.h
#ifndef PARALLEL_MCTS_H
#define PARALLEL_MCTS_H

#include <vector>
#include <atomic>
#include <mutex>
#include <random>
#include <future>
#include <thread>
#include <memory>
#include <condition_variable>
#include <functional>
#include "igamestate.h"
#include "mcts_node.h"
#include "thread_pool.h"
#include "neural_network.h"
#include "transposition_table.h"

enum class MCTSNodeSelection {
    UCB,             // Standard UCB formula
    PUCT,            // AlphaZero's PUCT formula
    PROGRESSIVE_BIAS // Progressive bias with visit count
};

class ParallelMCTS {
public:
    // Constructor
    ParallelMCTS(
        const IGameState& rootState,
        NeuralNetwork* nn = nullptr,
        TranspositionTable* tt = nullptr,
        int numThreads = 1,
        int numSimulations = 800,
        float cPuct = 1.5f,
        float fpuReduction = 0.0f,
        int virtualLoss = 3
    );
    
    // Destructor
    ~ParallelMCTS();
    
    // Non-copyable but movable
    ParallelMCTS(const ParallelMCTS&) = delete;
    ParallelMCTS& operator=(const ParallelMCTS&) = delete;
    ParallelMCTS(ParallelMCTS&&) noexcept = default;
    ParallelMCTS& operator=(ParallelMCTS&&) noexcept = default;
    
    // Run MCTS search
    void search();
    
    // Get best action according to visit counts
    int selectAction(bool isTraining = false, float temperature = 1.0f);
    
    // Get visit count distribution (for training)
    std::vector<float> getActionProbabilities(float temperature = 1.0f) const;
    
    // Get the value estimate of the root node
    float getRootValue() const;
    
    // Update tree with a move (keep subtree for the selected action)
    void updateWithMove(int action);
    
    // Add Dirichlet noise to root for exploration
    void addDirichletNoise(float alpha = 0.03f, float epsilon = 0.25f);
    
    // Set parameters
    void setNumThreads(int numThreads);
    void setNumSimulations(int numSimulations);
    void setCPuct(float cPuct) { cPuct_ = cPuct; }
    void setFpuReduction(float fpuReduction) { fpuReduction_ = fpuReduction; }
    void setVirtualLoss(int virtualLoss) { virtualLoss_ = virtualLoss; }
    
    // Set the neural network to use
    void setNeuralNetwork(NeuralNetwork* nn);
    
    // Set transposition table
    void setTranspositionTable(TranspositionTable* tt);
    
    // Set selection strategy
    void setSelectionStrategy(MCTSNodeSelection strategy) { selectionStrategy_ = strategy; }
    
    // Set deterministic mode for ELO calculation
    void setDeterministicMode(bool enable);
    
    // Debug utilities
    void setDebugMode(bool debug) { debugMode_ = debug; }
    void printSearchStats() const;
    void printSearchPath(int action) const;
    std::string getSearchInfo() const;
    
    // Set progress callback
    void setProgressCallback(std::function<void(int, int)> callback) {
        progressCallback_ = std::move(callback);
    }
    
    // Memory management
    size_t getMemoryUsage() const;
    void releaseMemory();
    
private:
    // Member variables
    std::unique_ptr<IGameState> rootState_;     // Current root state
    std::unique_ptr<MCTSNode> rootNode_;        // Root of the search tree
    NeuralNetwork* nn_;                         // Neural network for policy and value
    TranspositionTable* tt_;                    // Optional transposition table
    ThreadPool threadPool_;                     // Thread pool for parallel search
    std::atomic<int> pendingSimulations_;       // Counter for remaining simulations
    
    // MCTS parameters
    int numSimulations_;
    float cPuct_;
    float fpuReduction_;
    int virtualLoss_;
    bool deterministicMode_;                     // For ELO evaluation
    MCTSNodeSelection selectionStrategy_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniformDist_;
    
    // Debugging
    bool debugMode_;
    
    // Progress tracking
    std::function<void(int, int)> progressCallback_;
    
    // Synchronization
    std::mutex searchMutex_;
    std::condition_variable searchCondVar_;
    std::atomic<bool> searchInProgress_{false};
    
    // MCTS algorithm steps
    void runSingleSimulation();
    MCTSNode* selectLeaf(IGameState& state);
    void expandNode(MCTSNode* node, const IGameState& state);
    void backpropagate(MCTSNode* node, float value);
    
    // Helper methods
    std::pair<std::vector<float>, float> evaluateState(const IGameState& state);
    float getTemperatureVisitWeight(int visitCount, float temperature) const;
    
    // Node selection using different formulas
    MCTSNode* selectChildUcb(MCTSNode* node, const IGameState& state);
    MCTSNode* selectChildPuct(MCTSNode* node, const IGameState& state);
    MCTSNode* selectChildProgressiveBias(MCTSNode* node, const IGameState& state);
    
    // Game-specific Dirichlet noise parameters
    float getDirichletAlpha() const;
    
    // Memory management helpers
    void pruneAllExcept(MCTSNode* nodeToKeep);
};

#endif // PARALLEL_MCTS_H
```

### 8.3 Transposition Table Implementation

```cpp
// transposition_table.h
#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <memory>
#include <chrono>
#include <unordered_map>
#include "igamestate.h"

class TranspositionTable {
public:
    // Entry in the transposition table
    struct Entry {
        uint64_t hash;                 // Zobrist hash of the position
        GameType gameType;            // Game type this entry belongs to
        std::vector<float> policy;     // Policy vector from neural network
        float value;                   // Value estimate from neural network
        std::atomic<int> visitCount;  // Number of times this entry was used
        std::atomic<uint64_t> lastAccessTime;  // For aging/replacement policy
        std::atomic<bool> isValid;    // Whether this entry contains valid data
        
        Entry() : hash(0), gameType(GameType::GOMOKU), value(0.0f), 
                  visitCount(0), lastAccessTime(0), isValid(false) {}
    };
    
    // Constructor
    explicit TranspositionTable(size_t size = 1048576, size_t numShards = 1024);
    
    // Destructor
    ~TranspositionTable() = default;
    
    // Non-copyable but movable
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;
    TranspositionTable(TranspositionTable&&) noexcept = default;
    TranspositionTable& operator=(TranspositionTable&&) noexcept = default;
    
    // Look up an entry in the table
    bool lookup(uint64_t hash, GameType gameType, Entry& result) const;
    
    // Store an entry in the table
    void store(uint64_t hash, GameType gameType, const Entry& entry);
    
    // Store simplified version
    void store(uint64_t hash, GameType gameType, const std::vector<float>& policy, float value);
    
    // Clear the table
    void clear();
    
    // Set the replacement policy parameters
    void setReplacementPolicy(uint64_t maxAgeMs, int minVisitsThreshold);
    
    // Get stats
    size_t getSize() const { return size_; }
    float getHitRate() const { return lookups_ > 0 ? static_cast<float>(hits_) / lookups_ : 0.0f; }
    size_t getLookups() const { return lookups_; }
    size_t getHits() const { return hits_; }
    size_t getEntryCount() const;
    size_t getMemoryUsageBytes() const;
    
    // Optimize table size
    void resize(size_t newSize);
    
private:
    // Table data
    std::vector<Entry> table_;
    size_t size_;
    size_t sizeMask_;  // For fast modulo with power of 2
    
    // Sharding for reduced lock contention
    size_t numShards_;
    mutable std::vector<std::mutex> mutexShards_;
    
    // Replacement policy
    uint64_t maxAge_ = 60000;  // Maximum age in milliseconds (1 minute)
    int minVisits_ = 5;        // Minimum visits before replacement
    
    // Stats
    mutable std::atomic<size_t> lookups_{0};
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> collisions_{0};
    mutable std::atomic<size_t> replacements_{0};
    
    // Helper to get current time in milliseconds
    uint64_t getCurrentTime() const;
    
    // Helper to calculate hash index
    size_t getHashIndex(uint64_t hash) const { return hash & sizeMask_; }
    
    // Helper to calculate shard index
    size_t getShardIndex(uint64_t hash) const { return hash % numShards_; }
};

#endif // TRANSPOSITION_TABLE_H
```

## 9. ELO Rating System

### 9.1 ELO Tracker Interface

```cpp
// elo_tracker.h
#ifndef ELO_TRACKER_H
#define ELO_TRACKER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <mutex>
#include <memory>
#include <functional>
#include "igamestate.h"
#include "neural_network.h"
#include "parallel_mcts.h"

// Forward declarations
class IGameState;
class NeuralNetwork;
class ParallelMCTS;
class ThreadPool;

class EloTracker {
public:
    // Result of a match
    struct MatchResult {
        std::string player1;
        std::string player2;
        float score;  // 1.0 = win for player1, 0.0 = win for player2, 0.5 = draw
        GameType gameType;
        std::string timestamp;
        
        MatchResult(std::string p1, std::string p2, float s, GameType gt, std::string ts = "")
            : player1(std::move(p1)), player2(std::move(p2)), 
              score(s), gameType(gt), timestamp(std::move(ts)) {}
    };
    
    // ELO rating entry
    struct RatingEntry {
        float rating;
        int games;
        float winRate;
        std::string lastUpdated;
        
        RatingEntry(float r = 1500.0f, int g = 0, float wr = 0.0f, std::string lu = "")
            : rating(r), games(g), winRate(wr), lastUpdated(std::move(lu)) {}
    };
    
    // Constructor
    explicit EloTracker(float initialElo = 1500.0f, float kFactor = 32.0f);
    
    // Destructor
    ~EloTracker() = default;
    
    // Non-copyable but movable
    EloTracker(const EloTracker&) = delete;
    EloTracker& operator=(const EloTracker&) = delete;
    EloTracker(EloTracker&&) noexcept = default;
    EloTracker& operator=(EloTracker&&) noexcept = default;
    
    // Add a match result
    void addResult(const MatchResult& result);
    void addResult(const std::string& player1, const std::string& player2, 
                 float result, GameType gameType = GameType::GOMOKU);
    
    // Update ELO ratings based on match results
    void updateRatings();
    
    // Get ELO rating for a player
    RatingEntry getRating(const std::string& player, GameType gameType = GameType::GOMOKU);
    
    // Get all ratings
    std::unordered_map<std::string, RatingEntry> getAllRatings(GameType gameType = GameType::GOMOKU) const;
    
    // Save/load ratings to/from file
    bool saveRatings(const std::string& filename) const;
    bool loadRatings(const std::string& filename);
    
    // Run arena matches between two players
    float runArenaMatches(
        const std::string& player1Name, NeuralNetwork* player1Nn,
        const std::string& player2Name, NeuralNetwork* player2Nn,
        int numGames, int numSimulations, int numThreads,
        GameType gameType, int boardSize = 0, bool variantRules = false,
        bool verbose = true
    );
    
    // Run arena matches against fixed MCTS with random policy
    float benchmarkAgainstFixedMcts(
        const std::string& playerName, NeuralNetwork* playerNn,
        int numGames, int numSimulations, int numThreads,
        GameType gameType, int boardSize = 0, bool variantRules = false,
        bool verbose = true
    );
    
    // Set deterministic mode for ELO calculation
    void setDeterministicMode(bool enable) { deterministicMode_ = enable; }
    
    // Set progress callback
    void setProgressCallback(std::function<void(int, int, const MatchResult&)> callback) {
        progressCallback_ = std::move(callback);
    }
    
    // Get match history
    const std::vector<MatchResult>& getMatchHistory() const { return matchHistory_; }
    
    // Clear history and ratings
    void clear();
    
    // Get per-game ratings
    std::unordered_map<std::string, RatingEntry> getGameSpecificRatings(GameType gameType) const;
    
private:
    std::unordered_map<GameType, std::unordered_map<std::string, RatingEntry>> ratings_;
    std::vector<MatchResult> pendingResults_;
    std::vector<MatchResult> matchHistory_;
    float kFactor_;
    bool deterministicMode_; // For reproducible ELO calculation
    std::unique_ptr<ThreadPool> threadPool_;
    std::mutex ratingsMutex_;
    std::function<void(int, int, const MatchResult&)> progressCallback_;
    
    // Expected score calculation
    float expectedScore(float rating1, float rating2);
    
    // Play a single game between two MCTS engines
    MatchResult playSingleGame(
        ParallelMCTS* mcts1, ParallelMCTS* mcts2,
        std::unique_ptr<IGameState>& state, 
        bool player1PlaysFirst,
        const std::string& player1Name,
        const std::string& player2Name
    );
    
    // Get current timestamp
    std::string getCurrentTimestamp() const;
    
    // Update a specific rating
    void updateRating(const std::string& player, GameType gameType, float ratingChange, 
                     float winRate, int games);
};

// Random policy network for benchmarking
class RandomPolicyNetwork : public NeuralNetwork {
public:
    RandomPolicyNetwork(GameType gameType, int boardSize = 0, unsigned int seed = 0);
    
    std::pair<std::vector<float>, float> predict(const IGameState& state) override;
    
    void predictBatch(
        const std::vector<std::reference_wrapper<const IGameState>>& states,
        std::vector<std::vector<float>>& policies,
        std::vector<float>& values
    ) override;
    
    std::future<std::pair<std::vector<float>, float>> predictAsync(
        const IGameState& state
    ) override;
    
    bool isGpuAvailable() const override { return false; }
    std::string getDeviceInfo() const override { return "CPU (Random)"; }
    float getInferenceTimeMs() const override { return 0.1f; }
    int getBatchSize() const override { return 128; }
    
    std::string getModelInfo() const override { return "Random policy network"; }
    size_t getModelSizeBytes() const override { return 0; }
    
    void benchmark(int numIterations = 100, int batchSize = 16) override { /* No-op */ }
    
    void enableDebugMode(bool enable) override { /* No-op */ }
    void printModelSummary() const override { /* No-op */ }
    
private:
    GameType gameType_;
    int boardSize_;
    std::mt19937 rng_;
};

#endif // ELO_TRACKER_H
```

### 9.2 ELO Rating Calculation Implementation

```cpp
// elo_tracker.cpp (excerpt)
float EloTracker::expectedScore(float rating1, float rating2) {
    // Standard ELO formula for expected score
    return 1.0f / (1.0f + std::pow(10.0f, (rating2 - rating1) / 400.0f));
}

void EloTracker::updateRatings() {
    std::lock_guard<std::mutex> lock(ratingsMutex_);
    
    for (const auto& result : pendingResults_) {
        auto& gameRatings = ratings_[result.gameType];
        
        // Initialize ratings if players don't exist for this game type
        if (gameRatings.find(result.player1) == gameRatings.end()) {
            gameRatings[result.player1] = RatingEntry();
        }
        if (gameRatings.find(result.player2) == gameRatings.end()) {
            gameRatings[result.player2] = RatingEntry();
        }
        
        float r1 = gameRatings[result.player1].rating;
        float r2 = gameRatings[result.player2].rating;
        
        float expected = expectedScore(r1, r2);
        float ratingChange = kFactor_ * (result.score - expected);
        
        // Update player1 stats
        float winRate1 = gameRatings[result.player1].winRate;
        int games1 = gameRatings[result.player1].games;
        winRate1 = (winRate1 * games1 + result.score) / (games1 + 1);
        games1 += 1;
        
        // Update player2 stats
        float winRate2 = gameRatings[result.player2].winRate;
        int games2 = gameRatings[result.player2].games;
        winRate2 = (winRate2 * games2 + (1.0f - result.score)) / (games2 + 1);
        games2 += 1;
        
        // Apply rating changes
        updateRating(result.player1, result.gameType, ratingChange, winRate1, games1);
        updateRating(result.player2, result.gameType, -ratingChange, winRate2, games2);
        
        // Add to match history
        matchHistory_.push_back(result);
    }
    
    pendingResults_.clear();
}

void EloTracker::updateRating(const std::string& player, GameType gameType, 
                             float ratingChange, float winRate, int games) {
    auto& entry = ratings_[gameType][player];
    entry.rating += ratingChange;
    entry.winRate = winRate;
    entry.games = games;
    entry.lastUpdated = getCurrentTimestamp();
}
```

## 10. Self-Play Manager for Multi-Game Training

```cpp
// self_play_manager.h
#ifndef SELF_PLAY_MANAGER_H
#define SELF_PLAY_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <random>
#include "igamestate.h"
#include "neural_network.h"
#include "game_record.h"
#include "thread_pool.h"

class ParallelMCTS;
class TranspositionTable;

class SelfPlayManager {
public:
    // Constructor
    SelfPlayManager(
        NeuralNetwork* nn,
        GameType gameType,
        int numGames = 100,
        int numSimulations = 800,
        int numThreads = 4,
        int boardSize = 0,  // 0 means use default for game type
        bool variantRules = false,
        float temperature = 1.0f,
        const std::string& outputDir = "data",
        bool debugMode = false
    );
    
    // Destructor
    ~SelfPlayManager();
    
    // Non-copyable but movable
    SelfPlayManager(const SelfPlayManager&) = delete;
    SelfPlayManager& operator=(const SelfPlayManager&) = delete;
    SelfPlayManager(SelfPlayManager&&) noexcept = default;
    SelfPlayManager& operator=(SelfPlayManager&&) noexcept = default;
    
    // Generate games and save to disk
    std::vector<std::unique_ptr<GameRecord>> generateGames();
    
    // Set a progress callback
    void setProgressCallback(std::function<void(int, int, const GameRecord*)> callback) {
        progressCallback_ = std::move(callback);
    }
    
    // Temperature schedule for move selection
    void setTemperatureSchedule(float initialTemp, float finalTemp, int annealingMove);
    
    // Resignation threshold
    void enableResign(float threshold = -0.9f, int minMoves = 30);
    
    // Dirichlet noise at root for exploration
    void enableDirichletNoise(float alpha = 0.03f, float epsilon = 0.25f);
    
    // Options
    void setNumThreads(int threads) { numThreadsPerGame_ = threads; }
    void setMaxGameLength(int maxLength) { maxGameLength_ = maxLength; }
    void enableOpeningBook(bool enable) { useOpeningBook_ = enable; }
    void setTranspositionTable(TranspositionTable* tt) { transpositionTable_ = tt; }
    void setNumGames(int numGames) { numGames_ = numGames; }
    void setNumSimulations(int numSims) { numSimulations_ = numSims; }
    void setOutputDir(const std::string& dir) { outputDir_ = dir; }
    
    // Stats
    std::string getStatsString() const;
    
    // Stop generation
    void stop();
    
private:
    // Neural network and parameters
    NeuralNetwork* nn_;
    GameType gameType_;
    int numGames_;
    int numSimulations_;
    int numThreadsPerGame_;
    int boardSize_;
    bool variantRules_;
    float initialTemperature_;
    std::string outputDir_;
    bool debugMode_;
    TranspositionTable* transpositionTable_;
    
    // MCTS settings
    float cPuct_ = 1.5f;
    float fpuReduction_ = 0.0f;
    int virtualLoss_ = 3;
    
    // Temperature settings
    float tempFinal_;
    int tempAnnealingMove_;
    
    // Resignation settings
    bool allowResign_;
    float resignThreshold_;
    int minMovesBeforeResign_;
    
    // Exploration noise
    bool addDirichlet_;
    float dirichletAlpha_;
    float dirichletEpsilon_;
    
    // Opening book
    bool useOpeningBook_;
    
    // Maximum game length
    int maxGameLength_;
    
    // Callback for reporting progress
    std::function<void(int, int, const GameRecord*)> progressCallback_;
    
    // Thread pool for parallel game generation
    ThreadPool threadPool_;
    
    // Synchronization
    std::mutex fileMutex_;
    std::atomic<int> gamesCompleted_{0};
    std::atomic<bool> stopRequested_{false};
    
    // Play a single game
    std::unique_ptr<GameRecord> playSingleGame(int gameId);
    
    // Save a game to disk
    bool saveGame(const GameRecord& record, int gameId);
    
    // Get temperature for the given move number
    float getTemperature(int moveNumber) const;
    
    // Get game-specific parameters
    float getGameSpecificDirichletAlpha() const;
    int getDefaultMaxGameLength() const;
    
    // Random number generation
    std::mt19937 rng_;
};

#endif // SELF_PLAY_MANAGER_H
```

## 11. Game Record and Serialization

```cpp
// game_record.h
#ifndef GAME_RECORD_H
#define GAME_RECORD_H

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <nlohmann/json.hpp>
#include "igamestate.h"

// Forward declaration
class IGameState;

struct GameRecord {
    // Game metadata
    GameType gameType;
    int boardSize;
    bool variantRules;
    std::string timestamp;
    std::string modelVersion;
    
    // Game result
    GameResult result;
    float value;  // Final value from the network
    
    // Move data
    struct MoveData {
        int action;
        int player;
        std::vector<float> policy;  // MCTS policy (probabilities)
        float value;  // Value prediction
        int visitCount;
        std::chrono::milliseconds thinkingTime;
        
        // Serialization
        nlohmann::json toJson() const;
        static MoveData fromJson(const nlohmann::json& json);
    };
    
    std::vector<MoveData> moves;
    
    // Statistics
    int numSimulations;
    float avgMovesPerGame;
    
    // Constructors
    GameRecord();
    GameRecord(GameType type, int size, bool variant, const std::string& model);
    
    // Add a move
    void addMove(int action, int player, const std::vector<float>& policy, 
                float value, int visitCount, 
                std::chrono::milliseconds thinkingTime = std::chrono::milliseconds(0));
    
    // Set result
    void setResult(GameResult gameResult, float finalValue);
    
    // Serialization
    nlohmann::json toJson() const;
    static std::unique_ptr<GameRecord> fromJson(const nlohmann::json& json);
    
    // Load/save to file
    bool saveToFile(const std::string& filename) const;
    static std::unique_ptr<GameRecord> loadFromFile(const std::string& filename);
    
    // Validation
    bool validate() const;
    
    // Recreate the game state at a specific move
    std::unique_ptr<IGameState> reconstructGameState(int moveIndex = -1) const;
    
    // Get statistics
    std::string getStats() const;
};

// Data storage format for training
class GameRecordDataset {
public:
    // Add a game record
    void addGameRecord(const GameRecord& record);
    
    // Get training examples
    std::vector<std::pair<std::vector<std::vector<std::vector<float>>>, std::pair<std::vector<float>, float>>>
    getTrainingExamples(bool shuffle = true, int maxExamples = -1);
    
    // Save/load to file
    bool saveToFile(const std::string& filename) const;
    static std::unique_ptr<GameRecordDataset> loadFromFile(const std::string& filename);
    
    // Get size
    size_t size() const { return examples_.size(); }
    
private:
    struct TrainingExample {
        std::vector<std::vector<std::vector<float>>> boardState;
        std::vector<float> policy;
        float value;
        
        // Serialization
        nlohmann::json toJson() const;
        static TrainingExample fromJson(const nlohmann::json& json);
    };
    
    std::vector<TrainingExample> examples_;
};

#endif // GAME_RECORD_H
```

## 12. Python Bindings with PyBind11

### 12.1 Python Interface Design

```cpp
// python_bindings.h
#ifndef PYTHON_BINDINGS_H
#define PYTHON_BINDINGS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <memory>
#include <vector>

namespace py = pybind11;

// Forward declarations
class IGameState;
class NeuralNetwork;
class ParallelMCTS;
class EloTracker;
class SelfPlayManager;
class GameRecord;
class TranspositionTable;

// Module initialization function
void initGameAbstractionModule(py::module& m);
void initNeuralNetworkModule(py::module& m);
void initMCTSModule(py::module& m);
void initEloModule(py::module& m);
void initSelfPlayModule(py::module& m);
void initGameRecordModule(py::module& m);
void initUtilityModule(py::module& m);

// Helper functions for conversion between C++ and Python
py::array_t<float> tensorToNumpy(const std::vector<std::vector<std::vector<float>>>& tensor);
std::vector<std::vector<std::vector<float>>> numpyToTensor(const py::array_t<float>& array);

// Memory management utilities
void releaseGIL(const std::function<void()>& callback);
std::unique_ptr<IGameState> cloneState(const IGameState& state);

// Exception translation
void registerExceptions(py::module& m);

#endif // PYTHON_BINDINGS_H
```

### 11.2 PyBind11 Implementation with Memory Management

```cpp
// python_bindings.cpp (partial implementation)
#include "python_bindings.h"
#include "igamestate.h"
#include "gomoku_state.h"
#include "chess_state.h"
#include "go_state.h"
#include "neural_network.h"
#include "torch_neural_network.h"
#include "mcts_node.h"
#include "parallel_mcts.h"
#include "elo_tracker.h"
#include "self_play_manager.h"
#include "game_record.h"
#include "transposition_table.h"

PYBIND11_MODULE(alphazero_multi_game, m) {
    m.doc() = "AlphaZero-style Multi-Game AI Engine with C++ backend";
    
    // Version info
    m.attr("__version__") = "1.0.0";
    
    // Register exceptions
    registerExceptions(m);
    
    // Initialize all modules
    initGameAbstractionModule(m);
    initNeuralNetworkModule(m);
    initMCTSModule(m);
    initEloModule(m);
    initSelfPlayModule(m);
    initGameRecordModule(m);
    initUtilityModule(m);
}

void registerExceptions(py::module& m) {
    // Register C++ exceptions to be translated to Python exceptions
    py::register_exception<GameStateException>(m, "GameStateException");
    py::register_exception<IllegalMoveException>(m, "IllegalMoveException");
}

void initGameAbstractionModule(py::module& m) {
    // Game type enum
    py::enum_<GameType>(m, "GameType")
        .value("GOMOKU", GameType::GOMOKU)
        .value("CHESS", GameType::CHESS)
        .value("GO", GameType::GO)
        .export_values();
    
    // Game result enum
    py::enum_<GameResult>(m, "GameResult")
        .value("ONGOING", GameResult::ONGOING)
        .value("DRAW", GameResult::DRAW)
        .value("WIN_PLAYER1", GameResult::WIN_PLAYER1)
        .value("WIN_PLAYER2", GameResult::WIN_PLAYER2)
        .export_values();
    
    // Game state interface
    py::class_<IGameState, std::unique_ptr<IGameState>>(m, "IGameState")
        .def("getLegalMoves", &IGameState::getLegalMoves)
        .def("isLegalMove", &IGameState::isLegalMove)
        .def("makeMove", &IGameState::makeMove)
        .def("undoMove", &IGameState::undoMove)
        .def("isTerminal", &IGameState::isTerminal)
        .def("getGameResult", &IGameState::getGameResult)
        .def("getCurrentPlayer", &IGameState::getCurrentPlayer)
        .def("getBoardSize", &IGameState::getBoardSize)
        .def("getActionSpaceSize", &IGameState::getActionSpaceSize)
        .def("getTensorRepresentation", [](const IGameState& state) {
            return tensorToNumpy(state.getTensorRepresentation());
        })
        .def("getEnhancedTensorRepresentation", [](const IGameState& state) {
            return tensorToNumpy(state.getEnhancedTensorRepresentation());
        })
        .def("getHash", &IGameState::getHash)
        .def("clone", [](const IGameState& state) {
            return cloneState(state);
        })
        .def("actionToString", &IGameState::actionToString)
        .def("stringToAction", [](const IGameState& state, const std::string& moveStr) {
            auto action = state.stringToAction(moveStr);
            if (action.has_value()) {
                return py::cast(action.value());
            }
            return py::none();
        })
        .def("getGameType", &IGameState::getGameType)
        .def("getMoveHistory", &IGameState::getMoveHistory)
        .def("validate", &IGameState::validate)
        .def("__str__", &IGameState::toString);
    
    // Factory function for game states
    m.def("createGameState", &createGameState,
          py::arg("gameType"),
          py::arg("boardSize") = 0,
          py::arg("variantRules") = false,
          py::return_value_policy::take_ownership);
    
    // Game-specific state classes
    py::class_<GomokuState, IGameState, std::unique_ptr<GomokuState>>(m, "GomokuState")
        .def(py::init<int, bool>(),
             py::arg("boardSize") = GomokuState::DEFAULT_BOARD_SIZE,
             py::arg("useRenjuRules") = false)
        .def("getBoard", &GomokuState::getBoard)
        .def("positionToXY", &GomokuState::positionToXY)
        .def("xyToPosition", &GomokuState::xyToPosition)
        .def("usingRenjuRules", &GomokuState::usingRenjuRules)
        .def("isColor", &GomokuState::isColor,
             py::arg("position"), py::arg("color"));
}

void initNeuralNetworkModule(py::module& m) {
    // Neural network interface
    py::class_<NeuralNetwork, std::shared_ptr<NeuralNetwork>>(m, "NeuralNetwork")
        .def("predict", [](NeuralNetwork& nn, const IGameState& state) {
            py::gil_scoped_release release;
            return nn.predict(state);
        })
        .def("predictBatch", [](NeuralNetwork& nn,
                              const std::vector<std::reference_wrapper<const IGameState>>& states) {
            std::vector<std::vector<float>> policies;
            std::vector<float> values;
            {
                py::gil_scoped_release release;
                nn.predictBatch(states, policies, values);
            }
            return std::make_pair(policies, values);
        })
        .def("predictAsync", [](NeuralNetwork& nn, const IGameState& state) {
            return nn.predictAsync(state);
        })
        .def("isGpuAvailable", &NeuralNetwork::isGpuAvailable)
        .def("getDeviceInfo", &NeuralNetwork::getDeviceInfo)
        .def("getInferenceTimeMs", &NeuralNetwork::getInferenceTimeMs)
        .def("getBatchSize", &NeuralNetwork::getBatchSize)
        .def("getModelInfo", &NeuralNetwork::getModelInfo)
        .def("getModelSizeBytes", &NeuralNetwork::getModelSizeBytes)
        .def("benchmark", &NeuralNetwork::benchmark,
             py::arg("numIterations") = 100, py::arg("batchSize") = 16)
        .def("enableDebugMode", &NeuralNetwork::enableDebugMode)
        .def("printModelSummary", &NeuralNetwork::printModelSummary);
    
    // Factory method for neural network
    m.def("createNeuralNetwork", &NeuralNetwork::create,
          py::arg("modelPath"),
          py::arg("gameType"),
          py::arg("boardSize") = 0,
          py::arg("useGpu") = true,
          py::return_value_policy::take_ownership);
    
    // PyTorch neural network implementation
    py::class_<TorchNeuralNetwork, NeuralNetwork, std::shared_ptr<TorchNeuralNetwork>>(m, "TorchNeuralNetwork")
        .def(py::init<const std::string&, GameType, int, bool>(),
             py::arg("modelPath"),
             py::arg("gameType"),
             py::arg("boardSize") = 0,
             py::arg("useGpu") = true)
        .def("setMaxBatchSize", &TorchNeuralNetwork::setMaxBatchSize)
        .def("setFp16Mode", &TorchNeuralNetwork::setFp16Mode)
        .def("optimizeForInference", &TorchNeuralNetwork::optimizeForInference);
    
    // Random policy network for benchmarking
    py::class_<RandomPolicyNetwork, NeuralNetwork, std::shared_ptr<RandomPolicyNetwork>>(m, "RandomPolicyNetwork")
        .def(py::init<GameType, int, unsigned int>(),
             py::arg("gameType"),
             py::arg("boardSize") = 0,
             py::arg("seed") = 0);
}

// Helper functions for memory management
std::unique_ptr<IGameState> cloneState(const IGameState& state) {
    py::gil_scoped_release release;
    return state.clone();
}

void releaseGIL(const std::function<void()>& callback) {
    py::gil_scoped_release release;
    callback();
}

// Conversion between C++ tensors and NumPy arrays
py::array_t<float> tensorToNumpy(const std::vector<std::vector<std::vector<float>>>& tensor) {
    if (tensor.empty() || tensor[0].empty() || tensor[0][0].empty()) {
        return py::array_t<float>();
    }
    
    size_t depth = tensor.size();
    size_t height = tensor[0].size();
    size_t width = tensor[0][0].size();
    
    py::array_t<float> result({depth, height, width});
    py::buffer_info buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    for (size_t d = 0; d < depth; d++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                ptr[d * height * width + h * width + w] = tensor[d][h][w];
            }
        }
    }
    
    return result;
}

std::vector<std::vector<std::vector<float>>> numpyToTensor(const py::array_t<float>& array) {
    py::buffer_info buf = array.request();
    
    if (buf.ndim != 3) {
        throw std::runtime_error("NumPy array must have 3 dimensions");
    }
    
    float* ptr = static_cast<float*>(buf.ptr);
    size_t depth = buf.shape[0];
    size_t height = buf.shape[1];
    size_t width = buf.shape[2];
    
    std::vector<std::vector<std::vector<float>>> tensor(depth,
        std::vector<std::vector<float>>(height,
            std::vector<float>(width, 0.0f)));
    
    for (size_t d = 0; d < depth; d++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                tensor[d][h][w] = ptr[d * height * width + h * width + w];
            }
        }
    }
    
    return tensor;
}
```

### 11.3 Python Training Script

```python
# train.py
import os
import time
import argparse
import logging
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import alphazero_multi_game as az
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alphazero-training")

# Set random seeds for reproducibility
def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# DDW-RandWire neural network architecture
class DDWRandWireNetwork(nn.Module):
    def __init__(self, 
                 game_type: az.GameType,
                 board_size: int,
                 num_input_channels: int = 20,
                 num_nodes: int = 40,
                 num_channels: int = 144,
                 avg_out_degree: int = 6,
                 l2_reg: float = 1e-4):
        super(DDWRandWireNetwork, self).__init__()
        
        self.game_type = game_type
        self.board_size = board_size
        self.action_size = self._get_action_size(game_type, board_size)
        self.l2_reg = l2_reg
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        
        # Stem: 3×3 conv (num_input_channels→144), BN, ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Create DDW-RandWire backbone
        # Create residual blocks for each node
        self.node_blocks = nn.ModuleList([
            self._build_residual_block(num_channels) for _ in range(num_nodes)
        ])
        
        # Create routers for dynamic wiring
        self.routers = nn.ModuleList([
            self._build_se_router(num_channels) for _ in range(num_nodes)
        ])
        
        # Generate random graph topology
        self.connections = self._generate_random_graph(
            num_nodes, avg_out_degree, use_small_world=random.random() < 0.5
        )
        
        # Policy head: 1×1 conv (144→1) → reshape
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1),
            nn.Flatten(),
            nn.Linear(board_size * board_size, self.action_size)
        )
        
        # Value head: GlobalAvgPool→1×1 conv (144→144) → ReLU → 1×1 conv (144→1) → tanh
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_channels, num_channels, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_channels, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _get_action_size(self, game_type: az.GameType, board_size: int) -> int:
        # Create a temporary game state to get action space size
        state = az.createGameState(game_type, board_size)
        return state.getActionSpaceSize()
    
    def _build_residual_block(self, num_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
    
    def _build_se_router(self, num_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_channels, num_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(num_channels // 16, num_channels, 1),
            nn.Sigmoid()
        )
    
    def _generate_random_graph(self, nodes: int, avg_out_degree: int, use_small_world: bool) -> List[Tuple[int, int]]:
        connections = []
        
        if use_small_world:
            # Small-world network generation (Watts-Strogatz inspired)
            k = avg_out_degree
            rewire_prob = 0.1
            
            # Create regular ring lattice
            for i in range(nodes):
                for j in range(1, k // 2 + 1):
                    target = (i + j) % nodes
                    connections.append((i, target))
                    connections.append((i, (i - j + nodes) % nodes))
            
            # Rewire some connections randomly
            for i, (src, dst) in enumerate(connections):
                if random.random() < rewire_prob:
                    new_target = random.randint(0, nodes - 1)
                    while new_target == src:
                        new_target = random.randint(0, nodes - 1)
                    connections[i] = (src, new_target)
        else:
            # Scale-free network (Barabási–Albert inspired)
            degrees = [0] * nodes
            
            # Start with a small fully connected network
            initial_nodes = min(5, nodes)
            for i in range(initial_nodes):
                for j in range(i + 1, initial_nodes):
                    connections.append((i, j))
                    degrees[i] += 1
                    degrees[j] += 1
            
            # Add remaining nodes with preferential attachment
            for i in range(initial_nodes, nodes):
                edges_to_add = avg_out_degree
                for _ in range(edges_to_add):
                    # Preferential attachment - nodes with higher degrees
                    # are more likely to get new connections
                    targets = list(range(i))
                    weights = [degrees[t] + 1 for t in targets]  # +1 to avoid zero prob
                    target = random.choices(targets, weights=weights)[0]
                    
                    connections.append((i, target))
                    degrees[i] += 1
                    degrees[target] += 1
        
        return connections
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [batch_size, channels, board_size, board_size]
        x = self.stem(x)
        
        # Process through DDW-RandWire backbone
        node_outputs = [x]  # Start with stem output as first node
        
        # Process each node
        for i in range(1, self.num_nodes):
            # Find all incoming connections to this node
            incoming = [src for src, dst in self.connections if dst == i]
            
            if not incoming:
                # No incoming connections, use previous node as input
                node_input = node_outputs[i-1]
            else:
                # Combine all incoming connections
                node_input = sum(node_outputs[src] for src in incoming) / len(incoming)
            
            # Apply residual block
            residual = node_input
            node_output = self.node_blocks[i](node_input)
            node_output = node_output + residual
            node_output = F.relu(node_output)
            
            # Apply router (SE-style gating)
            router_weights = self.routers[i](node_output)
            node_output = node_output * router_weights
            
            node_outputs.append(node_output)
        
        # Find terminal nodes (those without outgoing connections)
        outgoing_nodes = set(src for src, _ in self.connections)
        terminal_nodes = [i for i in range(self.num_nodes) if i not in outgoing_nodes]
        
        if terminal_nodes:
            # Average outputs from all terminal nodes
            x = sum(node_outputs[i] for i in terminal_nodes) / len(terminal_nodes)
        else:
            # If no terminal nodes, use last node
            x = node_outputs[-1]
        
        # Policy head
        policy_logits = self.policy_head(x)
        
        # Value head
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_l2_regularization_loss(self) -> torch.Tensor:
        """Calculate L2 regularization loss for all weights"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss

# Dataset for AlphaZero training
class AlphaZeroDataset(Dataset):
    def __init__(self, states: List[np.ndarray], 
                policies: List[np.ndarray], 
                values: List[float],
                augment: bool = True):
        self.states = states
        self.policies = policies
        self.values = values
        self.augment = augment
        
        # Ensure all arrays are in correct format
        self.states = [np.array(s, dtype=np.float32) for s in self.states]
        self.policies = [np.array(p, dtype=np.float32) for p in self.policies]
        self.values = np.array(self.values, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.states[idx]
        policy = self.policies[idx]
        value = self.values[idx]
        
        # Data augmentation if enabled
        if self.augment:
            # Apply 8-fold symmetry for board games (rotations, reflections)
            # This depends on the game type and is primarily useful for Gomoku/Go
            # For now, we'll just implement simple horizontal flipping
            if random.random() < 0.5:
                state = np.flip(state, axis=2).copy()  # Flip horizontally
                
                # Also need to transform the policy according to the game's rules
                # This would be game-specific and requires knowledge of action representation
        
        return (torch.FloatTensor(state), 
                torch.FloatTensor(policy), 
                torch.FloatTensor([value]))

# Learning rate scheduler with warmup
class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + np.cos(np.pi * (self.last_epoch - self.warmup_epochs) / 
                              (self.max_epochs - self.warmup_epochs))) / 2
                   for base_lr in self.base_lrs]

# Training loop
def train_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               optimizer: optim.Optimizer, 
               device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    
    for states, policies, values in dataloader:
        # Move data to the device
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Forward pass
        policy_logits, value_preds = model(states)
        
        # Calculate loss
        policy_loss = nn.functional.cross_entropy(policy_logits, policies)
        value_loss = nn.functional.mse_loss(value_preds, values)
        regularization_loss = model.get_l2_regularization_loss()
        
        # Total loss
        loss = policy_loss + value_loss + regularization_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
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

# Validation loop
def validate(model: nn.Module, 
            dataloader: DataLoader, 
            device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    
    with torch.no_grad():
        for states, policies, values in dataloader:
            # Move data to the device
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            # Forward pass
            policy_logits, value_preds = model(states)
            
            # Calculate loss
            policy_loss = nn.functional.cross_entropy(policy_logits, policies)
            value_loss = nn.functional.mse_loss(value_preds, values)
            regularization_loss = model.get_l2_regularization_loss()
            
            # Total loss
            loss = policy_loss + value_loss + regularization_loss
            
            # Track statistics
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

# Self-play data generation
def generate_self_play_data(model: nn.Module, 
                          args: argparse.Namespace) -> List[az.GameRecord]:
    """Generate self-play games using the current model."""
    # Export model to TorchScript for C++ inference
    model.eval()
    example_input = torch.zeros(1, args.num_input_channels, args.board_size, args.board_size).to(args.device)
    traced_script_module = torch.jit.trace(model, example_input)
    script_model_path = os.path.join(args.model_dir, f"model_current_script.pt")
    traced_script_module.save(script_model_path)
    model.train()
    
    # Create neural network for self-play
    nn_self_play = az.createNeuralNetwork(
        script_model_path,
        args.game_type,
        args.board_size,
        args.use_gpu
    )
    
    # Configure self-play manager
    self_play_manager = az.SelfPlayManager(
        nn_self_play,
        args.game_type,
        args.self_play_games,
        args.num_simulations,
        args.num_threads,
        args.board_size,
        args.variant_rules,
        args.temperature,
        args.data_dir,
        args.debug
    )
    
    # Set self-play parameters
    self_play_manager.setTemperatureSchedule(1.0, 0.25, 30)  # Start high, then reduce
    self_play_manager.enableDirichletNoise(0.03, 0.25)  # Add noise to root for exploration
    
    # Generate games
    logger.info(f"Generating {args.self_play_games} self-play games...")
    game_records = self_play_manager.generateGames()
    logger.info(f"Generated {len(game_records)} games")
    
    return game_records

# Convert game records to training data
def process_game_records(game_records: List[az.GameRecord]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Process game records into training examples."""
    states = []
    policies = []
    values = []
    
    for record in game_records:
        # Process each move in the game
        for move_idx, move in enumerate(record.moves):
            # Reconstruct game state at this move
            state = record.reconstructGameState(move_idx)
            
            # Get state representation for neural network
            state_tensor = state.getEnhancedTensorRepresentation()
            
            # Get MCTS policy (already normalized)
            policy = move.policy
            
            # Get game outcome from this player's perspective
            if record.result == az.GameResult.DRAW:
                value = 0.0
            elif (record.result == az.GameResult.WIN_PLAYER1 and move.player == 1) or \
                 (record.result == az.GameResult.WIN_PLAYER2 and move.player == 2):
                value = 1.0
            else:
                value = -1.0
            
            # Add to training data
            states.append(state_tensor)
            policies.append(policy)
            values.append(value)
    
    return states, policies, values

# Evaluate model against a benchmark
def evaluate_model(current_model: nn.Module, 
                 best_model: nn.Module, 
                 args: argparse.Namespace) -> float:
    """Evaluate current model against best model in arena matches."""
    # Export current model to TorchScript
    current_model.eval()
    example_input = torch.zeros(1, args.num_input_channels, args.board_size, args.board_size).to(args.device)
    traced_script_module = torch.jit.trace(current_model, example_input)
    current_script_path = os.path.join(args.model_dir, f"model_current_eval.pt")
    traced_script_module.save(current_script_path)
    current_model.train()
    
    # Export best model to TorchScript if it exists
    best_script_path = os.path.join(args.model_dir, f"model_best_eval.pt")
    if best_model is not None:
        best_model.eval()
        traced_script_module = torch.jit.trace(best_model, example_input)
        traced_script_module.save(best_script_path)
    else:
        # If no best model, use random policy
        return 1.0  # Always replace if no best model
    
    # Load models in C++
    current_nn = az.createNeuralNetwork(
        current_script_path,
        args.game_type,
        args.board_size,
        args.use_gpu
    )
    
    best_nn = az.createNeuralNetwork(
        best_script_path,
        args.game_type,
        args.board_size,
        args.use_gpu
    )
    
    # Create ELO tracker
    elo_tracker = az.EloTracker(1500)
    elo_tracker.setDeterministicMode(True)  # For reproducible evaluation
    
    # Run arena matches
    logger.info(f"Running {args.eval_games} evaluation games...")
    win_rate = elo_tracker.runArenaMatches(
        "current_model", current_nn,
        "best_model", best_nn,
        args.eval_games,
        args.num_simulations,
        args.num_threads,
        args.game_type,
        args.board_size,
        args.variant_rules,
        args.debug
    )
    
    logger.info(f"Evaluation complete. Win rate: {win_rate:.3f}")
    return win_rate

# Evaluate and track ELO
def evaluate_and_track_elo(model: nn.Module, 
                         elo_tracker: az.EloTracker, 
                         args: argparse.Namespace) -> Tuple[float, float]:
    """Benchmark the current model against fixed random MCTS and track ELO progress."""
    # Export model to TorchScript for C++ inference
    model.eval()
    example_input = torch.zeros(1, args.num_input_channels, args.board_size, args.board_size).to(args.device)
    traced_script_module = torch.jit.trace(model, example_input)
    script_model_path = os.path.join(args.model_dir, f"model_current_script.pt")
    traced_script_module.save(script_model_path)
    model.train()
    
    # Load model in C++ 
    nn = az.createNeuralNetwork(
        script_model_path,
        args.game_type,
        args.board_size,
        args.use_gpu
    )
    
    # Benchmark against fixed MCTS
    win_rate = elo_tracker.benchmarkAgainstFixedMcts(
        "CurrentModel",
        nn,
        args.eval_games,
        args.num_simulations,
        args.num_threads,
        args.game_type,
        args.board_size,
        args.variant_rules,
        args.debug
    )
    
    # Get current ELO rating
    rating_entry = elo_tracker.getRating("CurrentModel", args.game_type)
    current_elo = rating_entry.rating
    
    # Log ELO progress
    logger.info(f"Current model ELO: {current_elo:.1f} (win rate vs random: {win_rate:.2f})")
    
    return current_elo, win_rate

# Plot training history
def plot_history(history: Dict[str, List], 
               elo_history: List[float], 
               win_rate_history: List[float], 
               filename: str) -> None:
    """Plot and save training history with ELO progress."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot policy and value loss
    ax2.plot(history['train_policy_loss'], label='Train Policy Loss')
    ax2.plot(history['val_policy_loss'], label='Validation Policy Loss')
    ax2.plot(history['train_value_loss'], label='Train Value Loss')
    ax2.plot(history['val_value_loss'], label='Validation Value Loss')
    ax2.set_title('Policy and Value Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot win rates
    if history['win_rates']:
        ax3.plot(history['win_rates'], marker='o')
        ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax3.set_title('Win Rate Against Previous Best Model')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Win Rate')
        ax3.set_ylim(0, 1)
        ax3.grid(True)
    
    # Add ELO rating plot
    if elo_history:
        iterations = range(len(elo_history))
        ax4.plot(iterations, elo_history, marker='o', color='green', label='ELO Rating')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(iterations, [wr * 100 for wr in win_rate_history], 
                     marker='s', color='orange', label='Win Rate %')
        ax4.set_title('ELO Rating Progress')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('ELO Rating')
        ax4_twin.set_ylabel('Win Rate %')
        ax4.grid(True)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main training loop
def training_loop(args: argparse.Namespace) -> None:
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Initialize model
    model = DDWRandWireNetwork(
        args.game_type,
        args.board_size,
        args.num_input_channels,
        args.num_nodes,
        args.num_channels,
        args.avg_out_degree,
        args.l2_reg
    )
    model = model.to(args.device)
    
    # Initialize best model to None
    best_model = None
    
    # Create ELO tracker
    elo_tracker = az.EloTracker(initial_elo=1500)
    elo_tracker.setDeterministicMode(True)  # For reproducible ELO calculation
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0  # L2 regularization handled in loss function
    )
    
    # Learning rate scheduler with warmup
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=5,
        max_epochs=args.epochs,
        eta_min=args.lr/10
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_policy_loss': [],
        'train_value_loss': [],
        'val_loss': [],
        'val_policy_loss': [],
        'val_value_loss': [],
        'win_rates': [],
        'iterations': []
    }
    
    # Keep track of ELO progress
    elo_history = []
    win_rate_history = []
    
    # Training iterations
    initial_iteration = 0
    
    # Check for existing model to resume from
    checkpoint_path = os.path.join(args.model_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path) and not args.no_resume:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        initial_iteration = checkpoint['iteration']
        history = checkpoint['history']
        elo_history = checkpoint.get('elo_history', [])
        win_rate_history = checkpoint.get('win_rate_history', [])
        
        if os.path.exists(os.path.join(args.model_dir, "best_model.pt")):
            logger.info("Loading best model")
            best_model = AlphaZeroNetwork(
                args.game_type,
                args.board_size,
                args.num_input_channels,
                args.num_blocks,
                args.num_channels,
                args.l2_reg
            ).to(args.device)
            best_model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt")))
        
        if os.path.exists('elo_ratings.csv'):
            logger.info("Loading ELO ratings")
            elo_tracker.loadRatings('elo_ratings.csv')
    
    # Main training loop
    for iteration in range(initial_iteration, args.iterations):
        logger.info(f"Starting iteration {iteration}")
        
        # 1. Generate self-play data
        start_time = time.time()
        game_records = generate_self_play_data(model, args)
        logger.info(f"Self-play took {time.time() - start_time:.2f} seconds")
        
        # 2. Process game records
        states, policies, values = process_game_records(game_records)
        logger.info(f"Processed {len(states)} training examples")
        
        # 3. Split data into training and validation
        split_idx = int(len(states) * 0.9)  # 90% for training, 10% for validation
        train_states, val_states = states[:split_idx], states[split_idx:]
        train_policies, val_policies = policies[:split_idx], policies[split_idx:]
        train_values, val_values = values[:split_idx], values[split_idx:]
        
        # 4. Create dataloaders
        train_dataset = AlphaZeroDataset(train_states, train_policies, train_values, augment=True)
        val_dataset = AlphaZeroDataset(val_states, val_policies, val_values, augment=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 5. Train for several epochs
        for epoch in range(args.epochs):
            start_time = time.time()
            
            # Training
            train_metrics = train_epoch(model, train_loader, optimizer, args.device)
            
            # Validation
            val_metrics = validate(model, val_loader, args.device)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            logger.info(
                f"Iteration {iteration}, Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                f"Time: {time.time() - start_time:.2f}s"
            )
            
            # Update history
            if epoch == args.epochs - 1:  # Only store metrics from the last epoch
                history['train_loss'].append(train_metrics['loss'])
                history['train_policy_loss'].append(train_metrics['policy_loss'])
                history['train_value_loss'].append(train_metrics['value_loss'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_policy_loss'].append(val_metrics['policy_loss'])
                history['val_value_loss'].append(val_metrics['value_loss'])
                history['iterations'].append(iteration)
        
        # 6. Evaluate model
        if best_model is not None:
            win_rate = evaluate_model(model, best_model, args)
            history['win_rates'].append(win_rate)
                            logger.info(f"Win rate against best model: {win_rate:.4f}")
            
            # Update best model if current model is better
            if win_rate >= args.replacement_threshold:
                logger.info("New best model!")
                best_model = AlphaZeroNetwork(
                    args.game_type, args.board_size, args.num_input_channels,
                    args.num_blocks, args.num_channels, args.l2_reg
                ).to(args.device)
                best_model.load_state_dict(model.state_dict())
                
                # Save best model
                torch.save(best_model.state_dict(), os.path.join(args.model_dir, "best_model.pt"))
        else:
            # First iteration - initialize best model
            best_model = AlphaZeroNetwork(
                args.game_type, args.board_size, args.num_input_channels,
                args.num_blocks, args.num_channels, args.l2_reg
            ).to(args.device)
            best_model.load_state_dict(model.state_dict())
            
            # Save best model
            torch.save(best_model.state_dict(), os.path.join(args.model_dir, "best_model.pt"))
            history['win_rates'].append(0.5)  # Placeholder for first iteration
        
        # 7. Evaluate and track ELO progress
        current_elo, win_rate = evaluate_and_track_elo(model, elo_tracker, args)
        elo_history.append(current_elo)
        win_rate_history.append(win_rate)
        
        # Save ELO ratings
        elo_tracker.saveRatings('elo_ratings.csv')
        
        # 8. Save checkpoint
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'elo_history': elo_history,
            'win_rate_history': win_rate_history,
        }, checkpoint_path)
        
        # Also save current model separately
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"model_{iteration}.pt"))
        
        # 9. Update history plot
        plot_history(history, elo_history, win_rate_history, 
                   os.path.join(args.model_dir, f"history_{iteration}.png"))
        
        logger.info(f"Completed iteration {iteration}")
    
    logger.info("Training complete!")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaZero-style agent")
    
    # Game settings
    parser.add_argument('--game', type=str, default='gomoku', 
                       choices=['gomoku', 'chess', 'go'],
                       help='Game to train on')
    parser.add_argument('--board-size', type=int, default=15,
                       help='Board size (default: 15 for Gomoku)')
    parser.add_argument('--variant-rules', action='store_true',
                       help='Use variant rules (Renju for Gomoku, Chess960 for Chess, Chinese rules for Go)')
    
    # Model parameters
    parser.add_argument('--num-nodes', type=int, default=40,
                       help='Number of nodes in the DDW-RandWire network')
    parser.add_argument('--num-channels', type=int, default=144,
                       help='Number of channels in the network')
    parser.add_argument('--avg-out-degree', type=int, default=6,
                       help='Average out-degree for nodes in DDW-RandWire network')
    parser.add_argument('--num-input-channels', type=int, default=20,
                       help='Number of input channels in the network')
    parser.add_argument('--l2-reg', type=float, default=1e-4,
                       help='L2 regularization coefficient')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per iteration')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations (self-play + training cycles)')
    parser.add_argument('--self-play-games', type=int, default=100,
                       help='Number of self-play games per iteration')
    parser.add_argument('--num-simulations', type=int, default=800,
                       help='Number of MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Initial temperature for move selection')
    parser.add_argument('--num-threads', type=int, default=4,
                       help='Number of threads for MCTS')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--replacement-threshold', type=float, default=0.55,
                       help='Win rate threshold to replace the best model')
    parser.add_argument('--eval-games', type=int, default=40,
                       help='Number of games for evaluation')
    
    # Paths and device
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory to save self-play data')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from checkpoint')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Set device
    args.use_gpu = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_gpu else 'cpu')
    
    # Convert game string to enum
    if args.game == 'gomoku':
        args.game_type = az.GameType.GOMOKU
    elif args.game == 'chess':
        args.game_type = az.GameType.CHESS
    elif args.game == 'go':
        args.game_type = az.GameType.GO
    
    # Set default board size based on game if not specified
    if args.board_size <= 0:
        if args.game == 'gomoku':
            args.board_size = 15
        elif args.game == 'chess':
            args.board_size = 8
        elif args.game == 'go':
            args.board_size = 19
    
    return args

# Entry point
if __name__ == "__main__":
    args = parse_args()
    
    # Log configuration
    logger.info(f"Training AlphaZero for {args.game} with board size {args.board_size}")
    logger.info(f"Using device: {args.device}")
    
    # Run training loop
    training_loop(args)
```

## 12. Build System with CMake

The build system uses CMake to provide cross-platform support for building the AlphaZero Multi-Game AI engine on both Linux and Windows platforms.

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
option(AZ_DEBUG_MODE "Enable debug features" OFF)
option(AZ_USE_CPU_ONLY "Force CPU-only mode even if GPU is available" OFF)
option(AZ_BUILD_GOMOKU "Build Gomoku game support" ON)
option(AZ_BUILD_CHESS "Build Chess game support" ON)
option(AZ_BUILD_GO "Build Go game support" ON)
option(AZ_CODE_COVERAGE "Enable code coverage" OFF)
option(AZ_STATIC_ANALYSIS "Enable static analysis" ON)

# Required packages
find_package(Threads REQUIRED)

# Set debug flags
if(AZ_DEBUG_MODE)
    add_definitions(-DAZ_DEBUG_MODE)
endif()

# Find packages
if(AZ_USE_TORCH)
    find_package(Torch REQUIRED)
    add_definitions(-DHAS_TORCH)
    message(STATUS "LibTorch found at: ${TORCH_INSTALL_PREFIX}")
    message(STATUS "LibTorch version: ${Torch_VERSION}")
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

# Find nlohmann_json for JSON handling
find_package(nlohmann_json REQUIRED)

# Compiler flags
if(MSVC)
    # Visual Studio compiler flags
    add_compile_options(/W4 /O2 /arch:AVX2 /fp:fast)
    if(AZ_STATIC_ANALYSIS)
        add_compile_options(/analyze)
    endif()
    
    # Add debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")
else()
    # GCC/Clang compiler flags
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -ffast-math)
    
    # Add debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    
    # Add code coverage flags if enabled
    if(AZ_CODE_COVERAGE)
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    endif()
    
    # Add sanitizers in debug mode
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
    endif()
    
    # Static analysis with clang-tidy if available
    if(AZ_STATIC_ANALYSIS)
        find_program(CLANG_TIDY_EXE clang-tidy)
        if(CLANG_TIDY_EXE)
            set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE} -checks=bugprone-*,cppcoreguidelines-*,performance-*,portability-*,readability-*)
        endif()
    endif()
endif()

# Source groups
set(CORE_SOURCES
    src/igamestate.cpp
    src/zobrist_hash.cpp
)

set(GOMOKU_SOURCES
    src/gomoku_state.cpp
)

set(CHESS_SOURCES
    src/chess_state.cpp
)

set(GO_SOURCES
    src/go_state.cpp
)

set(MCTS_SOURCES
    src/mcts_node.cpp
    src/parallel_mcts.cpp
    src/transposition_table.cpp
)

set(NN_SOURCES
    src/neural_network.cpp
    src/torch_neural_network.cpp
    src/batch_queue.cpp
    src/attack_defense_module.cpp
)

set(SELFPLAY_SOURCES
    src/game_record.cpp
    src/self_play_manager.cpp
)

set(ELO_SOURCES
    src/elo_tracker.cpp
)

set(PYTHON_SOURCES
    src/python_bindings.cpp
)

set(UI_SOURCES
    src/game_ui.cpp
    src/renderer.cpp
)

# Define include directories
include_directories(include)

# Define libraries
add_library(core ${CORE_SOURCES})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(core PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(core PUBLIC Threads::Threads)

if(AZ_BUILD_GOMOKU)
    add_library(gomoku ${GOMOKU_SOURCES})
    target_link_libraries(gomoku PUBLIC core)
    target_include_directories(gomoku PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_definitions(-DBUILD_GOMOKU)
endif()

if(AZ_BUILD_CHESS)
    add_library(chess ${CHESS_SOURCES})
    target_link_libraries(chess PUBLIC core)
    target_include_directories(chess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_definitions(-DBUILD_CHESS)
endif()

if(AZ_BUILD_GO)
    add_library(go ${GO_SOURCES})
    target_link_libraries(go PUBLIC core)
    target_include_directories(go PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_definitions(-DBUILD_GO)
endif()

add_library(mcts ${MCTS_SOURCES})
target_link_libraries(mcts PUBLIC core)
target_include_directories(mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(AZ_USE_TORCH)
    add_library(neural_net ${NN_SOURCES})
    target_link_libraries(neural_net PUBLIC 
        core 
        mcts
        ${TORCH_LIBRARIES}
    )
    target_include_directories(neural_net PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
    )
    
    add_library(elo ${ELO_SOURCES})
    target_link_libraries(elo PUBLIC
        core
        mcts
        neural_net
    )
    target_include_directories(elo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    add_library(selfplay ${SELFPLAY_SOURCES})
    target_link_libraries(selfplay PUBLIC
        core
        mcts
        neural_net
        elo
    )
    target_include_directories(selfplay PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# User interface library
add_library(ui ${UI_SOURCES})
target_link_libraries(ui PUBLIC
    core
    mcts
)
if(AZ_USE_TORCH)
    target_link_libraries(ui PUBLIC neural_net)
endif()
target_include_directories(ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Python module
if(AZ_USE_PYTHON)
    pybind11_add_module(alphazero_multi_game ${PYTHON_SOURCES})
    target_link_libraries(alphazero_multi_game PRIVATE 
        core
        mcts
        ui
    )
    target_include_directories(alphazero_multi_game PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    if(AZ_BUILD_GOMOKU)
        target_link_libraries(alphazero_multi_game PRIVATE gomoku)
    endif()
    
    if(AZ_BUILD_CHESS)
        target_link_libraries(alphazero_multi_game PRIVATE chess)
    endif()
    
    if(AZ_BUILD_GO)
        target_link_libraries(alphazero_multi_game PRIVATE go)
    endif()
    
    if(AZ_USE_TORCH)
        target_link_libraries(alphazero_multi_game PRIVATE
            neural_net
            selfplay
            elo
        )
    endif()
endif()

# CLI tool
add_executable(multi_game_cli src/cli_main.cpp)
target_link_libraries(multi_game_cli PRIVATE
    core
    mcts
    ui
)

if(AZ_BUILD_GOMOKU)
    target_link_libraries(multi_game_cli PRIVATE gomoku)
endif()

if(AZ_BUILD_CHESS)
    target_link_libraries(multi_game_cli PRIVATE chess)
endif()

if(AZ_BUILD_GO)
    target_link_libraries(multi_game_cli PRIVATE go)
endif()

if(AZ_USE_TORCH)
    target_link_libraries(multi_game_cli PRIVATE
        neural_net
        selfplay
        elo
    )
endif()

# GUI application
add_executable(multi_game_gui src/gui_main.cpp)
target_link_libraries(multi_game_gui PRIVATE
    core
    mcts
    ui
)

if(AZ_BUILD_GOMOKU)
    target_link_libraries(multi_game_gui PRIVATE gomoku)
endif()

if(AZ_BUILD_CHESS)
    target_link_libraries(multi_game_gui PRIVATE chess)
endif()

if(AZ_BUILD_GO)
    target_link_libraries(multi_game_gui PRIVATE go)
endif()

if(AZ_USE_TORCH)
    target_link_libraries(multi_game_gui PRIVATE
        neural_net
        selfplay
        elo
    )
endif()

# Tests
if(AZ_BUILD_TESTS)
    # Core tests
    add_executable(core_tests 
        tests/core_test.cpp 
        tests/game_abstraction_test.cpp
        tests/zobrist_test.cpp
    )
    target_link_libraries(core_tests PRIVATE core GTest::GTest GTest::Main)
    add_test(NAME CoreTests COMMAND core_tests)
    
    # Game-specific tests
    if(AZ_BUILD_GOMOKU)
        add_executable(gomoku_tests tests/gomoku_test.cpp)
        target_link_libraries(gomoku_tests PRIVATE core gomoku GTest::GTest GTest::Main)
        add_test(NAME GomokuTests COMMAND gomoku_tests)
    endif()
    
    if(AZ_BUILD_CHESS)
        add_executable(chess_tests tests/chess_test.cpp)
        target_link_libraries(chess_tests PRIVATE core chess GTest::GTest GTest::Main)
        add_test(NAME ChessTests COMMAND chess_tests)
    endif()
    
    if(AZ_BUILD_GO)
        add_executable(go_tests tests/go_test.cpp)
        target_link_libraries(go_tests PRIVATE core go GTest::GTest GTest::Main)
        add_test(NAME GoTests COMMAND go_tests)
    endif()
    
    # MCTS tests
    add_executable(mcts_tests 
        tests/mcts_test.cpp 
        tests/transposition_table_test.cpp
    )
    target_link_libraries(mcts_tests PRIVATE core mcts GTest::GTest GTest::Main)
    add_test(NAME MCTSTests COMMAND mcts_tests)
    
    if(AZ_USE_TORCH)
        # Neural network tests
        add_executable(nn_tests 
            tests/neural_network_test.cpp 
            tests/batch_queue_test.cpp
            tests/attack_defense_module_test.cpp
        )
        target_link_libraries(nn_tests PRIVATE 
            core 
            mcts 
            neural_net 
            GTest::GTest 
            GTest::Main
        )
        add_test(NAME NNTests COMMAND nn_tests)
        
        # ELO system tests
        add_executable(elo_tests tests/elo_tracker_test.cpp)
        target_link_libraries(elo_tests PRIVATE 
            core 
            mcts 
            neural_net
            elo
            GTest::GTest 
            GTest::Main
        )
        add_test(NAME EloTests COMMAND elo_tests)
        
        # Self-play tests
        add_executable(selfplay_tests 
            tests/self_play_test.cpp 
            tests/game_record_test.cpp
        )
        target_link_libraries(selfplay_tests PRIVATE 
            core 
            mcts 
            neural_net 
            selfplay 
            elo
            GTest::GTest 
            GTest::Main
        )
        add_test(NAME SelfPlayTests COMMAND selfplay_tests)
    endif()
    
    # UI tests
    add_executable(ui_tests tests/ui_test.cpp)
    target_link_libraries(ui_tests PRIVATE 
        core 
        mcts 
        ui 
        GTest::GTest 
        GTest::Main
    )
    add_test(NAME UITests COMMAND ui_tests)
    
    # Integration tests
    add_executable(integration_tests tests/integration_test.cpp)
    target_link_libraries(integration_tests PRIVATE 
        core 
        mcts 
        ui
    )
    if(AZ_BUILD_GOMOKU)
        target_link_libraries(integration_tests PRIVATE gomoku)
    endif()
    if(AZ_BUILD_CHESS)
        target_link_libraries(integration_tests PRIVATE chess)
    endif()
    if(AZ_BUILD_GO)
        target_link_libraries(integration_tests PRIVATE go)
    endif()
    if(AZ_USE_TORCH)
        target_link_libraries(integration_tests PRIVATE
            neural_net
            selfplay
            elo
        )
    endif()
    target_link_libraries(integration_tests PRIVATE GTest::GTest GTest::Main)
    add_test(NAME IntegrationTests COMMAND integration_tests)
endif()

# Installation
install(TARGETS core mcts ui
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

if(AZ_BUILD_GOMOKU)
    install(TARGETS gomoku
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

if(AZ_BUILD_CHESS)
    install(TARGETS chess
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

if(AZ_BUILD_GO)
    install(TARGETS go
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

if(AZ_USE_TORCH)
    install(TARGETS neural_net selfplay elo
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

install(TARGETS multi_game_cli multi_game_gui
    RUNTIME DESTINATION bin
)

if(AZ_USE_PYTHON)
    # Custom command to copy the Python module to the project directory
    add_custom_command(TARGET alphazero_multi_game POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:alphazero_multi_game>
        ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:alphazero_multi_game>
    )
    
    # Install the Python module
    install(TARGETS alphazero_multi_game
        LIBRARY DESTINATION ${Python_SITEARCH}
    )
endif()

# Install headers
install(DIRECTORY include/ DESTINATION include/alphazero_multi_game
    FILES_MATCHING PATTERN "*.h"
)

# Install configuration files
install(DIRECTORY config/ DESTINATION share/alphazero_multi_game/config)

# Install documentation
install(DIRECTORY docs/ DESTINATION share/alphazero_multi_game/docs)

# Create and install pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/alphazero_multi_game.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/alphazero_multi_game.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/alphazero_multi_game.pc
    DESTINATION lib/pkgconfig
)

# Print configuration summary
message(STATUS "AlphaZero Multi-Game Configuration:")
message(STATUS "  Build Tests: ${AZ_BUILD_TESTS}")
message(STATUS "  Use LibTorch: ${AZ_USE_TORCH}")
message(STATUS "  Build Python Bindings: ${AZ_USE_PYTHON}")
message(STATUS "  Debug Mode: ${AZ_DEBUG_MODE}")
message(STATUS "  CPU Only: ${AZ_USE_CPU_ONLY}")
message(STATUS "  Code Coverage: ${AZ_CODE_COVERAGE}")
message(STATUS "  Static Analysis: ${AZ_STATIC_ANALYSIS}")
message(STATUS "  Game Support:")
message(STATUS "    - Gomoku: ${AZ_BUILD_GOMOKU}")
message(STATUS "    - Chess: ${AZ_BUILD_CHESS}")
message(STATUS "    - Go: ${AZ_BUILD_GO}")
```

## 13. Project Timeline and Milestones

The project timeline is organized into a phased approach to manage complexity and ensure quality. Each game is fully implemented, tested, and optimized before moving to the next.

### 13.1 Phase 1: Foundation and Gomoku Implementation (12 weeks)

**Milestone 1.1: Core Architecture (4 weeks)**
*Deliverables:* 
- Abstract **IGameState** interface with complete specification
- Basic **MCTS** engine implementation with thread safety
- Neural network interface definition
- Standard residual network implementation
- Transposition table with thread-safe access
- Build system with CMake configuration
- CI/CD pipeline setup with automated testing

*Exit Criteria:* Complete architecture design with passing unit tests. Framework can be extended to multiple games.

**Milestone 1.2: Gomoku Implementation (4 weeks)**
*Deliverables:* 
- Complete **GomokuState** implementation with efficient bitboard representation
- Game-specific neural network input representation
- Attack/Defense module for Gomoku patterns
- Enhanced MCTS with game-specific optimizations
- User interface for Gomoku
- Comprehensive test suite

*Exit Criteria:* Functional Gomoku AI achieving > 80% win rate against random play. All tests pass.

**Milestone 1.3: Gomoku Training and Optimization (4 weeks)**
*Deliverables:* 
- Self-play training pipeline for Gomoku
- ELO rating system for tracking progress
- Memory optimization for MCTS search
- SIMD optimizations for critical operations
- Complete Python bindings for training
- Performance testing and profiling

*Exit Criteria:* Gomoku AI reaches target ELO rating (2000+) and performance metrics (move time < 150ms on CPU).

### 13.2 Phase 2: Chess Implementation (8 weeks)

**Milestone 2.1: Chess Game Logic (4 weeks)**
*Deliverables:* 
- Complete **ChessState** implementation with legal move generation
- Chess-specific neural network input representation
- Attack/Defense module adapted for chess
- User interface extended for chess visualization
- Testing and validation against standard chess positions

*Exit Criteria:* Functional Chess implementation with 100% rule compliance verified through test suite.

**Milestone 2.2: Chess Training and Optimization (4 weeks)**
*Deliverables:* 
- Self-play training pipeline adapted for chess
- Chess-specific MCTS optimizations
- Memory and performance optimizations
- Integration with ELO tracking system
- Cross-game consistency verification

*Exit Criteria:* Chess AI reaches target ELO rating (2200+) and performance metrics (move time < 300ms on CPU).

### 13.3 Phase 3: Go Implementation (12 weeks)

**Milestone 3.1: Go Game Logic (6 weeks)**
*Deliverables:* 
- Complete **GoState** implementation supporting both board sizes (9x9 and 19x19)
- Go-specific neural network input representation
- Territory scoring and rules implementation
- User interface extended for Go visualization
- Testing against standard Go problems

*Exit Criteria:* Functional Go implementation with correct territory scoring verified through test suite.

**Milestone 3.2: Go Training and Optimization (6 weeks)**
*Deliverables:* 
- Self-play training pipeline adapted for Go
- Go-specific MCTS optimizations
- Advanced memory management for large branching factor
- Performance optimizations for both 9x9 and 19x19 boards
- Integration with ELO tracking system

*Exit Criteria:* Go AI reaches target ELO rating (2000+) and performance metrics (move time < 500ms on CPU).

### 13.4 Phase 4: Integration and Polish (8 weeks)

**Milestone 4.1: System Integration (4 weeks)**
*Deliverables:* 
- Unified user interface for all games
- Cross-game tournament system
- Comprehensive game analysis tools
- Enhanced visualization and logging
- Deployment packages for all supported platforms

*Exit Criteria:* System operates seamlessly across all games with consistent interface.

**Milestone 4.2: Final Optimization and Documentation (4 weeks)**
*Deliverables:* 
- Final performance optimizations across all games
- Complete API documentation
- User guides and tutorials
- Deployment documentation
- Knowledge transfer materials

*Exit Criteria:* All performance targets met. Documentation complete and verified for accuracy.

## 14. Risk Assessment and Mitigation

### 14.1 Technical Risks

**Risk 1: Game-Specific Complexities**
- *Risk Level:* High for Go, Medium for Chess, Low for Gomoku
- *Impact:* Implementation delays, rule compliance issues
- *Mitigation:* 
  - Phased approach starting with simplest game (Gomoku)
  - Comprehensive test suites for each game
  - Regular rule compliance validation against standard test positions
  - Start with 9x9 Go before scaling to 19x19
  - Allocate extra time buffer for more complex games

**Risk 2: Performance Optimization Challenges**
- *Risk Level:* High for Go, Medium for Chess and Gomoku
- *Impact:* Failure to meet latency and throughput targets
- *Mitigation:*
  - Regular performance profiling from early development stages
  - Game-specific optimizations with shared optimization framework
  - Configurable search parameters based on hardware capabilities
  - Fallback strategies for resource-constrained environments
  - Transposition table with adaptive sizing based on available memory

**Risk 3: Memory Usage in Complex Games**
- *Risk Level:* High for Go, Medium for Chess, Low for Gomoku
- *Impact:* Excessive memory consumption, especially during MCTS search
- *Mitigation:*
  - Memory-efficient node representation using smart pointers
  - Progressive node pruning during search
  - Configurable search depth based on available memory
  - Periodic memory profiling as part of CI/CD pipeline
  - Implement memory budget constraints with graceful degradation

**Risk 4: Training Convergence Issues**
- *Risk Level:* Medium for all games
- *Impact:* Failure to reach target ELO ratings within schedule
- *Mitigation:*
  - Regular evaluation against fixed benchmarks
  - Progressive training approach with intermediate milestones
  - Hyperparameter tuning framework with automated experimentation
  - Provide fallback to simpler but more stable architectures if needed
  - Budget for extended training time in project schedule

**Risk 5: Cross-Platform Compatibility**
- *Risk Level:* Medium
- *Impact:* Build failures or inconsistent behavior across platforms
- *Mitigation:*
  - CI/CD pipeline testing on all target platforms
  - Platform-specific code isolation with clear abstraction boundaries
  - Regular cross-platform testing from early development
  - Focus on standard C++20 features with minimal platform-specific code
  - Containerization for deployment consistency

### 14.2 Process Risks

**Risk 6: Project Scope Management**
- *Risk Level:* High
- *Impact:* Feature creep leading to schedule overruns
- *Mitigation:*
  - Clear definition of in-scope vs. out-of-scope features
  - Phased delivery approach with discrete milestones
  - Regular scope review at milestone boundaries
  - Strict change control process for scope modifications
  - Focus on core functionality before optimizations

**Risk 7: Integration Challenges Between Components**
- *Risk Level:* Medium
- *Impact:* Delays in system integration, interface incompatibilities
- *Mitigation:*
  - Well-defined interfaces between components from the start
  - Regular integration testing throughout development
  - Automated API compatibility checking
  - Mock implementations for components under development
  - Clear ownership and responsibility for interfaces

**Risk 8: Python/C++ Interoperability Issues**
- *Risk Level:* Medium
- *Impact:* Training pipeline performance issues, memory leaks
- *Mitigation:*
  - Clear memory ownership model between languages
  - Comprehensive tests for Python bindings
  - Memory leak detection as part of testing
  - Performance testing of cross-language calls
  - GIL handling strategy defined early in development

**Risk 9: Dependency on External Libraries**
- *Risk Level:* Medium
- *Impact:* Version incompatibilities, build issues
- *Mitigation:*
  - Pin specific versions of all dependencies
  - Vendor critical libraries when feasible
  - Regular dependency updates as part of maintenance
  - Abstraction layers around external dependencies
  - Thorough documentation of all dependencies and versions

**Risk 10: Testing Complexity**
- *Risk Level:* High
- *Impact:* Missed bugs, regression issues
- *Mitigation:*
  - Comprehensive test coverage requirements (>90%)
  - Automated testing as part of CI/CD pipeline
  - Game-specific test suites with standard positions
  - Performance regression testing
  - Regular code reviews focused on testability

## 15. Acceptance Criteria

The project will be considered complete and successful when all of the following criteria are met:

### 15.1 Functional Criteria

1. **Complete Implementation of All Games:**
   - Gomoku, Chess, and Go are fully implemented with 100% rule compliance
   - All game-specific features function correctly in both self-play and human play

2. **AlphaZero Learning Pipeline:**
   - Self-play system generates proper training data
   - Training system can improve model performance over time
   - ELO rating increases consistently during training

3. **User Interface:**
   - Consistent interface across all games
   - Human players can play against the AI at any level
   - Game state visualization for all supported games

4. **Python Integration:**
   - Complete Python bindings for all components
   - Training system is fully operational through Python
   - Game analysis tools accessible through Python

### 15.2 Performance Criteria

1. **Game-Specific Performance Targets:**
   - Gomoku: < 150ms per move on CPU, < 50ms on GPU
   - Chess: < 300ms per move on CPU, < 200ms on GPU
   - Go: < 500ms per move on CPU, < 300ms on GPU

2. **Memory Usage:**
   - Gomoku: < 500 MB RAM
   - Chess: < 1 GB RAM
   - Go: < 2 GB RAM

3. **Training Efficiency:**
   - Competitive models in < 48 hours for Gomoku, < 96 hours for Chess, < 144 hours for Go

4. **Playing Strength:**
   - Gomoku: ELO rating 2000+
   - Chess: ELO rating 2200+
   - Go: ELO rating 2000+

### 15.3 Quality Criteria

1. **Code Quality:**
   - > 90% test coverage
   - Zero critical static analysis warnings
   - Adherence to defined style guides

2. **Reliability:**
   - Zero critical crashes in 1000+ consecutive games
   - Graceful degradation under resource constraints

3. **Documentation:**
   - Complete API documentation
   - User guides for all components
   - Developer documentation for extending the system

4. **Cross-Platform Support:**
   - Successful builds and tests on all target platforms
   - Consistent behavior across platforms

## 16. Glossary

- **AlphaZero:** DeepMind's reinforcement learning system that mastered chess, shogi, and Go from self-play without human knowledge.
- **Attack/Defense Module:** Component that analyzes board positions to identify tactical threats and defensive opportunities.
- **Bitboard:** A data structure that uses bit arrays to represent a game board, enabling efficient operations through bitwise operations.
- **ELO Rating:** A method for calculating the relative skill levels of players in zero-sum games, with higher numbers indicating stronger play.
- **Game Abstraction Layer:** A programming interface that provides a unified way to interact with different board games.
- **GIL (Global Interpreter Lock):** A mutex in Python that prevents multiple threads from executing Python code simultaneously.
- **IGameState:** The core interface that all game implementations must support to work with the AI engine.
- **MCTS (Monte Carlo Tree Search):** A heuristic search algorithm for decision processes using random sampling to build a search tree.
- **PyBind11:** A lightweight header-only library that exposes C++ types in Python and vice versa.
- **RandomPolicyNetwork:** A baseline neural network implementation that returns uniform random policies for benchmarking.
- **Residual Network:** A deep neural network architecture that uses skip connections to facilitate training of very deep networks.
- **SIMD (Single Instruction, Multiple Data):** A class of parallel computing that enables processing multiple data elements with a single instruction.
- **Transposition Table:** A cache that stores previously evaluated game positions to avoid redundant calculations.
- **UCB (Upper Confidence Bound):** Formula used in MCTS to balance exploration and exploitation during tree search.
- **Virtual Loss:** A technique in parallel MCTS where threads temporarily discourage each other from exploring the same nodes.
- **Zobrist Hashing:** A technique used to efficiently generate hash keys for game positions, particularly useful for transposition tables.

## 17. References

1. Silver, D., Schrittwieser, J., Simonyan, K. et al. "Mastering the game of Go without human knowledge." *Nature* 550, 354–359 (2017).
2. Silver, D., Hubert, T., Schrittwieser, J. et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science* 362, 1140-1144 (2018).
3. Hu, J., Shen, L., & Sun, G. "Squeeze-and-Excitation Networks." *IEEE/CVF Conference on Computer Vision and Pattern Recognition* (2018).
4. He, K., Zhang, X., Ren, S., & Sun, J. "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition* (2016).
5. Browne, C. B., Powley, E., Whitehouse, D., et al. "A Survey of Monte Carlo Tree Search Methods." *IEEE Transactions on Computational Intelligence and AI in Games* 4(1), 1–43 (2012).
6. Liu, H., Simonyan, K., & Yang, Y. "DARTS: Differentiable Architecture Search." *International Conference on Learning Representations* (2019).
7. Paszke, A., Gross, S., Massa, F., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems* 32 (2019).
8. "Leela Zero" open-source Go engine: https://github.com/leela-zero/leela-zero
9. "KataGo" open-source Go engine: https://github.com/lightvector/KataGo
10. "Stockfish" open-source Chess engine: https://github.com/official-stockfish/Stockfish
11. PyTorch C++ API documentation: https://pytorch.org/cppdocs/
12. PyBind11 documentation: https://pybind11.readthedocs.io/
13. "International Rules of Gomoku" by Renju International Federation: https://www.renju.net/rifrules/
14. "FIDE Laws of Chess" by World Chess Federation: https://handbook.fide.com/
15. "AGA Rules of Go" by American Go Association: https://www.usgo.org/aga-rules-go
16. Huang, S., Coulom, R., & Lin, S. "Monte-Carlo Simulation Balancing in Practice." *International Conference on Computers and Games* (2010).