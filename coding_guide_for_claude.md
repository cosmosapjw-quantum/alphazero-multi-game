# AlphaZero Multi-Game AI Engine: PRD Analysis and System Prompts

## 1. PRD Analysis Summary

The PRD outlines the development of a production-grade, AlphaZero-style Multi-Game AI engine capable of playing Gomoku, Chess, and Go at tournament-competitive levels without human-provided strategies. Key aspects include:

- **Modular Architecture**: A game abstraction layer allows different games to interface with the same core components
- **Core Technologies**: Combines Monte Carlo Tree Search (MCTS) with a deep neural network (DDW-RandWire-ResNet)
- **Implementation**: High-performance C++ with Python bindings for training and integration
- **Development Approach**: Phased implementation (Gomoku → Chess → Go) to manage complexity
- **Performance Metrics**: Clear targets for response times, resource utilization, and playing strength
- **Cross-Platform**: Support for both Linux and Windows environments

The document provides a detailed technical roadmap covering architecture, interfaces, algorithms, validation strategies, and licensing considerations.

## 2. Converted PRD

### 2.1 Project Overview and Scope

#### 2.1.1 Core Requirements

1. **Overall Objective**: Develop a production-grade, AlphaZero-style Multi-Game AI engine for Gomoku, Chess, and Go
2. **Learning Approach**: Pure self-play reinforcement learning without human knowledge
3. **Implementation Languages**: C++ core with Python bindings
4. **Cross-Platform Support**: Linux and Windows (64-bit)

#### 2.1.2 Success Metrics

5. **Playing Strength**:
   - Gomoku: 2000+ ELO rating
   - Chess: 2200+ ELO rating
   - Go: 2000+ ELO rating
   - 90%+ win rate against benchmark AIs

6. **Performance**:
   - **Move Decision Latency**:
     - Gomoku: 150ms CPU / 50ms GPU
     - Chess: 300ms CPU / 200ms GPU 
     - Go: 500ms CPU / 300ms GPU
   - **Self-Play Throughput**:
     - Gomoku: 50+ games/minute
     - Chess: 25+ games/minute
     - Go: 15+ games/minute
   - **Resource Utilization**: 90%+ GPU utilization, 70%+ CPU utilization

7. **Training Efficiency**:
   - Gomoku: <48 hours
   - Chess: <96 hours
   - Go: <144 hours
   - (On single NVIDIA RTX 3090 GPU or equivalent)

8. **Stability**: Zero critical crashes in 1000+ consecutive games

9. **Memory Efficiency**:
   - Gomoku: <500MB RAM
   - Chess: <1GB RAM
   - Go: <2GB RAM

10. **Code Quality**: 90%+ test coverage, adherence to style guides

#### 2.1.3 Components In Scope

11. **Core Engine Components**:
    - Multi-Game AI Engine Core
    - Game Abstraction Layer
    - Neural Network Architecture
    - AlphaZero-Style Self-Play Learning
    - ELO Rating System

12. **Game-Specific Components**:
    - Gomoku (15x15, optional Renju rules)
    - Chess (Standard and Chess960)
    - Go (9x9 and 19x19 boards)

13. **Interfaces**:
    - Game Interface (human vs. AI)
    - Command-Line Interface
    - REST API Interface
    - Python Bindings

14. **Support Components**:
    - Game Record Storage
    - CI/CD Pipeline
    - Documentation

#### 2.1.4 Components Out of Scope

15. Other board games beyond Gomoku, Chess, and Go
16. Distributed training across multiple machines
17. Mobile/web optimization and deployment
18. Additional game features (timers, undo/redo, analysis tools)
19. Human-crafted knowledge (opening books, endgame databases)
20. External tournament support

### 2.2 System Architecture

#### 2.2.1 Architecture Overview

21. **Layered Architecture**:
    - Game Abstraction Layer (top)
    - Core Engine Components (middle)
    - Support Systems (bottom)
    - Cross-cutting concerns (interfaces, CI/CD)

22. **Game Abstraction Layer**:
    - Abstract interface (IGameState)
    - Game-specific implementations
    - Game state management
    - Move validation and execution

23. **Core MCTS Engine**:
    - MCTSNode for statistics tracking
    - ParallelMCTS for multi-threaded search
    - Virtual loss handling
    - Thread-safe design

24. **Neural Network Engine**:
    - DDW-RandWire-ResNet architecture
    - Enhanced input representation
    - Attack/Defense scoring
    - GPU batch optimization

25. **ELO Rating System**:
    - Rating calculation
    - Match management
    - Historical tracking
    - Baseline opponents

26. **Support Systems**:
    - TranspositionTable for caching
    - ZobristHash for position hashing
    - Self-Play Manager
    - Game Record formats

#### 2.2.2 Game Abstraction Layer

27. **IGameState Interface**:
    - Methods for legal moves, game state, board representation
    - Game result determination
    - Tensor representation for neural network
    - Move history tracking

28. **Game-Specific Implementations**:
    - GomokuState with efficient bitboard representation
    - ChessState with complete rule validation
    - GoState with territory scoring

29. **Enhanced Input Representation**:
    - Multi-channel tensor representation
    - Attack/defense score channels
    - Move history channels
    - Position encoding channels

### 2.3 Core Technical Components

#### 2.3.1 Neural Network Architecture

30. **Network Design**:
    - DDW-RandWire-ResNet architecture
    - Dynamic graph topology
    - Stem layer + random graph backbone
    - Policy and value heads

31. **Key Features**:
    - Small-world and scale-free graph generation
    - SE-style router modules
    - Residual connections
    - Game-specific input/output dimensions

32. **Inference Optimization**:
    - Batch processing
    - Asynchronous execution
    - FP16 precision option
    - Memory-efficient design

#### 2.3.2 Monte Carlo Tree Search

33. **Core MCTS Algorithm**:
    - Selection using UCB/PUCT formula
    - Expansion of promising nodes
    - Evaluation via neural network
    - Backpropagation of results

34. **Parallelization Strategy**:
    - Virtual loss mechanism
    - Thread-safe node statistics
    - Lock-free tree traversal where possible
    - Multi-threaded batch inference

35. **Transposition Table**:
    - Efficient position caching
    - Sharded design for reduced contention
    - Thread-safe access patterns
    - Aging and replacement policies

#### 2.3.3 Self-Play and Training

36. **Self-Play Manager**:
    - Multi-threaded game generation
    - Temperature-based exploration
    - Dirichlet noise injection
    - Game record collection

37. **Training Pipeline**:
    - Data loading and augmentation
    - Loss function (policy + value + L2 regularization)
    - Learning rate scheduling
    - Model evaluation via ELO

38. **Game Record Management**:
    - Standardized format across games
    - Efficient serialization
    - Training dataset creation
    - Analysis and replay capabilities

### 2.4 Implementation Requirements

#### 2.4.1 Performance Requirements

39. **Latency Targets**:
    - Game-specific move decision times
    - Batch inference optimization
    - Critical path optimization
    - Response time consistency

40. **Throughput Requirements**:
    - Self-play generation rate
    - Training throughput
    - Node expansion rate during search
    - I/O and serialization performance

41. **Resource Utilization**:
    - GPU utilization during training
    - CPU threading efficiency
    - Memory consumption limits
    - Disk I/O patterns

#### 2.4.2 Technical Implementation

42. **C++ Implementation**:
    - C++20 standard
    - Modern memory management (smart pointers)
    - RAII principles
    - Const correctness

43. **Python Integration**:
    - PyBind11 bindings
    - GIL handling for performance
    - Memory safety
    - Type hints and documentation

44. **Build System**:
    - CMake configuration
    - Cross-platform support
    - CI/CD integration
    - Dependency management

45. **Testing Framework**:
    - Unit and integration tests
    - Performance benchmarks
    - Memory leak detection
    - Thread sanitizers

#### 2.4.3 Timeline and Validation

46. **Development Phases**:
    - Phase 1: Foundation and Gomoku (Weeks 1-12)
    - Phase 2: Chess Implementation (Weeks 13-20)
    - Phase 3: Go Implementation (Weeks 21-32)
    - Phase 4: Integration and Polish (Weeks 33-40)

47. **Validation Approach**:
    - Game rule compliance testing
    - Performance benchmarking
    - ELO rating evaluation
    - System stability verification

## 3. System Prompts

### 3.1 Core Architecture Prompts

#### 3.1.1 Game Abstraction Layer

```
You are implementing the Game Abstraction Layer for an AlphaZero-style Multi-Game AI Engine. This layer provides a unified interface (IGameState) for Gomoku, Chess, and Go to interact with the MCTS and neural network components.

Your task is to implement:
1. The IGameState interface with methods for:
   - Legal move generation and validation
   - Move execution and undoing
   - Game state evaluation and termination detection
   - Tensor representation for neural network input
   - Position hashing for transposition tables

2. The GomokuState implementation with:
   - Efficient bitboard representation
   - Support for 15x15 board and Renju rules
   - Five-in-a-row win detection
   - Enhanced input representation with attack/defense scoring

Ensure your implementation:
- Is thread-safe for parallel MCTS
- Optimizes for performance (<5% overhead vs. game-specific code)
- Properly handles edge cases (invalid moves, board boundaries)
- Supports cloning for MCTS simulation

The IGameState interface must support all three games through consistent abstractions while allowing game-specific optimizations.
```

#### 3.1.2 Monte Carlo Tree Search Implementation

```
You are implementing the Monte Carlo Tree Search (MCTS) algorithm for an AlphaZero-style Multi-Game AI Engine that supports Gomoku, Chess, and Go.

Your task is to implement:
1. The MCTSNode class with:
   - Visit count and value statistics
   - Parent-child relationships
   - Thread-safe update mechanisms
   - UCB/PUCT scoring for selection

2. The ParallelMCTS class with:
   - Multi-threaded search capabilities
   - Virtual loss for thread coordination
   - Integration with neural network evaluation
   - Support for different selection strategies (UCB, PUCT, progressive bias)
   - Transposition table integration

3. The TranspositionTable with:
   - Efficient caching of evaluated positions
   - Thread-safe access via sharding
   - Aging and replacement policies
   - Memory usage optimization

Ensure your implementation:
- Scales near-linearly with CPU threads (~80% efficiency)
- Correctly handles game-specific state evaluation
- Balances exploration vs. exploitation through proper tuning
- Provides clear interfaces for monitoring and debugging

The MCTS implementation should achieve the performance targets: node throughput of 5,000/s for Gomoku, 3,000/s for Chess, and 2,000/s for Go using 8 threads.
```

#### 3.1.3 Neural Network Architecture

```
You are implementing the Neural Network architecture for an AlphaZero-style Multi-Game AI Engine that supports Gomoku, Chess, and Go.

Your task is to implement:
1. The DDW-RandWire-ResNet architecture with:
   - Dynamic graph topology using small-world and scale-free network principles
   - Residual blocks for each node in the graph
   - SE-style routers for dynamic wiring
   - Policy and value heads for action probabilities and position evaluation

2. The TorchNeuralNetwork class with:
   - PyTorch/LibTorch implementation
   - Efficient batch processing
   - GPU acceleration with >90% utilization
   - Asynchronous inference capabilities
   - Support for both training and inference

3. The enhanced input representation with:
   - Game-specific tensor channels
   - Attack/defense scoring
   - Move history encoding
   - Position encoding (CoordConv)

Ensure your implementation:
- Achieves the specified inference latency targets for each game
- Optimizes memory usage for both training and inference
- Supports different model sizes for different games
- Provides proper interfaces for training and visualization

Include methods for model serialization, loading, and version management to support the training pipeline and ELO tracking system.
```

### 3.2 Game-Specific Implementation Prompts

#### 3.2.1 Gomoku Implementation

```
You are implementing the Gomoku game logic for an AlphaZero-style Multi-Game AI Engine.

Your task is to implement:
1. The GomokuState class derived from IGameState with:
   - Efficient bitboard representation for 15x15 board
   - Standard and Renju rule variations
   - Five-in-a-row win detection in all directions
   - Pattern recognition for common threats (open four, four, etc.)

2. The enhanced tensor representation with:
   - Current player's stones and opponent's stones
   - Attack and defense score channels
   - Previous move history channels
   - Position encoding channels

3. The Gomoku-specific optimizations:
   - Fast pattern detection algorithms
   - Symmetry-aware move generation
   - Efficient threat space analysis

Ensure your implementation:
- Achieves <150ms move time on CPU, <50ms on GPU
- Uses <500MB RAM
- Correctly handles all rule variations and edge cases
- Provides clear interfaces for visualization and human play

The implementation should support both standard Gomoku rules and Renju rules (which adds forbidden moves for the first player to balance the game).
```

#### 3.2.2 Chess Implementation

```
You are implementing the Chess game logic for an AlphaZero-style Multi-Game AI Engine.

Your task is to implement:
1. The ChessState class derived from IGameState with:
   - Complete rule implementation (including castling, en passant, promotion)
   - Support for standard chess and Chess960 variants
   - Check, checkmate, and stalemate detection
   - Draw detection (50-move rule, threefold repetition, insufficient material)

2. The enhanced tensor representation with:
   - Piece-centric planes for each piece type and color
   - Attack and defense mapping
   - Move history and castling rights
   - Check and repetition information

3. The Chess-specific optimizations:
   - Efficient legal move generation
   - Incremental board evaluation
   - Specialized endgame handling

Ensure your implementation:
- Achieves <300ms move time on CPU, <200ms on GPU
- Uses <1GB RAM
- Properly validates all rules and special cases
- Supports standard notation (PGN, FEN) for import/export

The implementation should achieve 2200+ ELO rating (FIDE Master level) in self-play evaluation with suitable training.
```

#### 3.2.3 Go Implementation

```
You are implementing the Go game logic for an AlphaZero-style Multi-Game AI Engine.

Your task is to implement:
1. The GoState class derived from IGameState with:
   - Support for both 9x9 and 19x19 board sizes
   - Complete rule implementation (captures, ko, suicide)
   - Territory scoring for both Japanese and Chinese rules
   - Life and death detection

2. The enhanced tensor representation with:
   - Stone placement and liberties
   - Territory influence mapping
   - Previous moves and ko situations
   - Distance transforms and feature embeddings

3. The Go-specific optimizations:
   - Efficient liberty counting
   - Group and connection tracking
   - Progressive widening for the large branching factor
   - Pattern-based local analysis

Ensure your implementation:
- Achieves <500ms move time on CPU, <300ms on GPU
- Uses <2GB RAM
- Correctly handles all rule variations and edge cases
- Supports standard notation (SGF) for import/export

The implementation should handle the large state space and branching factor of Go efficiently, with special attention to memory usage optimization for the 19x19 board.
```

### 3.3 Training and Evaluation Prompts

#### 3.3.1 Self-Play Training System

```
You are implementing the Self-Play Training system for an AlphaZero-style Multi-Game AI Engine.

Your task is to implement:
1. The SelfPlayManager class with:
   - Parallel game generation capabilities
   - Temperature-based exploration schedule
   - Dirichlet noise injection at the root
   - Game record collection and storage

2. The training data generation pipeline with:
   - State-action-value tuple extraction
   - Data augmentation through symmetries
   - Efficient serialization and loading
   - Batch preparation for neural network training

3. The Python training script with:
   - PyTorch implementation of the learning algorithm
   - Policy and value loss components
   - L2 regularization
   - Learning rate scheduling

Ensure your implementation:
- Achieves the specified self-play throughput (50+ games/minute for Gomoku, etc.)
- Balances exploration and exploitation during training
- Properly handles resource allocation between CPU and GPU tasks
- Includes monitoring and visualization of training progress

The system should demonstrate consistent ELO rating growth during training with an improvement of at least 100 ELO between consecutive iterations until plateauing.
```

#### 3.3.2 ELO Rating System

```
You are implementing the ELO Rating System for an AlphaZero-style Multi-Game AI Engine.

Your task is to implement:
1. The EloTracker class with:
   - Standard ELO calculation formulas
   - Support for match result recording
   - Rating updates based on game outcomes
   - Rating history tracking for progress visualization

2. The benchmarking system with:
   - Tournament match organization
   - Comparison against baseline models
   - Statistical significance testing
   - Cross-validation across different game types

3. The visualization and reporting tools with:
   - Rating progress charts
   - Win rate analysis
   - Confidence interval calculation
   - Performance against specific opponents

Ensure your implementation:
- Maintains separate ratings for each game type
- Handles draws and incomplete games appropriately
- Accounts for first-move advantage where relevant
- Provides reproducible results for proper comparison

The ELO system should help track the progress of models during training and validate that the final models achieve the target ratings (2000+ for Gomoku, 2200+ for Chess, 2000+ for Go).
```

### 3.4 Integration and Interface Prompts

#### 3.4.1 Python Bindings

```
You are implementing Python Bindings for an AlphaZero-style Multi-Game AI Engine using PyBind11.

Your task is to implement:
1. The Python module structure with:
   - Class bindings for all core components
   - Proper memory management and ownership transfer
   - GIL release during compute-intensive operations
   - Exception translation between C++ and Python

2. The training interface with:
   - Self-play data generation
   - Neural network training integration
   - Model saving and loading
   - Performance monitoring

3. The gameplay and visualization interface with:
   - Human vs. AI gameplay
   - Game state visualization
   - Analysis and evaluation tools
   - Tournament organization

Ensure your implementation:
- Preserves the performance of the C++ core
- Provides Pythonic interfaces following Python conventions
- Handles NumPy array conversion efficiently
- Properly documents all interfaces and examples

The Python bindings should support Python 3.8 through 3.11 on both Linux and Windows platforms, with proper version checks and compatibility handling.
```

#### 3.4.2 User Interface Implementation

```
You are implementing the User Interface components for an AlphaZero-style Multi-Game AI Engine.

Your task is to implement:
1. The graphical user interface with:
   - 2D board visualization for all supported games
   - Drag-and-drop move input
   - Game history navigation with undo/redo
   - AI thinking visualization (search tree, policy heat map)

2. The command-line interface with:
   - Batch processing of positions and games
   - Standardized input/output formats
   - Configuration parameters for engine settings
   - Logging and analysis tools

3. The REST API with:
   - Position analysis endpoints
   - Game play endpoints
   - Model management endpoints
   - Authentication and rate limiting

Ensure your implementation:
- Provides consistent interfaces across all games
- Handles errors gracefully with meaningful messages
- Supports standard game notation formats
- Scales appropriately for different display sizes

The interfaces should make the AI engine accessible for human play, automated testing, and integration with other systems while maintaining performance and usability.
```

#### 3.4.3 Build System and CI/CD

```
You are implementing the Build System and CI/CD pipeline for an AlphaZero-style Multi-Game AI Engine using CMake.

Your task is to implement:
1. The CMake configuration with:
   - Cross-platform support for Linux and Windows
   - Dependency management for PyTorch, PyBind11, etc.
   - Configurable build options for different components
   - Proper installation and packaging

2. The CI/CD pipeline with:
   - Automated building on multiple platforms
   - Unit and integration testing
   - Performance regression testing
   - Code quality checks

3. The documentation system with:
   - API documentation generation
   - User guides and tutorials
   - Example code and usage patterns
   - Development guidelines

Ensure your implementation:
- Works consistently across different compilers (GCC, Clang, MSVC)
- Handles dependencies properly with version checks
- Provides clear feedback for build errors
- Supports different build types (Debug, Release, etc.)

The build system should enable efficient development, testing, and deployment of the engine across all supported platforms while maintaining code quality and standards.
```

## 4. Review Notes

1. **Implementation Complexity**: The DDW-RandWire-ResNet architecture is complex and will require careful implementation and tuning. The random graph generation and dynamic wiring aspects are particularly novel and may require experimentation to optimize.

2. **Game-Specific Challenges**: Each game presents unique challenges:
   - **Gomoku**: Pattern recognition and efficient bitboard operations
   - **Chess**: Complex rule validation and specialized endgame handling
   - **Go**: Large state space, liberty counting, and territory scoring

3. **Performance Optimization**: Meeting the performance targets will require careful optimization at multiple levels:
   - Efficient game state representation and move generation
   - Parallelization of MCTS with minimal contention
   - GPU batch processing for neural network inference
   - Memory management for large game trees

4. **Phased Development**: The phased approach (Gomoku → Chess → Go) is sensible for managing complexity, but care should be taken to design shared components that work well for all games from the beginning.

5. **Testing Strategy**: The project will benefit from a comprehensive testing strategy:
   - Unit tests for individual components
   - Integration tests for component interactions
   - Performance benchmarks for key operations
   - Game-specific validation against known positions

6. **Extensibility Considerations**: While the current scope is limited to three games, the architecture should be designed with future extensions in mind:
   - Clean separation of game-specific and game-agnostic code
   - Well-defined interfaces between components
   - Configuration-driven behavior where possible
   - Documentation of extension points

The provided system prompts cover all major aspects of the PRD and should guide effective implementation of the AlphaZero Multi-Game AI Engine.