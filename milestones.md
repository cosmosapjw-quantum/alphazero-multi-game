# AlphaZero Multi-Game AI Engine: Detailed Implementation Milestones

This document provides a comprehensive, step-by-step implementation plan for the AlphaZero Multi-Game AI Engine. Each milestone represents a single, focused functionality with specific validation criteria.

## Phase 1: Foundation and Core Components (16 weeks)

### 1. Project Setup and Environment (2 weeks)

#### Milestone 1.1: Basic CMake Configuration
- **Task**: Create initial CMake configuration for the project
- **Validation**: 
  ```cpp
  // Test by creating a simple executable that prints "Hello AlphaZero"
  #include <gtest/gtest.h>
  
  TEST(ProjectSetupTest, CMakeConfigWorks) {
    EXPECT_TRUE(true);
  }
  ```
- **System Prompt**:
  ```
  Create a basic CMake configuration file for an AlphaZero-style Multi-Game AI Engine project. The file should:
  1. Set the project name to "AlphaZeroMultiGame" with version 1.0.0
  2. Require C++20 standard
  3. Set up basic compiler flags for different build types
  4. Create a single executable target for a "hello world" application
  5. Set up includes and output directories
  
  Only implement the basic CMake file without any actual game engine functionality.
  ```

#### Milestone 1.2: GoogleTest Integration
- **Task**: Set up GoogleTest framework for unit testing
- **Validation**: Create and run a simple passing test
- **System Prompt**:
  ```
  Integrate the GoogleTest framework into the AlphaZero Multi-Game AI Engine project. You should:
  1. Modify the CMake configuration to find and include GoogleTest
  2. Set up a basic test directory structure
  3. Create a simple test file with a passing test
  4. Configure CMake to build and run the tests
  
  Focus only on the testing setup, not any actual AlphaZero functionality.
  ```

#### Milestone 1.3: Continuous Integration Setup
- **Task**: Configure CI pipeline using GitHub Actions
- **Validation**: Verify CI builds project and runs tests
- **System Prompt**:
  ```
  Create a GitHub Actions workflow file for continuous integration of the AlphaZero Multi-Game AI Engine. The workflow should:
  1. Build the project on both Linux and Windows
  2. Run the unit tests
  3. Report success or failure
  4. Cache dependencies to speed up builds
  
  Focus only on CI configuration without adding any actual engine functionality.
  ```

#### Milestone 1.4: Project Directory Structure
- **Task**: Create proper directory structure for the project
- **Validation**: Verify all required directories exist and CMake can locate them
- **System Prompt**:
  ```
  Create a comprehensive directory structure for the AlphaZero Multi-Game AI Engine project. The structure should include:
  1. src/ directory for implementation files
  2. include/ directory for header files
  3. tests/ directory for unit tests
  4. docs/ directory for documentation
  5. examples/ directory for example applications
  
  Update the CMake configuration to use this directory structure properly.
  ```

### 2. Game Abstraction Layer (4 weeks)

#### Milestone 2.1: GameType and GameResult Enums
- **Task**: Define game type and result enumerations
- **Validation**: 
  ```cpp
  TEST(GameEnumsTest, EnumsAreDefined) {
    GameType type = GameType::GOMOKU;
    EXPECT_EQ(static_cast<int>(type), 0);
    
    GameResult result = GameResult::WIN_PLAYER1;
    EXPECT_NE(result, GameResult::ONGOING);
  }
  ```
- **System Prompt**:
  ```
  Implement the basic game type and result enumerations for the AlphaZero Multi-Game AI Engine. You should create:
  1. A GameType enum with values for GOMOKU, CHESS, and GO
  2. A GameResult enum with values for ONGOING, DRAW, WIN_PLAYER1, WIN_PLAYER2
  
  Include a simple test that verifies the enums are defined correctly.
  ```

#### Milestone 2.2: IGameState Interface Declaration
- **Task**: Define the pure abstract game state interface
- **Validation**: Compile tests that verify the interface declaration
- **System Prompt**:
  ```
  Create the IGameState interface for the AlphaZero Multi-Game AI Engine. This is a pure abstract interface that all game implementations will inherit from.

  The interface should declare:
  1. Virtual destructor
  2. Pure virtual methods for getting legal moves, making moves, checking game state
  3. Pure virtual methods for board representation and tensor conversion
  4. Pure virtual methods for action conversion and string representation
  
  Include appropriate comments for each method and a simple test that verifies the interface compiles.
  ```

#### Milestone 2.3: GameStateException Classes
- **Task**: Implement exception classes for game state errors
- **Validation**: Test that exceptions can be thrown and caught with proper messages
- **System Prompt**:
  ```
  Create exception classes for the AlphaZero Multi-Game AI Engine game state. Implement:
  1. A base GameStateException class derived from std::runtime_error
  2. An IllegalMoveException class for illegal move attempts
  3. Any other exception types needed for game state errors
  
  Include a test that verifies exceptions can be thrown and caught with proper messages.
  ```

#### Milestone 2.4: ZobristHash Basic Implementation
- **Task**: Implement basic Zobrist hashing for game positions
- **Validation**: Test that same positions produce identical hashes
- **System Prompt**:
  ```
  Implement a basic ZobristHash class for the AlphaZero Multi-Game AI Engine. This class should:
  1. Generate random numbers for position features
  2. Provide methods to compute and update hash values
  3. Support different board sizes
  4. Handle game-specific hash initialization
  
  Include a test that verifies identical positions produce the same hash values.
  ```

#### Milestone 2.5: Game Factory Method
- **Task**: Implement factory function for creating game states
- **Validation**: Test that the factory creates correct game objects based on type
- **System Prompt**:
  ```
  Implement a createGameState factory function for the AlphaZero Multi-Game AI Engine. This function should:
  1. Take a GameType parameter and optional configuration
  2. Return a unique_ptr to a new IGameState of the appropriate type
  3. Throw appropriate exceptions for invalid parameters
  
  Include a test that verifies the factory correctly creates a game state of the right type.
  ```

### 3. Monte Carlo Tree Search Core (4 weeks)

#### Milestone 3.1: MCTSNode Basic Structure
- **Task**: Implement basic MCTSNode class structure without search algorithm
- **Validation**: Test that nodes can be created and connected correctly
- **System Prompt**:
  ```
  Implement the basic structure of the MCTSNode class for the AlphaZero Multi-Game AI Engine. This class should:
  1. Store visit count and value statistics
  2. Support parent-child relationships
  3. Include atomic variables for thread safety
  4. Track game actions between nodes
  
  Do not implement search algorithm yet, just the data structure. Include a test that verifies nodes can be created and connected properly.
  ```

#### Milestone 3.2: UCB Score Calculation
- **Task**: Implement UCB formula for node selection
- **Validation**: Test that UCB scores are calculated correctly
- **System Prompt**:
  ```
  Implement the UCB (Upper Confidence Bound) score calculation method for the MCTSNode class. This method should:
  1. Follow the PUCT formula used in AlphaZero
  2. Balance exploration and exploitation
  3. Handle the current player perspective
  4. Account for zero-visit edge cases
  
  Include a test that verifies UCB scores are calculated correctly for different node visit counts and values.
  ```

#### Milestone 3.3: Virtual Loss Mechanism
- **Task**: Implement virtual loss for parallel MCTS
- **Validation**: Test that virtual loss affects node selection
- **System Prompt**:
  ```
  Implement the virtual loss mechanism for parallel MCTS in the AlphaZero Multi-Game AI Engine. This implementation should:
  1. Add methods to add and remove virtual losses
  2. Make virtual loss affect UCB score calculation
  3. Ensure thread safety for concurrent operations
  
  Include a test that verifies virtual loss correctly influences node selection.
  ```

#### Milestone 3.4: TranspositionTable Basic Structure
- **Task**: Implement basic TranspositionTable for position caching
- **Validation**: Test that positions can be stored and retrieved
- **System Prompt**:
  ```
  Implement a basic TranspositionTable class for the AlphaZero Multi-Game AI Engine. This class should:
  1. Define a table entry structure with position data
  2. Support storing and retrieving entries by hash
  3. Include basic statistics tracking
  4. Implement thread-safe access through sharding
  
  Include a test that verifies positions can be stored and retrieved correctly.
  ```

### 4. Neural Network Interface (3 weeks)

#### Milestone 4.1: NeuralNetwork Interface Declaration
- **Task**: Define pure abstract neural network interface
- **Validation**: Compile tests that verify the interface
- **System Prompt**:
  ```
  Create the NeuralNetwork interface for the AlphaZero Multi-Game AI Engine. This is a pure abstract interface for neural network implementations.

  The interface should declare:
  1. Methods for single state prediction (returning policy and value)
  2. Methods for batch prediction of multiple states
  3. Asynchronous prediction support
  4. Device information and performance methods
  
  Include a test that verifies the interface compiles correctly.
  ```

#### Milestone 4.2: RandomPolicyNetwork Implementation
- **Task**: Create a simple network that returns random policies for testing
- **Validation**: Test that random network outputs valid policies
- **System Prompt**:
  ```
  Implement a RandomPolicyNetwork class that derives from the NeuralNetwork interface. This implementation should:
  1. Generate random policy vectors for legal moves
  2. Return random values in the range [-1, 1]
  3. Support all methods from the NeuralNetwork interface
  4. Include options for deterministic output (with seed)
  
  Include a test that verifies the random policies are valid probability distributions.
  ```

#### Milestone 4.3: BatchQueue Implementation
- **Task**: Create a queue for batching neural network inputs
- **Validation**: Test that requests are correctly batched and processed
- **System Prompt**:
  ```
  Implement a BatchQueue class for efficient neural network inference in the AlphaZero Multi-Game AI Engine. This class should:
  1. Collect individual inference requests into batches
  2. Process batches asynchronously
  3. Return results through futures
  4. Handle timeouts for incomplete batches
  
  Include a test that verifies requests are correctly batched and processed.
  ```

### 5. Gomoku Implementation (3 weeks)

#### Milestone 5.1: GomokuState Basic Structure
- **Task**: Implement basic GomokuState class derived from IGameState
- **Validation**: Test that board can be created and accessed
- **System Prompt**:
  ```
  Implement the basic structure of the GomokuState class for the AlphaZero Multi-Game AI Engine. This class should:
  1. Derive from IGameState
  2. Use bitboards for efficient representation
  3. Support standard 15x15 board and optional Renju rules
  4. Include constants and basic accessors
  
  Do not implement game logic yet, just the structure. Include a test that verifies the board can be created and accessed.
  ```

#### Milestone 5.2: Gomoku Bitboard Operations
- **Task**: Implement efficient bitboard operations for Gomoku
- **Validation**: Test that stones can be placed and detected correctly
- **System Prompt**:
  ```
  Implement efficient bitboard operations for the GomokuState class. These operations should:
  1. Support setting and clearing bits for stone placement
  2. Provide methods to test if positions contain stones
  3. Handle board boundaries correctly
  4. Support boards of different sizes
  
  Include a test that verifies stones can be placed and detected correctly on the board.
  ```

#### Milestone 5.3: Gomoku Move Validation
- **Task**: Implement legal move validation for Gomoku
- **Validation**: Test that legal moves are correctly identified
- **System Prompt**:
  ```
  Implement move validation for the GomokuState class. This implementation should:
  1. Check if a position is within bounds
  2. Verify if a position is empty
  3. Handle Renju rule forbidden moves for the first player
  4. Generate a list of all legal moves
  
  Include a test that verifies legal moves are correctly identified and illegal moves are rejected.
  ```

#### Milestone 5.4: Gomoku Win Detection
- **Task**: Implement five-in-a-row detection for Gomoku
- **Validation**: Test win detection in all directions
- **System Prompt**:
  ```
  Implement win detection for the GomokuState class. This implementation should:
  1. Detect five consecutive stones in any direction
  2. Check horizontal, vertical, and diagonal lines
  3. Determine the game result (ongoing, win, draw)
  4. Be efficient for frequent checking
  
  Include a test that verifies wins are correctly detected in all directions.
  ```

#### Milestone 5.5: Gomoku Renju Rules
- **Task**: Implement Renju rules for forbidden moves
- **Validation**: Test that forbidden moves are correctly identified
- **System Prompt**:
  ```
  Implement the Renju rules for the GomokuState class. These rules should:
  1. Detect and forbid overlines (more than five stones)
  2. Detect and forbid "double-three" situations
  3. Detect and forbid "double-four" situations
  4. Only apply to the first player
  
  Include a test that verifies forbidden moves are correctly identified.
  ```

#### Milestone 5.6: Gomoku Basic Tensor Representation
- **Task**: Implement basic tensor representation for neural network input
- **Validation**: Test that tensor accurately represents board state
- **System Prompt**:
  ```
  Implement the basic tensor representation for the GomokuState class. This implementation should:
  1. Create a multi-channel tensor with player stones and turn information
  2. Represent the current player's stones in one channel
  3. Represent the opponent's stones in another channel
  4. Include a turn indicator channel
  
  Include a test that verifies the tensor accurately represents the board state.
  ```

#### Milestone 5.7: Gomoku Enhanced Tensor Representation
- **Task**: Add enhanced features to tensor representation
- **Validation**: Test that enhanced features are correctly included
- **System Prompt**:
  ```
  Implement enhanced tensor representation for the GomokuState class. This implementation should:
  1. Add channels for previous N moves history
  2. Add positional encoding channels (CoordConv)
  3. Ensure compatibility with the neural network interface
  4. Optimize for performance
  
  Include a test that verifies the enhanced features are correctly included in the tensor.
  ```

#### Milestone 5.8: Gomoku Zobrist Hashing
- **Task**: Implement specific Zobrist hashing for Gomoku
- **Validation**: Test that identical positions have same hash
- **System Prompt**:
  ```
  Implement Zobrist hashing for the GomokuState class. This implementation should:
  1. Initialize hash tables specific to Gomoku
  2. Update hash incrementally with each move
  3. Support the full board size
  4. Be consistent for identical positions
  
  Include a test that verifies identical positions have the same hash and different positions have different hashes.
  ```

#### Milestone 5.9: Gomoku Attack/Defense Scoring
- **Task**: Implement attack and defense pattern scoring
- **Validation**: Test that common patterns are correctly identified
- **System Prompt**:
  ```
  Implement attack and defense pattern scoring for the GomokuState class. This implementation should:
  1. Identify common patterns like open-four, four, open-three
  2. Assign scores based on pattern threats
  3. Support both offensive and defensive evaluation
  4. Be efficient for inclusion in the neural network input
  
  Include a test that verifies common patterns are correctly identified and scored.
  ```

#### Milestone 5.10: Complete GomokuState Implementation
- **Task**: Finalize the GomokuState with all required methods
- **Validation**: Test full game sequences and edge cases
- **System Prompt**:
  ```
  Complete the GomokuState implementation by finalizing all required methods. Ensure:
  1. All IGameState interface methods are properly implemented
  2. Move making and undoing work correctly
  3. Game result determination is accurate
  4. String representation and action conversion work properly
  
  Include comprehensive tests that verify full game sequences and edge cases.
  ```

### 6. MCTS Implementation (3 weeks)

#### Milestone 6.1: MCTS Selection Phase
- **Task**: Implement the selection phase of MCTS algorithm
- **Validation**: Test that promising nodes are selected
- **System Prompt**:
  ```
  Implement the selection phase of the MCTS algorithm for the AlphaZero Multi-Game AI Engine. This implementation should:
  1. Select leaf nodes based on UCB scores
  2. Traverse the tree using game state transitions
  3. Handle virtual loss for parallel search
  4. Stop at unexpanded or terminal nodes
  
  Include a test that verifies promising nodes are selected based on UCB scores.
  ```

#### Milestone 6.2: MCTS Expansion Phase
- **Task**: Implement the expansion phase of MCTS algorithm
- **Validation**: Test that nodes are correctly expanded with children
- **System Prompt**:
  ```
  Implement the expansion phase of the MCTS algorithm. This implementation should:
  1. Get legal moves from the game state
  2. Create child nodes for each legal move
  3. Initialize child nodes with prior probabilities
  4. Handle thread safety for parallel expansion
  
  Include a test that verifies nodes are correctly expanded with appropriate children.
  ```

#### Milestone 6.3: MCTS Evaluation and Backpropagation
- **Task**: Implement evaluation and backpropagation phases
- **Validation**: Test that values are correctly propagated up the tree
- **System Prompt**:
  ```
  Implement the evaluation and backpropagation phases of the MCTS algorithm. This implementation should:
  1. Evaluate leaf nodes using the neural network
  2. Handle terminal state evaluations directly
  3. Propagate values back up the tree
  4. Update visit counts and statistics
  
  Include a test that verifies values are correctly propagated through the tree.
  ```

#### Milestone 6.4: ParallelMCTS Class
- **Task**: Create ParallelMCTS to orchestrate multi-threaded search
- **Validation**: Test that parallel search works correctly
- **System Prompt**:
  ```
  Implement the ParallelMCTS class for multi-threaded Monte Carlo Tree Search. This class should:
  1. Manage the root node and game state
  2. Run multiple search threads concurrently
  3. Coordinate thread synchronization
  4. Extract final action probabilities
  
  Include a test that verifies parallel search works correctly and improves performance.
  ```

#### Milestone 6.5: Dirichlet Noise Implementation
- **Task**: Add Dirichlet noise to root node for exploration
- **Validation**: Test that noise affects action selection
- **System Prompt**:
  ```
  Implement Dirichlet noise for root exploration in the ParallelMCTS class. This implementation should:
  1. Add noise to prior probabilities at the root
  2. Control noise amount with epsilon parameter
  3. Use appropriate alpha values for each game type
  4. Apply only during training (not evaluation)
  
  Include a test that verifies noise affects action selection.
  ```

#### Milestone 6.6: Temperature-based Selection
- **Task**: Implement temperature scaling for move selection
- **Validation**: Test that temperature affects move distribution
- **System Prompt**:
  ```
  Implement temperature-based move selection for the ParallelMCTS class. This implementation should:
  1. Scale visit counts by temperature parameter
  2. Support temperature annealing over the course of a game
  3. Handle deterministic selection at temperature=0
  4. Return properly normalized probability distribution
  
  Include a test that verifies temperature affects move selection distribution.
  ```

### 7. Self-Play and Training System (4 weeks)

#### Milestone 7.1: GameRecord Structure
- **Task**: Implement GameRecord for storing self-play data
- **Validation**: Test that games can be recorded and loaded
- **System Prompt**:
  ```
  Implement the GameRecord class for storing self-play data. This class should:
  1. Store game metadata (type, size, rules)
  2. Record each move with policy, value, and timing data
  3. Store the final game result
  4. Support serialization to/from JSON
  
  Include a test that verifies games can be recorded and loaded correctly.
  ```

#### Milestone 7.2: Single-Game Self-Play
- **Task**: Implement single game self-play logic
- **Validation**: Test that a complete game can be played
- **System Prompt**:
  ```
  Implement single-game self-play logic for the AlphaZero Multi-Game AI Engine. This implementation should:
  1. Play a complete game from start to finish
  2. Use MCTS with neural network evaluation
  3. Apply temperature annealing over the course of the game
  4. Record moves, policies, and values
  
  Include a test that verifies a complete game can be played and recorded.
  ```

#### Milestone 7.3: SelfPlayManager Basic Structure
- **Task**: Create SelfPlayManager to orchestrate game generation
- **Validation**: Test manager configuration and basic operation
- **System Prompt**:
  ```
  Implement the basic structure of the SelfPlayManager class. This class should:
  1. Configure self-play parameters (simulations, threads, etc.)
  2. Support different game types and board sizes
  3. Include temperature and exploration settings
  4. Provide callbacks for progress reporting
  
  Include a test that verifies the manager can be configured and started correctly.
  ```

#### Milestone 7.4: Parallel Game Generation
- **Task**: Add multi-threaded game generation support
- **Validation**: Test that multiple games can be generated in parallel
- **System Prompt**:
  ```
  Implement parallel game generation in the SelfPlayManager class. This implementation should:
  1. Create multiple games concurrently
  2. Balance thread usage between games and MCTS
  3. Collect results safely from all threads
  4. Provide progress monitoring
  
  Include a test that verifies multiple games can be generated in parallel.
  ```

#### Milestone 7.5: Training Data Extraction
- **Task**: Implement training example extraction from game records
- **Validation**: Test that valid training data is produced
- **System Prompt**:
  ```
  Implement training data extraction from game records. This implementation should:
  1. Extract state-action-value tuples from games
  2. Apply appropriate augmentation (rotations, reflections)
  3. Format data for neural network training
  4. Support filtering and sampling options
  
  Include a test that verifies valid training data is produced from game records.
  ```

#### Milestone 7.6: ELO Rating System
- **Task**: Implement ELO rating calculation for model strength
- **Validation**: Test that ratings are calculated correctly
- **System Prompt**:
  ```
  Implement an ELO rating system for tracking model strength. This implementation should:
  1. Calculate and update ELO ratings based on game results
  2. Support game-specific ratings for different game types
  3. Track rating history over time
  4. Provide methods for comparing model strength
  
  Include a test that verifies ratings are calculated correctly based on game outcomes.
  ```

#### Milestone 7.7: Python Training Script
- **Task**: Create Python script for neural network training
- **Validation**: Test that the script loads data and trains a model
- **System Prompt**:
  ```
  Create a Python training script for the neural network. This script should:
  1. Load training data from game records
  2. Define the neural network architecture
  3. Implement the training loop with appropriate loss functions
  4. Save model checkpoints and track progress
  
  Include test code that verifies the script loads data and performs basic training.
  ```

### 8. User Interface and API (4 weeks)

#### Milestone 8.1: Basic Command-Line Interface
- **Task**: Implement basic CLI for interacting with the engine
- **Validation**: Test that commands are correctly processed
- **System Prompt**:
  ```
  Implement a basic command-line interface for the AlphaZero Multi-Game AI Engine. This CLI should:
  1. Parse and execute commands for different game types
  2. Support configuration of engine parameters
  3. Allow loading and saving games
  4. Provide both interactive and batch modes
  
  Include a test that verifies commands are correctly processed.
  ```

#### Milestone 8.2: Game Text Visualization
- **Task**: Add text-based visualization of game boards
- **Validation**: Test that boards are correctly displayed
- **System Prompt**:
  ```
  Implement text-based visualization for game boards. This implementation should:
  1. Display Gomoku boards with stones and coordinates
  2. Format the output for readability
  3. Support different board sizes
  4. Include move history information
  
  Include a test that verifies boards are correctly displayed.
  ```

#### Milestone 8.3: Human vs. AI Play Mode
- **Task**: Implement mode for human players to play against AI
- **Validation**: Test game flow with human input
- **System Prompt**:
  ```
  Implement a human vs. AI play mode for the command-line interface. This mode should:
  1. Accept human moves in standard notation
  2. Generate AI responses using MCTS
  3. Display the board after each move
  4. Support game configuration options
  
  Include a test that verifies the game flow with simulated human input.
  ```

#### Milestone 8.4: REST API Basic Endpoints
- **Task**: Create basic REST API for engine access
- **Validation**: Test that endpoints return correct responses
- **System Prompt**:
  ```
  Implement basic REST API endpoints for the AlphaZero Multi-Game AI Engine. These endpoints should:
  1. Create and manage game sessions
  2. Accept moves and return AI responses
  3. Provide game state information
  4. Support configuration parameters
  
  Include a test that verifies endpoints return correct responses.
  ```

## Phase 2: Chess Implementation (8 weeks)

### 9. Chess State Implementation (4 weeks)

#### Milestone 9.1: ChessState Basic Structure
- **Task**: Implement basic ChessState class derived from IGameState
- **Validation**: Test that the basic structure compiles and works
- **System Prompt**:
  ```
  Implement the basic structure of the ChessState class. This class should:
  1. Derive from IGameState
  2. Use an efficient board representation (bitboards recommended)
  3. Define piece types and constants
  4. Include basic accessors for board state
  
  Include a test that verifies the basic structure works correctly.
  ```

#### Milestone 9.2: Chess Piece Movement
- **Task**: Implement basic piece movement rules
- **Validation**: Test that pieces move according to chess rules
- **System Prompt**:
  ```
  Implement basic piece movement rules for the ChessState class. This implementation should:
  1. Define movement patterns for all piece types
  2. Handle blocked movement properly
  3. Implement capture rules
  4. Validate moves against the rules
  
  Include tests that verify each piece type moves correctly.
  ```

#### Milestone 9.3: Chess Check Detection
- **Task**: Implement check detection logic
- **Validation**: Test that check situations are correctly identified
- **System Prompt**:
  ```
  Implement check detection for the ChessState class. This implementation should:
  1. Detect when a king is under attack
  2. Prevent moves that leave/put the king in check
  3. Handle discovered checks
  4. Be efficient for frequent checking
  
  Include tests that verify check situations are correctly identified.
  ```

#### Milestone 9.4: Chess Special Moves (Castling, En Passant)
- **Task**: Implement special chess moves
- **Validation**: Test that special moves work correctly
- **System Prompt**:
  ```
  Implement special chess moves for the ChessState class. This implementation should:
  1. Support castling (kingside and queenside)
  2. Implement en passant captures
  3. Handle promotion of pawns
  4. Track required state for special move validation
  
  Include tests that verify special moves work correctly.
  ```

#### Milestone 9.5: Chess Game Result Detection
- **Task**: Implement checkmate and draw detection
- **Validation**: Test that game end conditions are correctly identified
- **System Prompt**:
  ```
  Implement game result detection for the ChessState class. This implementation should:
  1. Detect checkmate situations
  2. Identify stalemate positions
  3. Implement draw rules (fifty-move rule, insufficient material)
  4. Handle threefold repetition
  
  Include tests that verify different game end conditions are correctly identified.
  ```

#### Milestone 9.6: Chess FEN/PGN Support
- **Task**: Add support for FEN position parsing and PGN notation
- **Validation**: Test that positions can be loaded and saved
- **System Prompt**:
  ```
  Implement FEN and PGN support for the ChessState class. This implementation should:
  1. Parse FEN strings to set up board positions
  2. Generate FEN strings from current positions
  3. Parse basic PGN move notation
  4. Convert internal moves to PGN notation
  
  Include tests that verify positions can be correctly loaded from FEN and moves can be parsed from PGN.
  ```

#### Milestone 9.7: Chess960 Support
- **Task**: Add support for Chess960 (Fischer Random Chess)
- **Validation**: Test that valid Chess960 positions are generated
- **System Prompt**:
  ```
  Implement Chess960 support for the ChessState class. This implementation should:
  1. Generate valid Chess960 starting positions
  2. Apply correct castling rules for Chess960
  3. Support FEN notation for Chess960 positions
  4. Include proper validation for Chess960-specific rules
  
  Include tests that verify valid Chess960 positions are generated and castling works correctly.
  ```

#### Milestone 9.8: Chess Tensor Representation
- **Task**: Implement tensor representation for neural network input
- **Validation**: Test that tensor correctly represents chess positions
- **System Prompt**:
  ```
  Implement the tensor representation for ChessState. This representation should:
  1. Use piece-centric planes (one per piece type and color)
  2. Include auxiliary planes for castling rights, en passant, etc.
  3. Add repetition counters and move counters
  4. Support the neural network input requirements
  
  Include tests that verify the tensor representation correctly captures chess positions.
  ```

#### Milestone 9.9: Chess Enhanced Tensor Representation
- **Task**: Add enhanced features to the tensor representation
- **Validation**: Test that enhanced features are correctly included
- **System Prompt**:
  ```
  Implement an enhanced tensor representation for ChessState. This implementation should:
  1. Add attack and defense maps
  2. Include mobility information
  3. Add piece development features
  4. Include king safety features
  
  Include tests that verify the enhanced features are correctly represented in the tensor.
  ```

#### Milestone 9.10: Complete ChessState Implementation
- **Task**: Finalize the ChessState with all required methods
- **Validation**: Test full game sequences and edge cases
- **System Prompt**:
  ```
  Complete the ChessState implementation by finalizing all required methods. Ensure:
  1. All IGameState interface methods are properly implemented
  2. Move making and undoing work correctly with history
  3. Game result determination is accurate in all scenarios
  4. All edge cases are handled properly
  
  Include comprehensive tests for full game sequences and edge cases.
  ```

### 10. Chess-Specific Optimizations (4 weeks)

#### Milestone 10.1: Chess Move Generation Optimization
- **Task**: Optimize legal move generation for chess
- **Validation**: Test performance and correctness with benchmark positions
- **System Prompt**:
  ```
  Optimize legal move generation for the ChessState class. The optimization should:
  1. Use efficient bitboard operations for move generation
  2. Implement move generation for each piece type optimally
  3. Handle pinned pieces efficiently
  4. Meet performance targets (<1ms for standard positions)
  
  Include performance tests with benchmark positions (perft) to validate correctness and speed.
  ```

#### Milestone 10.2: Chess Zobrist Hashing
- **Task**: Implement specific Zobrist hashing for chess
- **Validation**: Test that identical positions have same hash
- **System Prompt**:
  ```
  Implement Zobrist hashing for the ChessState class. This implementation should:
  1. Create hash values for each piece on each square
  2. Handle castling rights, en passant, and side to move
  3. Update hash incrementally with each move
  4. Handle special moves correctly
  
  Include tests that verify identical positions have the same hash and different positions have different hashes.
  ```

#### Milestone 10.3: Chess Transposition Table Tuning
- **Task**: Tune transposition table for chess-specific needs
- **Validation**: Test hit rates and performance improvement
- **System Prompt**:
  ```
  Tune the TranspositionTable for chess-specific needs. This tuning should:
  1. Optimize entry size for chess positions
  2. Implement replacement strategies suitable for chess
  3. Handle hash collisions appropriately
  4. Measure and improve cache hit rates
  
  Include tests that measure hit rates and performance improvements from the transposition table.
  ```

#### Milestone 10.4: Chess Attack/Defense Mapping
- **Task**: Implement efficient attack and defense mapping
- **Validation**: Test that attack maps are correctly generated
- **System Prompt**:
  ```
  Implement efficient attack and defense mapping for the ChessState class. This implementation should:
  1. Track which squares are attacked by each side
  2. Calculate piece mobility efficiently
  3. Identify defended pieces
  4. Use bitboard operations for performance
  
  Include tests that verify attack maps are correctly generated for different positions.
  ```

## Phase 3: Go Implementation (8 weeks)

### 11. Go State Implementation (4 weeks)

#### Milestone 11.1: GoState Basic Structure
- **Task**: Implement basic GoState class derived from IGameState
- **Validation**: Test that the basic structure compiles and works
- **System Prompt**:
  ```
  Implement the basic structure of the GoState class. This class should:
  1. Derive from IGameState
  2. Support both 9x9 and 19x19 board sizes
  3. Define stone colors and basic constants
  4. Include basic accessors for board state
  
  Include a test that verifies the basic structure works correctly for different board sizes.
  ```

#### Milestone 11.2: Go Stone Placement Rules
- **Task**: Implement basic stone placement rules
- **Validation**: Test that stones are placed according to rules
- **System Prompt**:
  ```
  Implement basic stone placement rules for the GoState class. This implementation should:
  1. Allow placing stones on empty intersections
  2. Handle board boundaries correctly
  3. Support proper turn alternation
  4. Track move history
  
  Include tests that verify stones can be placed according to the rules.
  ```

#### Milestone 11.3: Go Liberty Counting
- **Task**: Implement liberty counting for stone groups
- **Validation**: Test that liberties are correctly counted
- **System Prompt**:
  ```
  Implement liberty counting for the GoState class. This implementation should:
  1. Track connected groups of stones
  2. Count liberties (empty adjacent intersections) for each group
  3. Be efficient for frequent liberty checking
  4. Handle edge and corner cases correctly
  
  Include tests that verify liberties are correctly counted for different stone arrangements.
  ```

#### Milestone 11.4: Go Capture Rules
- **Task**: Implement capture rules for stones without liberties
- **Validation**: Test that captures occur correctly
- **System Prompt**:
  ```
  Implement capture rules for the GoState class. This implementation should:
  1. Remove stones with zero liberties
  2. Handle multiple group captures in one move
  3. Track captured stones count
  4. Apply captures after each move
  
  Include tests that verify captures occur correctly in different scenarios.
  ```

#### Milestone 11.5: Go Ko Rule
- **Task**: Implement the Ko rule to prevent position repetition
- **Validation**: Test that Ko rule prevents immediate recapture
- **System Prompt**:
  ```
  Implement the Ko rule for the GoState class. This implementation should:
  1. Prevent immediate recapture that would repeat a position
  2. Track the Ko point when applicable
  3. Clear Ko restriction when appropriate
  4. Support both simple Ko and superko variants
  
  Include tests that verify the Ko rule prevents immediate recapture in relevant scenarios.
  ```

#### Milestone 11.6: Go Suicide Rule
- **Task**: Implement the suicide rule (self-capture)
- **Validation**: Test that suicide moves are handled correctly
- **System Prompt**:
  ```
  Implement the suicide rule for the GoState class. This implementation should:
  1. Determine if a move would result in self-capture
  2. Allow or prohibit suicide moves based on rule variant
  3. Support both Japanese (no suicide) and Chinese (allowed suicide) rules
  4. Handle complex cases where a move both captures and reduces own liberties
  
  Include tests that verify suicide moves are handled correctly according to the specified rules.
  ```

#### Milestone 11.7: Go Territory Scoring
- **Task**: Implement territory scoring for game end
- **Validation**: Test that territory is correctly scored
- **System Prompt**:
  ```
  Implement territory scoring for the GoState class. This implementation should:
  1. Identify territory controlled by each player
  2. Score according to either Japanese or Chinese rules
  3. Handle dead stones correctly
  4. Calculate final score including captures
  
  Include tests that verify territory is correctly scored in different end-game positions.
  ```

#### Milestone 11.8: Go Tensor Representation
- **Task**: Implement tensor representation for neural network input
- **Validation**: Test that tensor correctly represents Go positions
- **System Prompt**:
  ```
  Implement the tensor representation for GoState. This representation should:
  1. Encode stone positions for both players
  2. Include liberty counts for groups
  3. Encode previous moves for positional history
  4. Add metadata like ko situation
  
  Include tests that verify the tensor representation correctly captures Go positions.
  ```

#### Milestone 11.9: Go Enhanced Tensor Representation
- **Task**: Add enhanced features to the tensor representation
- **Validation**: Test that enhanced features are correctly included
- **System Prompt**:
  ```
  Implement an enhanced tensor representation for GoState. This implementation should:
  1. Add distance transforms from borders
  2. Include territory influence maps
  3. Add liberties and "ladder" status features
  4. Include features for move history
  
  Include tests that verify the enhanced features are correctly represented in the tensor.
  ```

#### Milestone 11.10: Complete GoState Implementation
- **Task**: Finalize the GoState with all required methods
- **Validation**: Test full game sequences and edge cases
- **System Prompt**:
  ```
  Complete the GoState implementation by finalizing all required methods. Ensure:
  1. All IGameState interface methods are properly implemented
  2. Move making and undoing work correctly with history
  3. Game result determination is accurate in all scenarios
  4. All edge cases are handled properly
  
  Include comprehensive tests for full game sequences and edge cases.
  ```

### 12. Go-Specific Optimizations (4 weeks)

#### Milestone 12.1: Go Group Tracking Optimization
- **Task**: Optimize connected group tracking for efficiency
- **Validation**: Test performance with large groups
- **System Prompt**:
  ```
  Optimize the connected group tracking system for the GoState class. This optimization should:
  1. Efficiently track and update stone groups
  2. Use union-find data structure for group merging
  3. Support incremental updates with each move
  4. Meet performance targets for 19x19 boards
  
  Include performance tests with different board sizes and group configurations.
  ```

#### Milestone 12.2: Go Zobrist Hashing
- **Task**: Implement specific Zobrist hashing for Go
- **Validation**: Test that identical positions have same hash
- **System Prompt**:
  ```
  Implement Zobrist hashing for the GoState class. This implementation should:
  1. Create hash values for each stone position
  2. Handle ko situation in the hash
  3. Update hash incrementally with each move
  4. Support both 9x9 and 19x19 board sizes
  
  Include tests that verify identical positions have the same hash and different positions have different hashes.
  ```

#### Milestone 12.3: MCTS Progressive Widening for Go
- **Task**: Implement progressive widening for the large branching factor
- **Validation**: Test that search focuses on promising moves
- **System Prompt**:
  ```
  Implement progressive widening for the MCTS algorithm specific to Go. This implementation should:
  1. Limit the number of children expanded at each node
  2. Incrementally increase this limit based on visit count
  3. Prioritize moves based on policy network output
  4. Balance exploration and focused search
  
  Include tests that verify the search effectively focuses on promising moves in Go positions.
  ```

#### Milestone 12.4: Go Pattern Recognition
- **Task**: Implement basic Go pattern recognition
- **Validation**: Test that common patterns are identified
- **System Prompt**:
  ```
  Implement basic pattern recognition for the GoState class. This implementation should:
  1. Identify common tactical patterns (ladders, nets, etc.)
  2. Recognize common corner and edge formations
  3. Detect basic eye shapes
  4. Support both defensive and offensive patterns
  
  Include tests that verify common patterns are correctly identified.
  ```

## Phase 4: Integration and Polish (4 weeks)

### 13. Multi-Game Integration (2 weeks)

#### Milestone 13.1: Unified Game Selection Interface
- **Task**: Create a unified interface for game selection
- **Validation**: Test that all games can be created and played
- **System Prompt**:
  ```
  Create a unified game selection interface for the AlphaZero Multi-Game AI Engine. This interface should:
  1. Provide a common entry point for creating any supported game
  2. Handle game-specific parameters properly
  3. Support configuration for variants of each game
  4. Include helper methods for common operations
  
  Include tests that verify all games can be created and played through this interface.
  ```

#### Milestone 13.2: Cross-Game Tournament System
- **Task**: Implement a system for running tournaments
- **Validation**: Test that tournaments can be run with different games
- **System Prompt**:
  ```
  Implement a cross-game tournament system for the AlphaZero Multi-Game AI Engine. This system should:
  1. Support tournaments for any of the supported games
  2. Allow different player types (AI, random, human)
  3. Track results and statistics
  4. Generate tournament reports
  
  Include tests that verify tournaments can be run with different games and player types.
  ```

#### Milestone 13.3: Game-Specific Configuration System
- **Task**: Create a configuration system for game-specific settings
- **Validation**: Test that configurations apply correctly
- **System Prompt**:
  ```
  Create a game-specific configuration system for the AlphaZero Multi-Game AI Engine. This system should:
  1. Support common parameters across all games
  2. Handle game-specific parameters properly
  3. Use a consistent format (JSON recommended)
  4. Support validation of configuration values
  
  Include tests that verify configurations apply correctly to each game type.
  ```

#### Milestone 13.4: Cross-Game Visualization
- **Task**: Implement unified visualization for all games
- **Validation**: Test that visualization works for all games
- **System Prompt**:
  ```
  Implement a unified visualization system for all supported games. This system should:
  1. Provide consistent board visualization across games
  2. Support game-specific rendering requirements
  3. Include move history visualization
  4. Offer both text and graphical output options
  
  Include tests that verify visualization works correctly for all game types.
  ```

### 14. Performance Optimization (2 weeks)

#### Milestone 14.1: Memory Usage Profiling
- **Task**: Implement memory profiling for all components
- **Validation**: Test that memory usage stays within limits
- **System Prompt**:
  ```
  Implement memory usage profiling for all components of the AlphaZero Multi-Game AI Engine. This implementation should:
  1. Track memory usage during operation
  2. Identify memory bottlenecks
  3. Verify compliance with memory requirements
  4. Support different game types and configurations
  
  Include tests that verify memory usage stays within the specified limits for each game.
  ```

#### Milestone 14.2: Neural Network Inference Optimization
- **Task**: Optimize neural network inference for performance
- **Validation**: Test that inference meets latency targets
- **System Prompt**:
  ```
  Optimize neural network inference for the AlphaZero Multi-Game AI Engine. This optimization should:
  1. Implement FP16 precision where appropriate
  2. Optimize batch sizes for GPU utilization
  3. Minimize CPU-GPU data transfer
  4. Meet latency targets for all supported games
  
  Include performance tests that verify inference meets the latency targets.
  ```

#### Milestone 14.3: MCTS Parallel Scaling Optimization
- **Task**: Optimize MCTS parallelization for linear scaling
- **Validation**: Test scaling efficiency across CPU cores
- **System Prompt**:
  ```
  Optimize MCTS parallelization for linear scaling. This optimization should:
  1. Reduce lock contention in the search tree
  2. Balance thread workloads effectively
  3. Optimize virtual loss parameters
  4. Achieve near-linear scaling up to 8+ cores
  
  Include scaling tests that verify efficiency across different numbers of CPU cores.
  ```

#### Milestone 14.4: Complete System Benchmarking
- **Task**: Create comprehensive benchmarks for the system
- **Validation**: Test that all performance targets are met
- **System Prompt**:
  ```
  Create comprehensive benchmarks for the AlphaZero Multi-Game AI Engine. These benchmarks should:
  1. Test all performance metrics in the PRD
  2. Include all supported games and configurations
  3. Measure both real-time performance and resource utilization
  4. Generate detailed reports for comparison
  
  Include benchmark runs that verify all performance targets are met.
  ```

### 15. Documentation and Release (2 weeks)

#### Milestone 15.1: API Documentation
- **Task**: Complete comprehensive API documentation
- **Validation**: Test that all public interfaces are documented
- **System Prompt**:
  ```
  Complete comprehensive API documentation for the AlphaZero Multi-Game AI Engine. This documentation should:
  1. Cover all public classes and methods
  2. Include parameter descriptions and return values
  3. Provide usage examples for key functionalities
  4. Follow a consistent documentation style
  
  Include checks that verify all public interfaces are properly documented.
  ```

#### Milestone 15.2: User Guides
- **Task**: Create user guides for different usage scenarios
- **Validation**: Test that guides cover all main use cases
- **System Prompt**:
  ```
  Create user guides for the AlphaZero Multi-Game AI Engine. These guides should cover:
  1. Installation and setup on different platforms
  2. Playing against the AI for each game type
  3. Training your own models
  4. Using the API for custom applications
  
  Include reviews that verify the guides cover all main use cases and are easy to follow.
  ```

#### Milestone 15.3: Code Cleanup and Refactoring
- **Task**: Clean up code and apply final refactoring
- **Validation**: Test that code meets quality standards
- **System Prompt**:
  ```
  Clean up code and apply final refactoring to the AlphaZero Multi-Game AI Engine. This task should:
  1. Ensure consistent coding style across all components
  2. Remove any redundant or unused code
  3. Resolve any technical debt accumulated during development
  4. Improve naming and organization for clarity
  
  Include static analysis checks that verify the code meets quality standards.
  ```

#### Milestone 15.4: Release Package Preparation
- **Task**: Prepare release packages for all platforms
- **Validation**: Test that packages install and run correctly
- **System Prompt**:
  ```
  Prepare release packages for the AlphaZero Multi-Game AI Engine. These packages should:
  1. Support both Linux and Windows platforms
  2. Include all necessary components and dependencies
  3. Provide installation scripts or instructions
  4. Include pre-trained models for all supported games
  
  Include installation tests that verify the packages install and run correctly on all supported platforms.
  ```

## Implementation Template Samples

### C++ Class Implementation Template

```cpp
/**
 * @file [filename].h
 * @brief [brief description]
 * 
 * [detailed description]
 * 
 * @copyright Copyright (c) 2025
 */

#ifndef [HEADER_GUARD]
#define [HEADER_GUARD]

#include <vector>
#include <string>
#include <memory>

/**
 * @class [ClassName]
 * @brief [brief description]
 * 
 * [detailed description]
 */
class [ClassName] {
public:
    /**
     * @brief Constructor
     * 
     * @param [param] [description]
     */
    explicit [ClassName]([params]);
    
    /**
     * @brief Destructor
     */
    ~[ClassName]();
    
    // Rule of five
    [ClassName](const [ClassName]&) = [default/delete];
    [ClassName]& operator=(const [ClassName]&) = [default/delete];
    [ClassName]([ClassName]&&) noexcept = [default/delete];
    [ClassName]& operator=([ClassName]&&) noexcept = [default/delete];
    
    /**
     * @brief [method description]
     * 
     * @param [param] [description]
     * @return [return description]
     * @throws [exception] [when]
     */
    [return_type] [method_name]([params]);
    
private:
    // Private members
    [type] [member_name];
    
    /**
     * @brief [private method description]
     * 
     * @param [param] [description]
     * @return [return description]
     */
    [return_type] [private_method_name]([params]);
};

#endif // [HEADER_GUARD]
```

### Unit Test Template

```cpp
/**
 * @file [test_filename].cpp
 * @brief Unit tests for [class/feature]
 */

#include <gtest/gtest.h>
#include "[header_to_test].h"

class [TestFixtureName] : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test environment
    }
    
    void TearDown() override {
        // Clean up test environment
    }
    
    // Test fixture members
};

TEST_F([TestFixtureName], [TestName]) {
    // Arrange
    [type] [variable] = [value];
    
    // Act
    [type] [result] = [call_method_under_test];
    
    // Assert
    EXPECT_EQ([expected], [result]);
}

TEST_F([TestFixtureName], [EdgeCaseTestName]) {
    // Test edge case
    
    // Arrange
    [type] [variable] = [edge_case_value];
    
    // Act & Assert
    EXPECT_THROW([call_method_under_test], [exception_type]);
}

// Run tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### Python Training Script Template

```python
#!/usr/bin/env python3
"""
[brief description]

[detailed description]
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("[module_name]")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="[description]")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--model", type=str, help="Path to load/save model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    return parser.parse_args()

class [ModelClass](nn.Module):
    """[model description]"""
    
    def __init__(self, [params]):
        """Initialize the model."""
        super([ModelClass], self).__init__()
        # Model layers
        
    def forward(self, x):
        """Forward pass."""
        # Implementation
        return output

def train(model, dataloader, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        # Training logic
        
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Validation logic
    
    return total_loss / len(dataloader)

def main():
    """Main function."""
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data loading
    
    # Model setup
    
    # Training loop
    
    # Save results
    
if __name__ == "__main__":
    main()
```

These detailed milestones provide a comprehensive step-by-step plan for implementing the AlphaZero Multi-Game AI Engine. Each milestone focuses on a single functionality with clear validation criteria and system prompts for code generation. The templates provide a starting point for creating consistent, well-structured code throughout the project.

### 16. Final Testing and Validation (2 weeks)

#### Milestone 16.1: Regression Test Suite
- **Task**: Create comprehensive regression test suite
- **Validation**: Test that all features work correctly after changes
- **System Prompt**:
  ```
  Create a comprehensive regression test suite for the AlphaZero Multi-Game AI Engine. This suite should:
  1. Test all core functionalities across all supported games
  2. Include test cases for known edge cases and past bugs
  3. Verify API compatibility across versions
  4. Support automated execution in the CI pipeline
  
  Include test runs that verify all features continue to work correctly after changes.
  ```

#### Milestone 16.2: Game Rules Validation
- **Task**: Implement formal validation for game rules
- **Validation**: Test that all games follow official rules exactly
- **System Prompt**:
  ```
  Implement formal validation for game rules compliance. This validation should:
  1. Verify all Gomoku rules including Renju variations
  2. Test all Chess rules including special moves and draw conditions
  3. Validate Go rules for both Japanese and Chinese variations
  4. Include test positions from official rule books
  
  Include verification tests that confirm all games follow their official rules exactly.
  ```

#### Milestone 16.3: ELO Rating Verification
- **Task**: Verify achievement of ELO rating targets
- **Validation**: Test all games against benchmark opponents
- **System Prompt**:
  ```
  Implement ELO rating verification for all game AIs. This verification should:
  1. Test Gomoku AI to verify 2000+ ELO rating
  2. Test Chess AI to verify 2200+ ELO rating
  3. Test Go AI to verify 2000+ ELO rating
  4. Compare against established benchmark opponents
  
  Include tournament tests that verify each game AI achieves its target ELO rating.
  ```

#### Milestone 16.4: Performance Requirements Validation
- **Task**: Validate all performance requirements
- **Validation**: Test that all performance metrics meet targets
- **System Prompt**:
  ```
  Implement validation for all performance requirements. This validation should:
  1. Verify move decision latency targets for all games
  2. Test self-play throughput for training
  3. Measure resource utilization (CPU, GPU, memory)
  4. Validate model training efficiency
  
  Include benchmark tests that verify all performance metrics meet their specified targets.
  ```

### 17. Neural Network Architecture Refinement (2 weeks)

#### Milestone 17.1: DDW-RandWire Topology Optimization
- **Task**: Optimize the neural network graph topology
- **Validation**: Test performance against baseline network
- **System Prompt**:
  ```
  Optimize the DDW-RandWire network topology for the AlphaZero Multi-Game AI Engine. This optimization should:
  1. Experiment with different node counts and connectivity patterns
  2. Compare small-world vs. scale-free network generation
  3. Measure inference performance and playing strength
  4. Find optimal configuration for each game type
  
  Include comparative tests against baseline network configurations to verify improvements.
  ```

#### Milestone 17.2: Game-Specific Network Tuning
- **Task**: Tune neural network parameters for each game
- **Validation**: Test that tuned networks improve performance
- **System Prompt**:
  ```
  Tune neural network parameters specifically for each game. This tuning should:
  1. Adjust network size based on game complexity
  2. Optimize input channels for each game's features
  3. Fine-tune learning rates and regularization
  4. Balance policy and value head weights
  
  Include tests that verify tuned networks improve performance for their specific games.
  ```

#### Milestone 17.3: Enhanced Input Representation Refinement
- **Task**: Refine the enhanced input representation for each game
- **Validation**: Test that refined representations improve learning
- **System Prompt**:
  ```
  Refine the enhanced input representation for each game. This refinement should:
  1. Add game-specific feature planes based on domain knowledge
  2. Optimize the representation for learning efficiency
  3. Balance input complexity with performance
  4. Measure impact on playing strength
  
  Include tests that verify refined representations improve learning efficiency and playing strength.
  ```

#### Milestone 17.4: Model Export for Inference Optimization
- **Task**: Implement model export for optimized inference
- **Validation**: Test that exported models meet latency targets
- **System Prompt**:
  ```
  Implement model export for optimized inference. This implementation should:
  1. Support exporting to optimized formats (TorchScript, ONNX)
  2. Include quantization for reduced model size
  3. Optimize for inference on both CPU and GPU
  4. Maintain accuracy while improving speed
  
  Include performance tests that verify exported models meet the latency targets on target hardware.
  ```

### 18. Final Release and Deployment (2 weeks)

#### Milestone 18.1: Integration Testing with External Systems
- **Task**: Test integration with external systems via API
- **Validation**: Test that API works correctly with sample clients
- **System Prompt**:
  ```
  Implement integration testing with external systems. This testing should:
  1. Verify REST API functionality with sample clients
  2. Test Python bindings with example applications
  3. Validate command-line interface with scripts
  4. Check compatibility with game visualization tools
  
  Include tests that verify all external interfaces work correctly with sample client implementations.
  ```

#### Milestone 18.2: Model Distribution Package
- **Task**: Create package for model distribution
- **Validation**: Test that models can be distributed and loaded
- **System Prompt**:
  ```
  Create a package for model distribution. This package should:
  1. Include pre-trained models for all supported games
  2. Support versioning and compatibility checks
  3. Include metadata for model capabilities
  4. Allow efficient distribution and loading
  
  Include tests that verify models can be correctly distributed and loaded in different environments.
  ```

#### Milestone 18.3: Installation Package Creation
- **Task**: Create installation packages for all platforms
- **Validation**: Test installation on different systems
- **System Prompt**:
  ```
  Create installation packages for all supported platforms. These packages should:
  1. Support Windows and Linux installation
  2. Include all required dependencies
  3. Provide clear installation instructions
  4. Configure paths and permissions automatically
  
  Include installation tests on different systems to verify correct functionality.
  ```

#### Milestone 18.4: Final Release Checklist
- **Task**: Complete final release checklist
- **Validation**: Verify all release requirements are met
- **System Prompt**:
  ```
  Complete the final release checklist for the AlphaZero Multi-Game AI Engine. This checklist should:
  1. Verify all requirements from the PRD are met
  2. Ensure all documentation is complete and accurate
  3. Confirm all tests pass on all supported platforms
  4. Validate license compliance for all components
  
  Include a verification process that confirms all release requirements are met before final release.
  ```

## Additional Templates and Resources

### Game State Implementation Template

```cpp
/**
 * @file [game_name]_state.h
 * @brief Implementation of the [GameName]State class
 */

#ifndef [GAME_NAME]_STATE_H
#define [GAME_NAME]_STATE_H

#include "igamestate.h"
#include "zobrist_hash.h"
#include <vector>
#include <string>
#include <mutex>

/**
 * @class [GameName]State
 * @brief Game state implementation for [GameName]
 */
class [GameName]State : public IGameState {
public:
    // Constants
    static constexpr int DEFAULT_BOARD_SIZE = [size];
    
    /**
     * @brief Constructor
     * 
     * @param boardSize Board size
     * @param variantRules Whether to use variant rules
     */
    explicit [GameName]State(int boardSize = DEFAULT_BOARD_SIZE, bool variantRules = false);
    
    /**
     * @brief Copy constructor
     */
    [GameName]State(const [GameName]State& other);
    
    /**
     * @brief Move constructor
     */
    [GameName]State([GameName]State&& other) noexcept;
    
    /**
     * @brief Copy assignment operator
     */
    [GameName]State& operator=(const [GameName]State& other);
    
    /**
     * @brief Move assignment operator
     */
    [GameName]State& operator=([GameName]State&& other) noexcept;
    
    // IGameState interface implementation
    std::vector<int> getLegalMoves() const override;
    bool isLegalMove(int action) const override;
    void makeMove(int action) override;
    bool undoMove() override;
    bool isTerminal() const override;
    GameResult getGameResult() const override;
    int getCurrentPlayer() const override;
    int getBoardSize() const override;
    int getActionSpaceSize() const override;
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override;
    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override;
    uint64_t getHash() const override;
    std::unique_ptr<IGameState> clone() const override;
    std::string actionToString(int action) const override;
    std::optional<int> stringToAction(const std::string& moveStr) const override;
    std::string toString() const override;
    bool equals(const IGameState& other) const override;
    std::vector<int> getMoveHistory() const override;
    bool validate() const override;
    
    // [GameName]-specific methods
    
private:
    // Board representation
    int boardSize_;
    int currentPlayer_;
    
    // Game rules
    bool variantRules_;
    
    // Move history
    std::vector<int> moveHistory_;
    
    // Zobrist hashing
    ZobristHash zobrist_;
    uint64_t currentHash_;
    
    // Caching for optimization
    mutable std::mutex cacheMutex_;
    mutable bool resultCached_;
    mutable GameResult cachedResult_;
    mutable std::vector<int> cachedLegalMoves_;
    
    // Private helper methods
};

#endif // [GAME_NAME]_STATE_H
```

### Neural Network Training Loop Template

```python
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    batch_count = 0
    
    # Progress bar for monitoring
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (states, policies, values) in enumerate(progress_bar):
        # Move data to device
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        policy_logits, value_preds = model(states)
        
        # Calculate losses
        policy_loss = F.cross_entropy(policy_logits, policies)
        value_loss = F.mse_loss(value_preds.squeeze(-1), values)
        l2_reg = model.get_l2_regularization_loss()
        
        # Combined loss
        loss = policy_loss + value_loss + l2_reg
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * states.size(0)
        policy_loss_sum += policy_loss.item() * states.size(0)
        value_loss_sum += value_loss.item() * states.size(0)
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "p_loss": policy_loss.item(),
            "v_loss": value_loss.item()
        })
    
    # Update learning rate
    if scheduler is not None:
        scheduler.step()
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader.dataset)
    avg_policy_loss = policy_loss_sum / len(dataloader.dataset)
    avg_value_loss = value_loss_sum / len(dataloader.dataset)
    
    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss
    }
```

### MCTS Node Implementation Template

```cpp
/**
 * @file mcts_node.h
 * @brief Implementation of the Monte Carlo Tree Search node
 */

#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include "igamestate.h"

/**
 * @class MCTSNode
 * @brief Node in the Monte Carlo Tree Search tree
 */
class MCTSNode {
public:
    /**
     * @brief Constructor
     * 
     * @param state The game state this node represents (not owned)
     * @param parent The parent node (nullptr for root)
     * @param prior The prior probability from policy network
     */
    MCTSNode(const IGameState* state, MCTSNode* parent = nullptr, float prior = 0.0f);
    
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
    
    /**
     * @brief Get the value estimate for this node
     * 
     * @return The value estimate [-1,1]
     */
    float getValue() const;
    
    /**
     * @brief Get the UCB score for this node
     * 
     * @param cPuct Exploration constant
     * @param currentPlayer Current player
     * @param fpuReduction First play urgency reduction
     * @return The UCB score
     */
    float getUcbScore(float cPuct, int currentPlayer, float fpuReduction = 0.0f) const;
    
    /**
     * @brief Add virtual loss for parallel search
     * 
     * @param virtualLoss Amount of virtual loss to add
     */
    void addVirtualLoss(int virtualLoss);
    
    /**
     * @brief Remove virtual loss after search completes
     * 
     * @param virtualLoss Amount of virtual loss to remove
     */
    void removeVirtualLoss(int virtualLoss);
    
    /**
     * @brief Get the best action based on visit counts
     * 
     * @return The action with highest visit count
     */
    int getBestAction() const;
    
    /**
     * @brief Get action visit count distribution
     * 
     * @param temperature Temperature parameter for exploration
     * @return Vector of probabilities for each action
     */
    std::vector<float> getVisitCountDistribution(float temperature = 1.0f) const;
    
    /**
     * @brief Convert terminal game result to value
     * 
     * @param perspectivePlayer Player perspective for value
     * @return Value [-1,1] from perspective player's view
     */
    float getTerminalValue(int perspectivePlayer) const;
    
    /**
     * @brief Get debug string representation
     * 
     * @param maxDepth Maximum depth to print
     * @return String representation of the node and children
     */
    std::string toString(int maxDepth = 1) const;
};

#endif // MCTS_NODE_H
```

These templates and detailed milestones provide a comprehensive roadmap for implementing the AlphaZero Multi-Game AI Engine. Each milestone is focused on a single functionality with clear validation criteria, making the implementation process manageable and incremental.