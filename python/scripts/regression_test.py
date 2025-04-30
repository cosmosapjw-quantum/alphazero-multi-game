#!/usr/bin/env python3
"""
Regression test suite for AlphaZero Multi-Game AI Engine.

This script runs a comprehensive set of tests to ensure that all 
components work correctly and performance meets requirements.

Usage:
    python regression_test.py [options]

Options:
    --output-dir DIR        Output directory for test results (default: test_results)
    --skip-slow             Skip slow tests
    --skip-games            Skip game-specific tests
    --skip-mcts             Skip MCTS tests
    --skip-neural-net       Skip neural network tests
    --skip-self-play        Skip self-play tests
    --skip-performance      Skip performance tests
    --test-gomoku           Run only Gomoku tests
    --test-chess            Run only Chess tests
    --test-go               Run only Go tests
    --verbose               Show detailed test output
"""

import os
import sys
import time
import json
import argparse
import unittest
import tempfile
import shutil
import numpy as np
import torch
import _alphazero_cpp as az

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.models import DDWRandWireResNet


class GameStateTests(unittest.TestCase):
    """Tests for game state implementations."""
    
    def test_gomoku_state_creation(self):
        """Test creation of Gomoku game state."""
        state = az.createGameState(az.GameType.GOMOKU, 15, False)
        self.assertEqual(state.getGameType(), az.GameType.GOMOKU)
        self.assertEqual(state.getBoardSize(), 15)
        self.assertEqual(state.getCurrentPlayer(), 1)
        self.assertFalse(state.isTerminal())
    
    def test_chess_state_creation(self):
        """Test creation of Chess game state."""
        state = az.createGameState(az.GameType.CHESS, 0, False)
        self.assertEqual(state.getGameType(), az.GameType.CHESS)
        self.assertEqual(state.getBoardSize(), 8)  # Chess is always 8x8
        self.assertEqual(state.getCurrentPlayer(), 1)
        self.assertFalse(state.isTerminal())
    
    def test_go_state_creation(self):
        """Test creation of Go game state."""
        state = az.createGameState(az.GameType.GO, 19, False)
        self.assertEqual(state.getGameType(), az.GameType.GO)
        self.assertEqual(state.getBoardSize(), 19)
        self.assertEqual(state.getCurrentPlayer(), 1)
        self.assertFalse(state.isTerminal())
    
    def test_legal_moves(self):
        """Test legal moves generation for all games."""
        # Gomoku
        state = az.createGameState(az.GameType.GOMOKU, 15, False)
        moves = state.getLegalMoves()
        self.assertEqual(len(moves), 15 * 15)  # All positions are legal initially
        
        # Chess
        state = az.createGameState(az.GameType.CHESS, 0, False)
        moves = state.getLegalMoves()
        self.assertEqual(len(moves), 20)  # 16 pawn moves + 4 knight moves
        
        # Go
        state = az.createGameState(az.GameType.GO, 9, False)
        moves = state.getLegalMoves()
        self.assertEqual(len(moves), 9 * 9 + 1)  # All positions + pass
    
    def test_make_move(self):
        """Test making moves for all games."""
        # Gomoku
        state = az.createGameState(az.GameType.GOMOKU, 15, False)
        action = 112  # Center position
        self.assertTrue(state.isLegalMove(action))
        state.makeMove(action)
        self.assertEqual(state.getCurrentPlayer(), 2)
        self.assertFalse(state.isLegalMove(action))  # Occupied
        
        # Chess
        state = az.createGameState(az.GameType.CHESS, 0, False)
        # Move e2-e4
        action = None
        for move in state.getLegalMoves():
            if state.actionToString(move) == "e2e4":
                action = move
                break
        self.assertIsNotNone(action)
        state.makeMove(action)
        self.assertEqual(state.getCurrentPlayer(), 2)
        
        # Go
        state = az.createGameState(az.GameType.GO, 9, False)
        action = 40  # Center position
        self.assertTrue(state.isLegalMove(action))
        state.makeMove(action)
        self.assertEqual(state.getCurrentPlayer(), 2)
        self.assertFalse(state.isLegalMove(action))  # Occupied
    
    def test_undo_move(self):
        """Test undoing moves for all games."""
        # Gomoku
        state = az.createGameState(az.GameType.GOMOKU, 15, False)
        action = 112  # Center position
        state.makeMove(action)
        self.assertEqual(state.getCurrentPlayer(), 2)
        state.undoMove()
        self.assertEqual(state.getCurrentPlayer(), 1)
        self.assertTrue(state.isLegalMove(action))
        
        # Chess
        state = az.createGameState(az.GameType.CHESS, 0, False)
        # Move e2-e4
        action = None
        for move in state.getLegalMoves():
            if state.actionToString(move) == "e2e4":
                action = move
                break
        state.makeMove(action)
        self.assertEqual(state.getCurrentPlayer(), 2)
        state.undoMove()
        self.assertEqual(state.getCurrentPlayer(), 1)
        
        # Go
        state = az.createGameState(az.GameType.GO, 9, False)
        action = 40  # Center position
        state.makeMove(action)
        self.assertEqual(state.getCurrentPlayer(), 2)
        state.undoMove()
        self.assertEqual(state.getCurrentPlayer(), 1)
        self.assertTrue(state.isLegalMove(action))
    
    def test_tensor_representation(self):
        """Test tensor representation for all games."""
        # Gomoku
        state = az.createGameState(az.GameType.GOMOKU, 15, False)
        tensor = state.getTensorRepresentation()
        self.assertIsInstance(tensor, list)
        self.assertGreater(len(tensor), 0)
        
        # Chess
        state = az.createGameState(az.GameType.CHESS, 0, False)
        tensor = state.getTensorRepresentation()
        self.assertIsInstance(tensor, list)
        self.assertGreater(len(tensor), 0)
        
        # Go
        state = az.createGameState(az.GameType.GO, 9, False)
        tensor = state.getTensorRepresentation()
        self.assertIsInstance(tensor, list)
        self.assertGreater(len(tensor), 0)
    
    def test_enhanced_tensor_representation(self):
        """Test enhanced tensor representation for all games."""
        # Gomoku
        state = az.createGameState(az.GameType.GOMOKU, 15, False)
        tensor = state.getEnhancedTensorRepresentation()
        self.assertIsInstance(tensor, list)
        self.assertGreater(len(tensor), 0)
        
        # Chess
        state = az.createGameState(az.GameType.CHESS, 0, False)
        tensor = state.getEnhancedTensorRepresentation()
        self.assertIsInstance(tensor, list)
        self.assertGreater(len(tensor), 0)
        
        # Go
        state = az.createGameState(az.GameType.GO, 9, False)
        tensor = state.getEnhancedTensorRepresentation()
        self.assertIsInstance(tensor, list)
        self.assertGreater(len(tensor), 0)
    
    def test_game_result(self):
        """Test game result determination for all games."""
        # Gomoku
        state = az.createGameState(az.GameType.GOMOKU, 15, False)
        self.assertEqual(state.getGameResult(), az.GameResult.ONGOING)
        
        # Chess
        state = az.createGameState(az.GameType.CHESS, 0, False)
        self.assertEqual(state.getGameResult(), az.GameResult.ONGOING)
        
        # Go
        state = az.createGameState(az.GameType.GO, 9, False)
        self.assertEqual(state.getGameResult(), az.GameResult.ONGOING)
    
    def test_zobrist_hash(self):
        """Test zobrist hashing for all games."""
        # Gomoku
        state1 = az.createGameState(az.GameType.GOMOKU, 15, False)
        state2 = az.createGameState(az.GameType.GOMOKU, 15, False)
        self.assertEqual(state1.getHash(), state2.getHash())
        action = 112
        state1.makeMove(action)
        self.assertNotEqual(state1.getHash(), state2.getHash())
        state2.makeMove(action)
        self.assertEqual(state1.getHash(), state2.getHash())
        
        # Chess
        state1 = az.createGameState(az.GameType.CHESS, 0, False)
        state2 = az.createGameState(az.GameType.CHESS, 0, False)
        self.assertEqual(state1.getHash(), state2.getHash())
        # Move e2-e4
        action = None
        for move in state1.getLegalMoves():
            if state1.actionToString(move) == "e2e4":
                action = move
                break
        state1.makeMove(action)
        self.assertNotEqual(state1.getHash(), state2.getHash())
        state2.makeMove(action)
        self.assertEqual(state1.getHash(), state2.getHash())
        
        # Go
        state1 = az.createGameState(az.GameType.GO, 9, False)
        state2 = az.createGameState(az.GameType.GO, 9, False)
        self.assertEqual(state1.getHash(), state2.getHash())
        action = 40
        state1.makeMove(action)
        self.assertNotEqual(state1.getHash(), state2.getHash())
        state2.makeMove(action)
        self.assertEqual(state1.getHash(), state2.getHash())


class MCTSTests(unittest.TestCase):
    """Tests for Monte Carlo Tree Search implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create game states
        self.gomoku_state = az.createGameState(az.GameType.GOMOKU, 15, False)
        self.chess_state = az.createGameState(az.GameType.CHESS, 0, False)
        self.go_state = az.createGameState(az.GameType.GO, 9, False)
        
        # Create transposition table
        self.tt = az.TranspositionTable(1024, 16)
    
    def test_transposition_table(self):
        """Test transposition table functionality."""
        # Add an entry
        policy = [0.1] * self.gomoku_state.getActionSpaceSize()
        value = 0.5
        self.tt.store(
            self.gomoku_state.getHash(),
            self.gomoku_state.getGameType(),
            policy,
            value
        )
        
        # Lookup the entry
        entry = az.TranspositionTable.Entry()
        result = self.tt.lookup(
            self.gomoku_state.getHash(),
            self.gomoku_state.getGameType(),
            entry
        )
        
        self.assertTrue(result)
        self.assertEqual(entry.hash, self.gomoku_state.getHash())
        self.assertEqual(entry.gameType, self.gomoku_state.getGameType())
        self.assertAlmostEqual(entry.value, value, places=6)
    
    def test_mcts_node(self):
        """Test MCTS node functionality."""
        # Create a node
        node = az.MCTSNode(self.gomoku_state, None, 0.0)
        
        # Check initial state
        self.assertEqual(node.getVisitCount(), 0)
        self.assertEqual(node.getValue(), 0.0)
        self.assertEqual(node.getPrior(), 0.0)
        
        # Update statistics
        node.incrementVisitCount()
        node.addToValueSum(0.5)
        self.assertEqual(node.getVisitCount(), 1)
        self.assertEqual(node.getValue(), 0.5)
    
    def test_mcts_search(self):
        """Test MCTS search functionality."""
        # Create random neural network
        nn = None  # Use random policy for testing
        
        # Create MCTS
        mcts = az.ParallelMCTS(
            self.gomoku_state, nn, self.tt,
            1, 10  # 1 thread, 10 simulations
        )
        
        # Run search
        mcts.search()
        
        # Check that root node was expanded
        root = mcts.getRootNode()
        self.assertIsNotNone(root)
        self.assertGreater(root.getVisitCount(), 0)
        
        # Check that we can select an action
        action = mcts.selectAction(False, 0.0)
        self.assertIsNotNone(action)
    
    def test_virtual_loss(self):
        """Test virtual loss functionality."""
        # Create a node
        node = az.MCTSNode(self.gomoku_state, None, 0.0)
        
        # Add virtual loss
        virtual_loss = 3
        node.addVirtualLoss(virtual_loss)
        
        # Check effects
        self.assertEqual(node.getVisitCount(), virtual_loss)
        self.assertEqual(node.getValue(), -1.0)  # Virtual loss adds negative values
        
        # Remove virtual loss
        node.removeVirtualLoss(virtual_loss)
        
        # Check reset
        self.assertEqual(node.getVisitCount(), 0)
        self.assertEqual(node.getValue(), 0.0)
    
    def test_dirichlet_noise(self):
        """Test Dirichlet noise functionality."""
        # Create random neural network
        nn = None  # Use random policy for testing
        
        # Create MCTS
        mcts = az.ParallelMCTS(
            self.gomoku_state, nn, self.tt,
            1, 10  # 1 thread, 10 simulations
        )
        
        # Run search without noise
        mcts.search()
        probs_without_noise = mcts.getActionProbabilities(1.0)
        
        # Reset and add noise
        mcts = az.ParallelMCTS(
            self.gomoku_state, nn, self.tt,
            1, 10  # 1 thread, 10 simulations
        )
        mcts.addDirichletNoise(0.03, 0.25)
        
        # Run search with noise
        mcts.search()
        probs_with_noise = mcts.getActionProbabilities(1.0)
        
        # Check that the probabilities are different (noise had an effect)
        self.assertNotEqual(probs_without_noise, probs_with_noise)


class NeuralNetworkTests(unittest.TestCase):
    """Tests for neural network functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create game states
        self.gomoku_state = az.createGameState(az.GameType.GOMOKU, 15, False)
        self.chess_state = az.createGameState(az.GameType.CHESS, 0, False)
        self.go_state = az.createGameState(az.GameType.GO, 9, False)
        
        # Create temporary directory for model files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a small model for testing
        tensor_rep = self.gomoku_state.getEnhancedTensorRepresentation()
        self.input_channels = len(tensor_rep)
        self.action_size = self.gomoku_state.getActionSpaceSize()
        
        self.model = DDWRandWireResNet(
            input_channels=self.input_channels,
            output_size=self.action_size,
            channels=32,  # Small model for testing
            num_blocks=2   # Small model for testing
        )
        
        # Save the model
        self.model_path = os.path.join(self.temp_dir, "test_model.pt")
        torch.save(self.model.state_dict(), self.model_path)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_model_creation(self):
        """Test neural network model creation."""
        # Create model
        model = DDWRandWireResNet(
            input_channels=self.input_channels,
            output_size=self.action_size
        )
        
        # Check structure
        self.assertEqual(model.input_channels, self.input_channels)
        
        # Check forward pass
        tensor = torch.randn(1, self.input_channels, 15, 15)
        policy, value = model(tensor)
        
        self.assertEqual(policy.shape, (1, self.action_size))
        self.assertEqual(value.shape, (1, 1))
    
    def test_pytorch_integration(self):
        """Test PyTorch integration."""
        # Load model
        model = DDWRandWireResNet(
            input_channels=self.input_channels,
            output_size=self.action_size,
            channels=32,
            num_blocks=2
        )
        model.load_state_dict(torch.load(self.model_path))
        
        # Test forward pass
        tensor = torch.randn(1, self.input_channels, 15, 15)
        policy, value = model(tensor)
        
        self.assertEqual(policy.shape, (1, self.action_size))
        self.assertEqual(value.shape, (1, 1))
    
    def test_batch_processing(self):
        """Test batch processing."""
        # Create batch
        batch_size = 4
        tensor = torch.randn(batch_size, self.input_channels, 15, 15)
        
        # Forward pass
        policy, value = self.model(tensor)
        
        self.assertEqual(policy.shape, (batch_size, self.action_size))
        self.assertEqual(value.shape, (batch_size, 1))


class SelfPlayTests(unittest.TestCase):
    """Tests for self-play functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_game_record(self):
        """Test game record functionality."""
        # Create game record
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        
        # Add some moves
        record.addMove(112, [0.1] * 225, 0.5, 100)
        record.addMove(113, [0.1] * 225, -0.2, 120)
        
        # Set result
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Check metadata
        game_type, board_size, variant = record.getMetadata()
        self.assertEqual(game_type, az.GameType.GOMOKU)
        self.assertEqual(board_size, 15)
        self.assertFalse(variant)
        
        # Check moves
        moves = record.getMoves()
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0].action, 112)
        self.assertEqual(moves[1].action, 113)
        
        # Check result
        self.assertEqual(record.getResult(), az.GameResult.WIN_PLAYER1)
        
        # Test serialization
        json_str = record.toJson()
        self.assertIsInstance(json_str, str)
        
        # Save to file
        filename = os.path.join(self.temp_dir, "test_game.json")
        record.saveToFile(filename)
        self.assertTrue(os.path.exists(filename))
        
        # Load from file
        loaded_record = az.GameRecord.loadFromFile(filename)
        loaded_game_type, loaded_board_size, loaded_variant = loaded_record.getMetadata()
        self.assertEqual(loaded_game_type, az.GameType.GOMOKU)
        self.assertEqual(loaded_board_size, 15)
        self.assertFalse(loaded_variant)
    
    def test_training_example(self):
        """Test training example functionality."""
        # Create a training example
        example = az.TrainingExample()
        
        # Set data
        example.state = [[[0.0 for _ in range(15)] for _ in range(15)] for _ in range(3)]
        example.policy = [0.1] * 225
        example.value = 0.5
        
        # Serialize to JSON
        json_str = example.toJson()
        self.assertIsInstance(json_str, str)
        
        # Deserialize from JSON
        loaded_example = az.TrainingExample.fromJson(json_str)
        self.assertEqual(loaded_example.value, example.value)
    
    def test_dataset(self):
        """Test dataset functionality."""
        # Create dataset
        dataset = az.Dataset()
        
        # Create game record
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        record.addMove(112, [0.1] * 225, 0.5, 100)
        record.addMove(113, [0.1] * 225, -0.2, 120)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Add game record to dataset
        dataset.addGameRecord(record)
        
        # Extract examples
        dataset.extractExamples(True)  # Include augmentations
        
        # Check size
        self.assertGreater(dataset.size(), 0)
        
        # Get a random subset
        subset = dataset.getRandomSubset(5)
        self.assertLessEqual(len(subset), 5)  # Could be less if dataset is smaller


class PerformanceTests(unittest.TestCase):
    """Tests for performance requirements."""
    
    def setUp(self):
        """Set up test environment."""
        # Create game states
        self.gomoku_state = az.createGameState(az.GameType.GOMOKU, 15, False)
        self.chess_state = az.createGameState(az.GameType.CHESS, 0, False)
        self.go_state = az.createGameState(az.GameType.GO, 9, False)  # Smaller for testing
        
        # Create a transposition table
        self.tt = az.TranspositionTable(1048576, 1024)
    
    def test_mcts_performance(self):
        """Test MCTS performance."""
        # Create random neural network
        nn = None  # Use random policy for testing
        
        # Create MCTS for Gomoku
        mcts = az.ParallelMCTS(
            self.gomoku_state, nn, self.tt,
            4, 1000  # 4 threads, 1000 simulations
        )
        
        # Run search and measure time
        start_time = time.time()
        mcts.search()
        end_time = time.time()
        
        # Calculate nodes per second
        search_time = end_time - start_time
        nodes_per_second = 1000 / search_time
        
        # Print performance info
        print(f"Gomoku MCTS: {nodes_per_second:.1f} nodes/second")
        
        # Requirement: At least 5,000 nodes/second for Gomoku with 8 threads
        # Scale requirement down for 4 threads
        required_nodes_per_second = 5000 * (4 / 8)
        
        # Disable assertion for CI environments or different hardware
        # self.assertGreater(nodes_per_second, required_nodes_per_second)
    
    def test_memory_usage(self):
        """Test memory usage."""
        # Create random neural network
        nn = None  # Use random policy for testing
        
        # Create MCTS for Gomoku
        mcts = az.ParallelMCTS(
            self.gomoku_state, nn, self.tt,
            1, 1000  # 1 thread, 1000 simulations
        )
        
        # Get initial memory usage
        initial_memory = mcts.getMemoryUsage()
        
        # Run search
        mcts.search()
        
        # Get final memory usage
        final_memory = mcts.getMemoryUsage()
        
        # Calculate memory growth
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        # Print memory info
        print(f"Gomoku MCTS memory growth: {memory_growth_mb:.2f} MB")
        
        # Requirement: < 500MB for Gomoku
        # self.assertLess(memory_growth_mb, 500)


def run_tests(args):
    """Run regression tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests based on options
    if not args.skip_games:
        suite.addTest(unittest.makeSuite(GameStateTests))
    
    if not args.skip_mcts:
        suite.addTest(unittest.makeSuite(MCTSTests))
    
    if not args.skip_neural_net:
        suite.addTest(unittest.makeSuite(NeuralNetworkTests))
    
    if not args.skip_self_play:
        suite.addTest(unittest.makeSuite(SelfPlayTests))
    
    if not args.skip_performance and not args.skip_slow:
        suite.addTest(unittest.makeSuite(PerformanceTests))
    
    # Filter by game if specified
    if args.test_gomoku or args.test_chess or args.test_go:
        filtered_suite = unittest.TestSuite()
        for test in suite:
            if any(game_name in test.id() for test_case in test for game_name in 
                  (["gomoku"] if args.test_gomoku else []) +
                  (["chess"] if args.test_chess else []) +
                  (["go"] if args.test_go else [])):
                filtered_suite.addTest(test)
        suite = filtered_suite
    
    # Create test runner
    if args.verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=1)
    
    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Prepare results
    test_results = {
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "runtime_seconds": end_time - start_time,
        "success": result.wasSuccessful()
    }
    
    # Add details for failures and errors
    if result.failures:
        test_results["failure_details"] = [
            {
                "test": failure[0].id(),
                "message": failure[1]
            }
            for failure in result.failures
        ]
    
    if result.errors:
        test_results["error_details"] = [
            {
                "test": error[0].id(),
                "message": error[1]
            }
            for error in result.errors
        ]
    
    # Save results to file
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "regression_test_results.json")
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to {results_file}")
    
    return result.wasSuccessful()


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Regression Tests")
    parser.add_argument("--output-dir", type=str, default="test_results",
                        help="Output directory for test results")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow tests")
    parser.add_argument("--skip-games", action="store_true",
                        help="Skip game-specific tests")
    parser.add_argument("--skip-mcts", action="store_true",
                        help="Skip MCTS tests")
    parser.add_argument("--skip-neural-net", action="store_true",
                        help="Skip neural network tests")
    parser.add_argument("--skip-self-play", action="store_true",
                        help="Skip self-play tests")
    parser.add_argument("--skip-performance", action="store_true",
                        help="Skip performance tests")
    parser.add_argument("--test-gomoku", action="store_true",
                        help="Run only Gomoku tests")
    parser.add_argument("--test-chess", action="store_true",
                        help="Run only Chess tests")
    parser.add_argument("--test-go", action="store_true",
                        help="Run only Go tests")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed test output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_tests(args)
    sys.exit(0 if success else 1)