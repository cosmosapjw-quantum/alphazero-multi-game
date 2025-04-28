#!/usr/bin/env python3
"""
Unit tests for Python bindings to the AlphaZero C++ implementation.
"""

import unittest
import os
import sys
import tempfile
import numpy as np

# Add parent directory to path to import pyalphazero
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pyalphazero as az
except ImportError:
    print("Error: pyalphazero module not found. Make sure it's properly built and installed.")
    sys.exit(1)

class TestGameState(unittest.TestCase):
    """Tests for game state bindings."""
    
    def test_create_game_state(self):
        """Test creating game states for different games."""
        # Test Gomoku
        gomoku_state = az.createGameState(az.GameType.GOMOKU, 15, False)
        self.assertEqual(gomoku_state.getGameType(), az.GameType.GOMOKU)
        self.assertEqual(gomoku_state.getBoardSize(), 15)
        self.assertEqual(gomoku_state.getCurrentPlayer(), 1)
        
        # Test Chess
        chess_state = az.createGameState(az.GameType.CHESS, 0, False)
        self.assertEqual(chess_state.getGameType(), az.GameType.CHESS)
        self.assertEqual(chess_state.getBoardSize(), 8)  # Chess is always 8x8
        self.assertEqual(chess_state.getCurrentPlayer(), 1)
        
        # Test Go
        go_state = az.createGameState(az.GameType.GO, 19, False)
        self.assertEqual(go_state.getGameType(), az.GameType.GO)
        self.assertEqual(go_state.getBoardSize(), 19)
        self.assertEqual(go_state.getCurrentPlayer(), 1)
    
    def test_game_state_moves(self):
        """Test making and undoing moves."""
        # Create a Gomoku state (smallest and simplest for testing)
        state = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Get legal moves
        legal_moves = state.getLegalMoves()
        self.assertEqual(len(legal_moves), 9 * 9)  # All positions are legal initially
        
        # Make a move
        center = 4 * 9 + 4  # Center position
        self.assertTrue(state.isLegalMove(center))
        state.makeMove(center)
        
        # Check state after move
        self.assertEqual(state.getCurrentPlayer(), 2)
        self.assertFalse(state.isLegalMove(center))  # Position now occupied
        
        # Undo the move
        self.assertTrue(state.undoMove())
        
        # Check state after undo
        self.assertEqual(state.getCurrentPlayer(), 1)
        self.assertTrue(state.isLegalMove(center))
        
        # Get move history
        self.assertEqual(len(state.getMoveHistory()), 0)
        
        # Make two moves
        state.makeMove(center)
        state.makeMove(center + 1)
        
        # Check move history
        history = state.getMoveHistory()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], center)
        self.assertEqual(history[1], center + 1)
    
    def test_game_state_tensor(self):
        """Test tensor representation."""
        # Create a Gomoku state
        state = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Get tensor representation
        tensor = state.getTensorRepresentation()
        self.assertIsInstance(tensor, list)
        self.assertGreater(len(tensor), 0)
        
        # Get enhanced tensor representation
        enhanced_tensor = state.getEnhancedTensorRepresentation()
        self.assertIsInstance(enhanced_tensor, list)
        self.assertGreater(len(enhanced_tensor), 0)
        
        # Make a move and check tensor again
        state.makeMove(40)  # Center position
        tensor = state.getTensorRepresentation()
        self.assertIsInstance(tensor, list)
        
        # Verify tensor has player 1's stone
        # First plane should represent player 1's stones
        self.assertGreater(tensor[0][4][4], 0.5)  # Should have a stone at center
    
    def test_game_state_hash(self):
        """Test Zobrist hashing for game states."""
        # Create two identical game states
        state1 = az.createGameState(az.GameType.GOMOKU, 9, False)
        state2 = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Initial hashes should be equal
        self.assertEqual(state1.getHash(), state2.getHash())
        
        # Make a move on state1
        state1.makeMove(40)
        
        # Hashes should now be different
        self.assertNotEqual(state1.getHash(), state2.getHash())
        
        # Make the same move on state2
        state2.makeMove(40)
        
        # Hashes should be equal again
        self.assertEqual(state1.getHash(), state2.getHash())


class TestMCTS(unittest.TestCase):
    """Tests for MCTS bindings."""
    
    def test_mcts_creation(self):
        """Test creating MCTS objects."""
        # Create a game state
        state = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Create transposition table
        tt = az.TranspositionTable(1024, 16)
        
        # Create MCTS without neural network (random policy)
        mcts = az.ParallelMCTS(state, None, tt, 1, 100)
        
        # Set MCTS parameters
        mcts.setCPuct(1.5)
        mcts.setFpuReduction(0.0)
        
        # Verify parameters
        self.assertEqual(mcts.getNumSimulations(), 100)
        self.assertEqual(mcts.getNumThreads(), 1)
    
    def test_mcts_search(self):
        """Test MCTS search functionality."""
        # Create a game state
        state = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Create transposition table
        tt = az.TranspositionTable(1024, 16)
        
        # Create MCTS with small number of simulations for testing
        mcts = az.ParallelMCTS(state, None, tt, 1, 10)
        
        # Run search
        mcts.search()
        
        # Get action probabilities
        probs = mcts.getActionProbabilities(1.0)
        self.assertEqual(len(probs), state.getActionSpaceSize())
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        
        # Select best action
        action = mcts.selectAction(False, 0.0)
        self.assertIsNotNone(action)
        self.assertTrue(state.isLegalMove(action))
    
    def test_mcts_update(self):
        """Test updating MCTS tree after a move."""
        # Create a game state
        state = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Create transposition table
        tt = az.TranspositionTable(1024, 16)
        
        # Create MCTS
        mcts = az.ParallelMCTS(state, None, tt, 1, 10)
        
        # Run search
        mcts.search()
        
        # Get root value before move
        root_value_before = mcts.getRootValue()
        
        # Select and make a move
        action = mcts.selectAction(False, 0.0)
        state.makeMove(action)
        
        # Update MCTS tree
        mcts.updateWithMove(action)
        
        # Run search again
        mcts.search()
        
        # Root value should be different (negated from opponent's perspective)
        root_value_after = mcts.getRootValue()
        
        # The values might not be exactly negated due to random rollouts,
        # but they should generally have opposite signs
        if abs(root_value_before) > 0.1:  # If value is significant
            self.assertNotEqual(np.sign(root_value_before), np.sign(root_value_after))


class TestTranspositionTable(unittest.TestCase):
    """Tests for TranspositionTable bindings."""
    
    def test_transposition_table(self):
        """Test transposition table functionality."""
        # Create transposition table
        tt = az.TranspositionTable(1024, 16)
        
        # Create a game state
        state = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Store a dummy policy and value
        policy = [1.0 / state.getActionSpaceSize()] * state.getActionSpaceSize()
        value = 0.5
        tt.store(state.getHash(), state.getGameType(), policy, value)
        
        # Create an entry to lookup
        entry = az.TranspositionTable.Entry()
        
        # Lookup the entry
        found = tt.lookup(state.getHash(), state.getGameType(), entry)
        self.assertTrue(found)
        self.assertEqual(entry.hash, state.getHash())
        self.assertEqual(entry.gameType, state.getGameType())
        self.assertAlmostEqual(entry.value, value, places=6)
        
        # Modify the state
        state.makeMove(40)
        
        # Lookup with new hash (should not find)
        found = tt.lookup(state.getHash(), state.getGameType(), entry)
        self.assertFalse(found)
        
        # Store new state
        tt.store(state.getHash(), state.getGameType(), policy, -value)
        
        # Lookup again
        found = tt.lookup(state.getHash(), state.getGameType(), entry)
        self.assertTrue(found)
        self.assertAlmostEqual(entry.value, -value, places=6)
    
    def test_transposition_table_stats(self):
        """Test transposition table statistics."""
        # Create transposition table
        tt = az.TranspositionTable(1024, 16)
        
        # Get initial stats
        self.assertEqual(tt.getHits(), 0)
        self.assertEqual(tt.getLookups(), 0)
        self.assertEqual(tt.getCollisions(), 0)
        self.assertEqual(tt.getReplacements(), 0)
        
        # Create a game state
        state = az.createGameState(az.GameType.GOMOKU, 9, False)
        
        # Store a dummy policy and value
        policy = [1.0 / state.getActionSpaceSize()] * state.getActionSpaceSize()
        value = 0.5
        tt.store(state.getHash(), state.getGameType(), policy, value)
        
        # Lookup that will succeed
        entry = az.TranspositionTable.Entry()
        tt.lookup(state.getHash(), state.getGameType(), entry)
        
        # Lookup that will fail
        state.makeMove(40)
        tt.lookup(state.getHash(), state.getGameType(), entry)
        
        # Check stats
        self.assertEqual(tt.getHits(), 1)
        self.assertEqual(tt.getLookups(), 2)
        self.assertEqual(tt.getHitRate(), 0.5)


class TestSelfPlay(unittest.TestCase):
    """Tests for self-play bindings."""
    
    def test_game_record(self):
        """Test game record functionality."""
        # Create a game record
        record = az.GameRecord(az.GameType.GOMOKU, 9, False)
        
        # Add moves
        record.addMove(40, [0.1] * 81, 0.5, 100)
        record.addMove(41, [0.1] * 81, -0.2, 120)
        
        # Set game result
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Get metadata
        game_type, board_size, variant = record.getMetadata()
        self.assertEqual(game_type, az.GameType.GOMOKU)
        self.assertEqual(board_size, 9)
        self.assertFalse(variant)
        
        # Get moves
        moves = record.getMoves()
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0].action, 40)
        self.assertEqual(moves[1].action, 41)
        
        # Check result
        self.assertEqual(record.getResult(), az.GameResult.WIN_PLAYER1)
        
        # Test serialization
        json_str = record.toJson()
        self.assertIsInstance(json_str, str)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Save to file
            record.saveToFile(temp_path)
            
            # Load from file
            loaded_record = az.GameRecord.loadFromFile(temp_path)
            
            # Check metadata
            loaded_type, loaded_size, loaded_variant = loaded_record.getMetadata()
            self.assertEqual(loaded_type, az.GameType.GOMOKU)
            self.assertEqual(loaded_size, 9)
            self.assertFalse(loaded_variant)
            
            # Check moves
            loaded_moves = loaded_record.getMoves()
            self.assertEqual(len(loaded_moves), 2)
            self.assertEqual(loaded_moves[0].action, 40)
            self.assertEqual(loaded_moves[1].action, 41)
            
            # Check result
            self.assertEqual(loaded_record.getResult(), az.GameResult.WIN_PLAYER1)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_dataset(self):
        """Test dataset functionality."""
        # Create dataset
        dataset = az.Dataset()
        
        # Create a game record
        record = az.GameRecord(az.GameType.GOMOKU, 9, False)
        record.addMove(40, [0.1] * 81, 0.5, 100)
        record.addMove(41, [0.1] * 81, -0.2, 120)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Add game record to dataset
        dataset.addGameRecord(record, True)  # Use enhanced features
        
        # Extract examples
        dataset.extractExamples(True)  # Include augmentations
        
        # Check size (should have examples from both moves plus augmentations)
        self.assertGreaterEqual(dataset.size(), 2)
        
        # Get random subset
        subset_size = min(5, dataset.size())
        subset = dataset.getRandomSubset(subset_size)
        self.assertEqual(len(subset), subset_size)
        
        # Verify example contents
        example = subset[0]
        self.assertIsInstance(example.state, list)
        self.assertIsInstance(example.policy, list)
        self.assertIsInstance(example.value, float)


if __name__ == "__main__":
    unittest.main()