#!/usr/bin/env python3
"""
Unit tests for the AlphaZero training system.

This module tests the training pipeline, including:
- GameRecord and training data extraction
- Dataset handling
- Loss function
- Neural network training
- Self-play integration
"""

import os
import sys
import tempfile
import unittest
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AlphaZero components
import pyalphazero as az
from alphazero.models import DDWRandWireResNet
from alphazero.training import AlphaZeroLoss
from alphazero.training.dataset import AlphaZeroDataset, GameDatasetBuilder
from alphazero.utils.elo import EloRating, calculate_elo_change


class TestGameRecord(unittest.TestCase):
    """Test the GameRecord class for storing self-play data."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_game_record_creation(self):
        """Test creating a GameRecord."""
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        
        # Check initial state
        game_type, board_size, variant = record.getMetadata()
        self.assertEqual(game_type, az.GameType.GOMOKU)
        self.assertEqual(board_size, 15)
        self.assertFalse(variant)
        self.assertEqual(record.getResult(), az.GameResult.ONGOING)

    def test_adding_moves(self):
        """Test adding moves to a GameRecord."""
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        
        # Create mock policy and add moves
        policy = [1.0 / 225] * 225  # Uniform policy for 15x15 board
        record.addMove(112, policy, 0.5, 100)  # Action, policy, value, time
        record.addMove(113, policy, -0.3, 120)
        
        # Check moves
        moves = record.getMoves()
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0].action, 112)
        self.assertEqual(moves[1].action, 113)
        self.assertAlmostEqual(moves[0].value, 0.5)
        self.assertAlmostEqual(moves[1].value, -0.3)

    def test_setting_result(self):
        """Test setting game result."""
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        
        # Set result
        record.setResult(az.GameResult.WIN_PLAYER1)
        self.assertEqual(record.getResult(), az.GameResult.WIN_PLAYER1)
        
        # Change result
        record.setResult(az.GameResult.DRAW)
        self.assertEqual(record.getResult(), az.GameResult.DRAW)

    def test_serialization(self):
        """Test serialization and deserialization of GameRecord."""
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        policy = [1.0 / 225] * 225
        record.addMove(112, policy, 0.5, 100)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Save to file
        filename = os.path.join(self.temp_dir, "test_game.json")
        record.saveToFile(filename)
        
        # Load from file
        loaded = az.GameRecord.loadFromFile(filename)
        
        # Check metadata
        game_type, board_size, variant = loaded.getMetadata()
        self.assertEqual(game_type, az.GameType.GOMOKU)
        self.assertEqual(board_size, 15)
        self.assertFalse(variant)
        
        # Check moves and result
        moves = loaded.getMoves()
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].action, 112)
        self.assertEqual(loaded.getResult(), az.GameResult.WIN_PLAYER1)

    def test_json_conversion(self):
        """Test JSON conversion of GameRecord."""
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        policy = [1.0 / 225] * 225
        record.addMove(112, policy, 0.5, 100)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Convert to JSON
        json_str = record.toJson()
        
        # Load from JSON
        loaded = az.GameRecord.fromJson(json_str)
        
        # Check metadata
        game_type, board_size, variant = loaded.getMetadata()
        self.assertEqual(game_type, az.GameType.GOMOKU)
        self.assertEqual(board_size, 15)
        self.assertFalse(variant)
        
        # Check moves and result
        moves = loaded.getMoves()
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].action, 112)
        self.assertEqual(loaded.getResult(), az.GameResult.WIN_PLAYER1)


class TestTrainingExample(unittest.TestCase):
    """Test the TrainingExample class."""

    def test_example_creation(self):
        """Test creating a TrainingExample."""
        example = az.TrainingExample()
        
        # Set fields
        example.state = [[[0 for _ in range(15)] for _ in range(15)] for _ in range(3)]
        example.policy = [1.0 / 225] * 225
        example.value = 0.5
        
        # Check fields
        self.assertEqual(len(example.state), 3)  # 3 feature planes
        self.assertEqual(len(example.policy), 225)  # 15x15 action space
        self.assertEqual(example.value, 0.5)

    def test_json_conversion(self):
        """Test JSON conversion of TrainingExample."""
        example = az.TrainingExample()
        example.state = [[[0 for _ in range(15)] for _ in range(15)] for _ in range(3)]
        example.policy = [1.0 / 225] * 225
        example.value = 0.5
        
        # Convert to JSON
        json_str = example.toJson()
        
        # Load from JSON
        loaded = az.TrainingExample.fromJson(json_str)
        
        # Check fields
        self.assertEqual(len(loaded.state), 3)
        self.assertEqual(len(loaded.policy), 225)
        self.assertEqual(loaded.value, 0.5)


class TestDataset(unittest.TestCase):
    """Test the Dataset class for handling training examples."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_creation(self):
        """Test creating a Dataset."""
        dataset = az.Dataset()
        self.assertEqual(dataset.size(), 0)

    def test_adding_game_record(self):
        """Test adding a GameRecord to Dataset."""
        dataset = az.Dataset()
        
        # Create a GameRecord
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        policy = [1.0 / 225] * 225
        record.addMove(112, policy, 0.5, 100)
        record.addMove(113, policy, -0.3, 120)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Add to dataset
        dataset.addGameRecord(record, True)  # Use enhanced features
        
        # Extract examples (without augmentations)
        dataset.extractExamples(False)
        
        # Check size (should have 2 examples for 2 moves)
        self.assertEqual(dataset.size(), 2)

    def test_augmentation(self):
        """Test data augmentation in Dataset."""
        dataset = az.Dataset()
        
        # Create a GameRecord
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        policy = [1.0 / 225] * 225
        record.addMove(112, policy, 0.5, 100)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Add to dataset
        dataset.addGameRecord(record, True)
        
        # Extract examples with augmentations
        dataset.extractExamples(True)
        
        # Check size (should have augmentations)
        # For Gomoku, typically 8 augmentations (4 rotations * 2 reflections)
        self.assertGreaterEqual(dataset.size(), 1)

    def test_random_subset(self):
        """Test getting a random subset of examples."""
        dataset = az.Dataset()
        
        # Create a GameRecord with 10 moves
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        policy = [1.0 / 225] * 225
        for i in range(10):
            record.addMove(i, policy, 0.0, 100)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Add to dataset
        dataset.addGameRecord(record, True)
        
        # Extract examples
        dataset.extractExamples(False)
        
        # Get random subset
        subset = dataset.getRandomSubset(5)
        self.assertEqual(len(subset), 5)

    def test_save_load(self):
        """Test saving and loading a Dataset."""
        dataset = az.Dataset()
        
        # Create a GameRecord
        record = az.GameRecord(az.GameType.GOMOKU, 15, False)
        policy = [1.0 / 225] * 225
        record.addMove(112, policy, 0.5, 100)
        record.setResult(az.GameResult.WIN_PLAYER1)
        
        # Add to dataset and extract examples
        dataset.addGameRecord(record, True)
        dataset.extractExamples(False)
        
        # Save to file
        filename = os.path.join(self.temp_dir, "test_dataset.bin")
        dataset.saveToFile(filename)
        
        # Create new dataset and load
        loaded_dataset = az.Dataset()
        loaded_dataset.loadFromFile(filename)
        
        # Check size
        self.assertEqual(loaded_dataset.size(), dataset.size())


class TestPytorchDataset(unittest.TestCase):
    """Test the PyTorch Dataset implementation."""

    def test_alphazero_dataset(self):
        """Test AlphaZeroDataset class."""
        # Create some example data
        states = [np.random.rand(3, 15, 15) for _ in range(5)]
        policies = [np.random.rand(225) for _ in range(5)]
        values = [np.random.uniform(-1, 1) for _ in range(5)]
        
        # Normalize policies
        for i in range(5):
            policies[i] = policies[i] / policies[i].sum()
        
        # Create examples
        examples = []
        for i in range(5):
            example = az.TrainingExample()
            example.state = states[i].tolist()
            example.policy = policies[i].tolist()
            example.value = values[i]
            examples.append(example)
        
        # Create dataset
        dataset = AlphaZeroDataset(examples)
        
        # Check length
        self.assertEqual(len(dataset), 5)
        
        # Check getitem
        state, policy, value = dataset[0]
        self.assertIsInstance(state, torch.Tensor)
        self.assertIsInstance(policy, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(state.shape, torch.Size([3, 15, 15]))
        self.assertEqual(policy.shape, torch.Size([225]))
        self.assertEqual(value.shape, torch.Size([1]))


class TestGameDatasetBuilder(unittest.TestCase):
    """Test the GameDatasetBuilder class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample game records
        self.records = []
        for i in range(3):
            record = az.GameRecord(az.GameType.GOMOKU, 15, False)
            policy = [1.0 / 225] * 225
            record.addMove(112, policy, 0.5, 100)
            record.addMove(113, policy, -0.3, 120)
            record.setResult(az.GameResult.WIN_PLAYER1)
            
            # Save to file
            filename = os.path.join(self.temp_dir, f"game_{i}.json")
            record.saveToFile(filename)
            
            self.records.append(record)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_builder_creation(self):
        """Test creating a GameDatasetBuilder."""
        builder = GameDatasetBuilder(az.GameType.GOMOKU, True, True)
        self.assertIsNotNone(builder)

    def test_add_game_record(self):
        """Test adding a game record to builder."""
        builder = GameDatasetBuilder(az.GameType.GOMOKU, True, True)
        builder.add_game_record(self.records[0])
        
        # Extract examples
        examples = builder.extract_examples()
        self.assertGreater(len(examples), 0)

    def test_add_games_from_directory(self):
        """Test adding games from a directory."""
        builder = GameDatasetBuilder(az.GameType.GOMOKU, True, False)
        count = builder.add_games_from_directory(self.temp_dir)
        
        # Should find 3 games
        self.assertEqual(count, 3)
        
        # Extract examples (2 moves per game, no augmentation)
        examples = builder.extract_examples()
        self.assertEqual(len(examples), 6)

    def test_build_torch_dataset(self):
        """Test building a PyTorch dataset."""
        builder = GameDatasetBuilder(az.GameType.GOMOKU, True, False)
        builder.add_games_from_directory(self.temp_dir)
        
        # Build PyTorch dataset
        dataset = builder.build_torch_dataset()
        self.assertEqual(len(dataset), 6)

    def test_create_data_loader(self):
        """Test creating a data loader."""
        builder = GameDatasetBuilder(az.GameType.GOMOKU, True, False)
        builder.add_games_from_directory(self.temp_dir)
        
        # Create data loader
        loader = builder.create_data_loader(batch_size=2)
        
        # Check batch
        for state, policy, value in loader:
            self.assertEqual(state.shape[0], 2)  # Batch size
            self.assertEqual(state.shape[1], 3)  # Feature planes (assuming 3 planes)
            break


class TestLossFunction(unittest.TestCase):
    """Test the AlphaZero loss function."""

    def test_loss_calculation(self):
        """Test calculating the AlphaZero loss."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Create loss function
        loss_fn = AlphaZeroLoss(l2_reg=1e-4)
        
        # Create fake data
        policy_output = torch.randn(5, 10)  # 5 samples, 10 actions
        value_output = torch.randn(5, 1)    # 5 samples, 1 value
        policy_target = torch.softmax(torch.randn(5, 10), dim=1)  # Target policy
        value_target = torch.empty(5).uniform_(-1, 1)  # Target value
        
        # Calculate loss
        total_loss, policy_loss, value_loss, l2_loss = loss_fn(
            policy_output, value_output, policy_target, value_target, model
        )
        
        # Check that loss values are reasonable
        self.assertGreater(policy_loss.item(), 0)
        self.assertGreater(value_loss.item(), 0)
        self.assertGreater(l2_loss.item(), 0)
        self.assertAlmostEqual(
            total_loss.item(),
            policy_loss.item() + value_loss.item() + l2_loss.item(),
            places=5
        )


class TestNeuralNetworkTraining(unittest.TestCase):
    """Test neural network training."""

    def setUp(self):
        """Skip tests if CUDA is not available to prevent test failures."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping training tests")

    def test_model_training(self):
        """Test training a neural network model."""
        # Create a small model for testing
        model = DDWRandWireResNet(
            input_channels=3,      # Sample: 3 input channels
            output_size=225,       # Sample: 15x15 board
            channels=16,           # Very small for testing
            num_blocks=1           # Very small for testing
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create loss function
        loss_fn = AlphaZeroLoss(l2_reg=1e-4)
        
        # Create fake training data
        batch_size = 4
        policy_target = torch.softmax(torch.randn(batch_size, 225), dim=1).to(device)
        value_target = torch.empty(batch_size).uniform_(-1, 1).to(device)
        
        # Initial loss
        model.train()
        input_tensor = torch.randn(batch_size, 3, 15, 15).to(device)
        policy_output, value_output = model(input_tensor)
        
        initial_loss, _, _, _ = loss_fn(
            policy_output, value_output, policy_target, value_target, model
        )
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            policy_output, value_output = model(input_tensor)
            loss, _, _, _ = loss_fn(
                policy_output, value_output, policy_target, value_target, model
            )
            loss.backward()
            optimizer.step()
        
        # Final loss
        model.eval()
        with torch.no_grad():
            policy_output, value_output = model(input_tensor)
            final_loss, _, _, _ = loss_fn(
                policy_output, value_output, policy_target, value_target, model
            )
        
        # Check that loss decreased
        self.assertLess(final_loss.item(), initial_loss.item())


class TestEloRating(unittest.TestCase):
    """Test the ELO rating system."""

    def test_elo_rating_initialization(self):
        """Test initializing the ELO rating tracker."""
        elo = EloRating(initial_rating=1500.0, k_factor=32.0)
        self.assertEqual(elo.initial_rating, 1500.0)
        self.assertEqual(elo.k_factor, 32.0)

    def test_get_rating(self):
        """Test getting a player's rating."""
        elo = EloRating()
        self.assertEqual(elo.get_rating("player1"), 1500.0)  # Default rating

    def test_add_game_result(self):
        """Test adding a game result."""
        elo = EloRating()
        
        # Player1 wins
        new_rating = elo.add_game_result("player1", "player2", 1.0)
        
        # Check new rating
        self.assertGreater(new_rating, 1500.0)
        self.assertLess(elo.get_rating("player2"), 1500.0)

    def test_add_match_results(self):
        """Test adding match results."""
        elo = EloRating()
        
        # Player1: 2 wins, 1 draw, 0 losses
        final_rating, rating_change = elo.add_match_results(
            "player1", "player2", 2, 1, 0
        )
        
        # Check rating change
        self.assertGreater(rating_change, 0)
        self.assertEqual(final_rating, elo.get_rating("player1"))

    def test_calculation(self):
        """Test ELO calculation formula."""
        # Expected score for equal ratings
        expected = calculate_elo_change(1500, 1500, 1.0, k_factor=32)
        self.assertEqual(expected, 16.0)  # Should be K_factor/2
        
        # Win against higher-rated player gives more points
        change_vs_stronger = calculate_elo_change(1500, 1600, 1.0)
        change_vs_weaker = calculate_elo_change(1500, 1400, 1.0)
        self.assertGreater(change_vs_stronger, change_vs_weaker)


class TestSelfPlay(unittest.TestCase):
    """Test self-play functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping self-play tests")

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_self_play_manager(self):
        """Test SelfPlayManager basic creation."""
        # Create a dummy neural network
        nn = None  # Use random policy for testing
        
        # Create manager
        manager = az.SelfPlayManager(nn, 1, 10, 1)  # 1 game, 10 simulations, 1 thread
        self.assertIsNotNone(manager)

    def test_game_generation(self):
        """Test generating self-play games."""
        # Create a dummy neural network
        nn = None  # Use random policy for testing
        
        # Create manager
        manager = az.SelfPlayManager(nn, 1, 10, 1)
        
        # Enable game saving
        manager.setSaveGames(True, self.temp_dir)
        
        # Generate games
        games = manager.generateGames(az.GameType.GOMOKU, 9, False)  # Small board for speed
        
        # Check that we got a game
        self.assertEqual(len(games), 1)
        
        # Check that game file was created
        files = os.listdir(self.temp_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".json"))

    def test_exploration_params(self):
        """Test setting exploration parameters."""
        # Create a dummy neural network
        nn = None  # Use random policy for testing
        
        # Create manager
        manager = az.SelfPlayManager(nn, 1, 10, 1)
        
        # Set exploration params
        manager.setExplorationParams(
            dirichletAlpha=0.03,
            dirichletEpsilon=0.25,
            initialTemperature=1.0,
            temperatureDropMove=30,
            finalTemperature=0.0
        )
        
        # Generate a game
        games = manager.generateGames(az.GameType.GOMOKU, 9, False)
        
        # Check that game was generated
        self.assertEqual(len(games), 1)


if __name__ == "__main__":
    unittest.main()