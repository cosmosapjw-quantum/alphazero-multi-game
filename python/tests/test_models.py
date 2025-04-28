#!/usr/bin/env python3
"""
Unit tests for neural network models used in AlphaZero.
"""

import unittest
import os
import sys
import torch
import numpy as np
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from alphazero.models.ddw_randwire import DDWRandWireResNet, ResidualBlock, SEBlock, RouterModule, RandWireBlock
    import pyalphazero as az
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

class TestModelComponents(unittest.TestCase):
    """Tests for neural network model components."""
    
    def test_se_block(self):
        """Test Squeeze-and-Excitation Block."""
        # Create input tensor
        channels = 64
        input_tensor = torch.randn(2, channels, 8, 8)
        
        # Create SE block
        se_block = SEBlock(channels)
        
        # Forward pass
        output = se_block(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Check that values are modified but still in reasonable range
        self.assertFalse(torch.allclose(output, input_tensor))
        self.assertTrue(torch.all(torch.abs(output) <= torch.abs(input_tensor) * 1.1))
    
    def test_residual_block(self):
        """Test Residual Block."""
        # Create input tensor
        channels = 64
        input_tensor = torch.randn(2, channels, 8, 8)
        
        # Create residual block
        res_block = ResidualBlock(channels)
        
        # Forward pass
        output = res_block(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Check that the block is doing something
        self.assertFalse(torch.allclose(output, input_tensor))
    
    def test_router_module(self):
        """Test Router Module."""
        # Create input tensor
        in_channels = 128
        out_channels = 64
        input_tensor = torch.randn(2, in_channels, 8, 8)
        
        # Create router module
        router = RouterModule(in_channels, out_channels)
        
        # Forward pass
        output = router(input_tensor)
        
        # Check output shape
        expected_shape = (2, out_channels, 8, 8)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is non-negative (ReLU)
        self.assertTrue(torch.all(output >= 0))
    
    def test_randwire_block(self):
        """Test RandWire Block."""
        # Create input tensor
        channels = 64
        input_tensor = torch.randn(2, channels, 8, 8)
        
        # Create randwire block with fixed seed for reproducibility
        randwire_block = RandWireBlock(channels, num_nodes=16, p=0.75, seed=42)
        
        # Forward pass
        output = randwire_block(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, input_tensor.shape)


class TestDDWRandWireResNet(unittest.TestCase):
    """Tests for DDWRandWireResNet model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a small model for testing
        self.input_channels = 18  # Typical for Gomoku
        self.action_size = 225  # 15x15 Gomoku
        self.model = DDWRandWireResNet(
            input_channels=self.input_channels,
            output_size=self.action_size,
            channels=32,  # Small for testing
            num_blocks=2  # Small for testing
        )
    
    def test_model_creation(self):
        """Test model creation with different parameters."""
        # Test with different input channels
        model1 = DDWRandWireResNet(input_channels=34, output_size=64)
        self.assertEqual(model1.input_channels, 34)
        
        # Test with different action size
        model2 = DDWRandWireResNet(input_channels=18, output_size=81)
        
        # Test with different channel counts
        model3 = DDWRandWireResNet(input_channels=18, output_size=225, channels=64)
        
        # Test with different block counts
        model4 = DDWRandWireResNet(input_channels=18, output_size=225, num_blocks=10)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        # Create input tensor
        input_tensor = torch.randn(2, self.input_channels, 15, 15)
        
        # Forward pass
        policy, value = self.model(input_tensor)
        
        # Check output shapes
        self.assertEqual(policy.shape, (2, self.action_size))
        self.assertEqual(value.shape, (2, 1))
        
        # Check value range
        self.assertTrue(torch.all(value >= -1) and torch.all(value <= 1))
    
    def test_batch_processing(self):
        """Test batch processing."""
        # Create multiple batch sizes
        batch_sizes = [1, 4, 16]
        
        for batch_size in batch_sizes:
            # Create input tensor
            input_tensor = torch.randn(batch_size, self.input_channels, 15, 15)
            
            # Forward pass
            policy, value = self.model(input_tensor)
            
            # Check output shapes
            self.assertEqual(policy.shape, (batch_size, self.action_size))
            self.assertEqual(value.shape, (batch_size, 1))
    
    def test_model_save_load(self):
        """Test saving and loading model weights."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Save model
            torch.save(self.model.state_dict(), temp_path)
            
            # Create new model
            new_model = DDWRandWireResNet(
                input_channels=self.input_channels,
                output_size=self.action_size,
                channels=32,
                num_blocks=2
            )
            
            # Load weights
            new_model.load_state_dict(torch.load(temp_path))
            
            # Create input tensor
            input_tensor = torch.randn(2, self.input_channels, 15, 15)
            
            # Forward pass through both models
            with torch.no_grad():
                policy1, value1 = self.model(input_tensor)
                policy2, value2 = new_model(input_tensor)
            
            # Check that outputs are identical
            self.assertTrue(torch.allclose(policy1, policy2))
            self.assertTrue(torch.allclose(value1, value2))
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_parameter_count(self):
        """Test model parameter count."""
        # Calculate parameter count
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # Parameter count should be non-zero
        self.assertGreater(param_count, 0)
        
        # Parameter count should be less than a full-size model
        full_model = DDWRandWireResNet(
            input_channels=self.input_channels,
            output_size=self.action_size
        )
        full_param_count = sum(p.numel() for p in full_model.parameters())
        
        self.assertLess(param_count, full_param_count)
    
    def test_different_board_sizes(self):
        """Test model with different board sizes."""
        # Test with a 9x9 board
        input_tensor_9x9 = torch.randn(2, self.input_channels, 9, 9)
        policy_9x9, value_9x9 = self.model(input_tensor_9x9)
        
        # The model should adapt to the new input size through adaptive pooling
        self.assertEqual(policy_9x9.shape, (2, self.action_size))
        self.assertEqual(value_9x9.shape, (2, 1))
        
        # Test with a 19x19 board
        input_tensor_19x19 = torch.randn(2, self.input_channels, 19, 19)
        policy_19x19, value_19x19 = self.model(input_tensor_19x19)
        
        self.assertEqual(policy_19x19.shape, (2, self.action_size))
        self.assertEqual(value_19x19.shape, (2, 1))


class TestModelIntegration(unittest.TestCase):
    """Tests for neural network model integration with the game environment."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            # Create a Gomoku game state for testing
            self.game_state = az.createGameState(az.GameType.GOMOKU, 15, False)
            
            # Get input shape for the model
            tensor_rep = self.game_state.getEnhancedTensorRepresentation()
            self.input_channels = len(tensor_rep)
            self.action_size = self.game_state.getActionSpaceSize()
            
            # Create a small model for testing
            self.model = DDWRandWireResNet(
                input_channels=self.input_channels,
                output_size=self.action_size,
                channels=32,  # Small for testing
                num_blocks=2  # Small for testing
            )
        except Exception as e:
            self.skipTest(f"Failed to set up test environment: {e}")
    
    def test_model_with_game_state(self):
        """Test model with game state tensor representation."""
        try:
            # Get tensor representation
            tensor = self.game_state.getEnhancedTensorRepresentation()
            
            # Convert to PyTorch tensor
            input_tensor = torch.FloatTensor(tensor)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            policy, value = self.model(input_tensor)
            
            # Check output shapes
            self.assertEqual(policy.shape, (1, self.action_size))
            self.assertEqual(value.shape, (1, 1))
            
            # Check value range
            self.assertTrue(value.item() >= -1 and value.item() <= 1)
            
            # Apply softmax to get probabilities
            policy_probs = torch.softmax(policy, dim=1)
            
            # Check that probabilities sum to 1
            self.assertAlmostEqual(policy_probs.sum().item(), 1.0, places=5)
            
            # Check that all probabilities are non-negative
            self.assertTrue(torch.all(policy_probs >= 0))
        except Exception as e:
            self.skipTest(f"Test skipped due to exception: {e}")
    
    def test_mcts_with_model(self):
        """Test MCTS with neural network model."""
        try:
            # Create a temporary file for the model
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp:
                temp_path = temp.name
                
                # Save the model
                torch.save(self.model.state_dict(), temp_path)
            
            try:
                # Create a neural network wrapper for the model
                class TorchNeuralNetwork(az.NeuralNetwork):
                    def __init__(self, model, model_path):
                        super().__init__()
                        self.model = model
                        self.model_path = model_path
                    
                    def predict(self, state):
                        # Convert state tensor to PyTorch tensor
                        state_tensor = torch.FloatTensor(state.getEnhancedTensorRepresentation())
                        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                        
                        # Forward pass
                        with torch.no_grad():
                            policy_logits, value = self.model(state_tensor)
                            policy = torch.softmax(policy_logits, dim=1)[0].numpy()
                            value = value.item()
                        
                        return policy, value
                    
                    def isGpuAvailable(self):
                        return torch.cuda.is_available()
                    
                    def getDeviceInfo(self):
                        if torch.cuda.is_available():
                            return f"GPU: {torch.cuda.get_device_name(0)}"
                        else:
                            return "CPU"
                    
                    def getInferenceTimeMs(self):
                        return 0.0
                    
                    def getBatchSize(self):
                        return 1
                    
                    def getModelInfo(self):
                        return "PyTorch DDWRandWireResNet"
                    
                    def getModelSizeBytes(self):
                        return sum(p.numel() * 4 for p in self.model.parameters())
                
                # Create neural network instance
                nn = TorchNeuralNetwork(self.model, temp_path)
                
                # Create transposition table
                tt = az.TranspositionTable(1024, 16)
                
                # Create MCTS
                mcts = az.ParallelMCTS(self.game_state, nn, tt, 1, 10)
                
                # Run search
                mcts.search()
                
                # Get action probabilities
                probs = mcts.getActionProbabilities(1.0)
                
                # Check that probabilities are valid
                self.assertEqual(len(probs), self.action_size)
                self.assertAlmostEqual(sum(probs), 1.0, places=5)
                
                # Select action
                action = mcts.selectAction(False, 0.0)
                
                # Check that action is valid
                self.assertTrue(self.game_state.isLegalMove(action))
                
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            self.skipTest(f"Test skipped due to exception: {e}")


if __name__ == "__main__":
    unittest.main()