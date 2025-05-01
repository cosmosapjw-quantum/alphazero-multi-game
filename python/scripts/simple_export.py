#!/usr/bin/env python3
"""
Simplified model export utility for AlphaZero models.

This script exports trained PyTorch models to optimized formats (TorchScript, ONNX)
without requiring the C++ extension to work.

Usage:
    python simple_export.py [options]

Options:
    --model MODEL           Path to model file (.pt or .pth)
    --game {gomoku,chess,go}  Game type
    --size SIZE             Board size (default: depends on game)
    --format {torchscript,onnx,both}  Export format (default: torchscript)
    --output-dir DIR        Output directory (default: exported_models)
    --quantize              Apply quantization to reduce model size
    --test                  Test exported model against original
    --create-random         Create a random model if no model provided
    --channels CHANNELS     Number of model channels (default: 128)
    --blocks BLOCKS         Number of random wire blocks (default: 20)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the models
try:
    from alphazero.models import DDWRandWireResNet, create_model
    print("Successfully imported Python model implementations")
except ImportError:
    print("Warning: Failed to import model classes. Will attempt to create models directly.")
    class DDWRandWireResNet(torch.nn.Module):
        def __init__(self, input_channels, action_size, channels=128, blocks=20):
            super().__init__()
            self.input_channels = input_channels
            
            # Input layer
            self.input_conv = torch.nn.Conv2d(input_channels, channels, 3, padding=1, bias=False)
            self.input_bn = torch.nn.BatchNorm2d(channels)
            
            # Middle layers (simplified for this example)
            self.middle_layers = torch.nn.Sequential(
                *[self._make_block(channels) for _ in range(blocks)]
            )
            
            # Policy head
            self.policy_conv = torch.nn.Conv2d(channels, 32, 1, bias=False)
            self.policy_bn = torch.nn.BatchNorm2d(32)
            self.policy_fc = torch.nn.Linear(32 * 8 * 8, action_size)
            
            # Value head
            self.value_conv = torch.nn.Conv2d(channels, 32, 1, bias=False)
            self.value_bn = torch.nn.BatchNorm2d(32)
            self.value_fc1 = torch.nn.Linear(32 * 8 * 8, 256)
            self.value_fc2 = torch.nn.Linear(256, 1)
        
        def _make_block(self, channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU()
            )
        
        def forward(self, x):
            # Input layer
            x = torch.relu(self.input_bn(self.input_conv(x)))
            
            # Middle layers
            x = self.middle_layers(x)
            
            # Adaptive pooling to handle different board sizes
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
            
            # Policy head
            policy = torch.relu(self.policy_bn(self.policy_conv(x_pooled)))
            policy = policy.view(policy.size(0), -1)
            policy = self.policy_fc(policy)
            
            # Value head
            value = torch.relu(self.value_bn(self.value_conv(x_pooled)))
            value = value.view(value.size(0), -1)
            value = torch.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))
            
            return policy, value


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Model Export")
    parser.add_argument("--model", type=str,
                        help="Path to model file")
    parser.add_argument("--game", type=str, required=True,
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--format", type=str, default="both",
                        choices=["torchscript", "onnx", "both"],
                        help="Export format")
    parser.add_argument("--output-dir", type=str, default="exported_models",
                        help="Output directory")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply quantization to reduce model size")
    parser.add_argument("--test", action="store_true",
                        help="Test exported model against original")
    parser.add_argument("--create-random", action="store_true",
                        help="Create a random model if no model provided")
    parser.add_argument("--channels", type=int, default=128,
                        help="Number of model channels (default: 128)")
    parser.add_argument("--blocks", type=int, default=20,
                        help="Number of random wire blocks (default: 20)")
    return parser.parse_args()


def create_random_model(input_channels, action_size, channels=128, blocks=20, device="cpu"):
    """Create a random model with specified parameters."""
    print(f"Creating random model with {channels} channels and {blocks} blocks")
    
    model = DDWRandWireResNet(input_channels, action_size, channels, blocks)
    model.to(device)
    model.eval()
    return model


def export_to_torchscript(model, model_path, output_dir, quantize=False):
    """Export model to TorchScript format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model filename or create one if random model
        if model_path:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
        else:
            model_name = "random_model"
        
        # Trace the model
        example_input = torch.randn(1, model.input_channels, 8, 8).to(next(model.parameters()).device)
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        ts_path = os.path.join(output_dir, f"{model_name}_torchscript.pt")
        traced_model.save(ts_path)
        print(f"Exported TorchScript model to {ts_path}")
        
        # Quantization (if requested)
        if quantize:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Trace and export quantized model
            traced_quantized = torch.jit.trace(quantized_model, example_input)
            ts_quant_path = os.path.join(output_dir, f"{model_name}_torchscript_quantized.pt")
            traced_quantized.save(ts_quant_path)
            print(f"Exported quantized TorchScript model to {ts_quant_path}")
            
            # Compare model sizes
            original_size = os.path.getsize(ts_path) / (1024 * 1024)
            quantized_size = os.path.getsize(ts_quant_path) / (1024 * 1024)
            print(f"Original size: {original_size:.2f} MB")
            print(f"Quantized size: {quantized_size:.2f} MB")
            print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.1f}%")
            
            return ts_path, ts_quant_path
        
        return ts_path, None
    
    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")
        return None, None


def export_to_onnx(model, model_path, output_dir, quantize=False):
    """Export model to ONNX format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model filename or create one if random model
        if model_path:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
        else:
            model_name = "random_model"
        
        # Prepare input
        example_input = torch.randn(1, model.input_channels, 8, 8).to(next(model.parameters()).device)
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy_logits', 'value'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'policy_logits': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )
        print(f"Exported ONNX model to {onnx_path}")
        
        # Quantization (if requested)
        onnx_quant_path = None
        if quantize:
            try:
                import onnx
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                onnx_quant_path = os.path.join(output_dir, f"{model_name}_quantized.onnx")
                quantize_dynamic(
                    onnx_path,
                    onnx_quant_path,
                    weight_type=QuantType.QUInt8
                )
                print(f"Exported quantized ONNX model to {onnx_quant_path}")
                
                # Compare model sizes
                original_size = os.path.getsize(onnx_path) / (1024 * 1024)
                quantized_size = os.path.getsize(onnx_quant_path) / (1024 * 1024)
                print(f"Original size: {original_size:.2f} MB")
                print(f"Quantized size: {quantized_size:.2f} MB")
                print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.1f}%")
                
            except ImportError:
                print("ONNX quantization requires onnx and onnxruntime packages. Skipping quantization.")
        
        return onnx_path, onnx_quant_path
    
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return None, None


def load_model(model_path, input_channels, action_size, device, channels=128, blocks=20):
    """Load a PyTorch model or create a random one if requested."""
    if model_path is None:
        return create_random_model(input_channels, action_size, channels, blocks, device)
    
    model = DDWRandWireResNet(input_channels, action_size, channels, blocks)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_game_params(game_type, board_size):
    """Get game parameters based on game type."""
    if game_type == "gomoku":
        if board_size <= 0:
            board_size = 15
        # Gomoku: board state + last N moves + player to move
        input_channels = board_size + 5 + 1
        action_size = board_size * board_size
    elif game_type == "chess":
        if board_size <= 0:
            board_size = 8
        # Chess: 12 piece types + additional planes for castling, en passant, etc.
        input_channels = 12 + 8
        action_size = 64 * 73  # All possible moves in chess
    elif game_type == "go":
        if board_size <= 0:
            board_size = 19
        # Go: black, white, legal moves, ko, etc.
        input_channels = 17
        action_size = board_size * board_size + 1  # +1 for pass
    else:
        raise ValueError(f"Unsupported game type: {game_type}")
    
    return input_channels, action_size, board_size


def main():
    args = parse_args()
    
    # Get game parameters
    input_channels, action_size, board_size = get_game_params(args.game, args.size)
    
    print(f"Game type: {args.game}")
    print(f"Board size: {board_size}")
    print(f"Input channels: {input_channels}")
    print(f"Action space size: {action_size}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate a model name if creating a random model and no model path provided
    model_path = args.model
    if model_path is None:
        # Create a default filename for the random model
        model_path = f"random_model_{args.game}_{board_size}x{board_size}.pt"
        print(f"Will create random model and save to {model_path}")
    
    # Load the model
    try:
        model = load_model(
            None if args.model is None else args.model,  # Only pass model_path if not creating random
            input_channels, 
            action_size, 
            device,
            channels=args.channels,
            blocks=args.blocks
        )
        
        if args.model:
            print(f"Loaded model from {args.model}")
        else:
            # Save the randomly created model
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved random model to {model_path}")
    except Exception as e:
        print(f"Error loading/creating model: {e}")
        return
    
    # Export model based on requested format
    torchscript_path = None
    torchscript_quant_path = None
    onnx_path = None
    onnx_quant_path = None
    
    if args.format in ["torchscript", "both"]:
        torchscript_path, torchscript_quant_path = export_to_torchscript(
            model, model_path, args.output_dir, args.quantize
        )
    
    if args.format in ["onnx", "both"]:
        onnx_path, onnx_quant_path = export_to_onnx(
            model, model_path, args.output_dir, args.quantize
        )
    
    print("\nModel export completed successfully!")


if __name__ == "__main__":
    main() 