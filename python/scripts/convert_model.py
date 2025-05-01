#!/usr/bin/env python3
"""
Utility script to convert models between Python and C++ implementations.

This script can:
1. Convert a Python DDWRandWireResNet model to a C++ compatible format
2. Export a C++ or Python model to TorchScript format for fast inference

Usage:
    python convert_model.py --input-model INPUT_PATH --output-model OUTPUT_PATH
                           [--mode {py2cpp,cpp2py,export}]
                           [--game {gomoku,chess,go}]
                           [--size SIZE]
                           [--channels CHANNELS]
                           [--blocks NUM_BLOCKS]
"""

import os
import sys
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from alphazero.models import DDWRandWireResNet, DDWRandWireResNetWrapper
    import _alphazero_cpp as az
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the project is properly built and the Python package is installed.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert between Python and C++ models")
    
    parser.add_argument("--input-model", type=str, required=True,
                        help="Path to input model file")
    parser.add_argument("--output-model", type=str, required=True,
                        help="Path to output model file")
    parser.add_argument("--mode", type=str, choices=["py2cpp", "cpp2py", "export"], 
                        default="export", help="Conversion mode")
    parser.add_argument("--game", type=str, choices=["gomoku", "chess", "go"], 
                        default="gomoku", help="Game type")
    parser.add_argument("--size", type=int, default=0, 
                        help="Board size (0 for default based on game)")
    parser.add_argument("--channels", type=int, default=128, 
                        help="Number of model channels")
    parser.add_argument("--blocks", type=int, default=20, 
                        help="Number of random wire blocks")
    
    return parser.parse_args()

def get_default_board_size(game_type):
    """Get default board size for a game."""
    if game_type == "chess":
        return 8
    elif game_type == "go":
        return 19
    else:  # gomoku
        return 15

def get_input_channels(game_type):
    """Get number of input channels for a game."""
    if game_type == "chess":
        return 14  # 6 piece types x 2 colors + auxiliary channels
    elif game_type == "go":
        return 8   # Current player stones, opponent stones, history, and auxiliary channels
    else:  # gomoku
        return 8   # Current player stones, opponent stones, history, and auxiliary channels

def get_output_size(game_type, board_size):
    """Get output size (policy dimension) for a game."""
    if game_type == "chess":
        return 64 * 73  # 64 squares, 73 possible moves per square (max)
    elif game_type == "go":
        return board_size * board_size + 1  # +1 for pass move
    else:  # gomoku
        return board_size * board_size

def convert_python_to_cpp(input_path, output_path, input_channels, output_size, channels, num_blocks):
    """Convert a Python model to a C++ compatible format."""
    print(f"Converting Python model {input_path} to C++ format {output_path}")
    
    # Load the Python model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    python_model = DDWRandWireResNet(input_channels, output_size, channels, num_blocks)
    python_model.load_state_dict(torch.load(input_path, map_location=device))
    python_model.to(device).eval()
    
    # Create input tensor for tracing
    dummy_input = torch.zeros((1, input_channels, output_size, output_size), device=device)
    
    # Trace the model
    traced_model = torch.jit.trace(python_model, dummy_input)
    
    # Save the traced model
    traced_model.save(output_path)
    print(f"C++ compatible model saved to {output_path}")
    
    return traced_model

def convert_cpp_to_python(input_path, output_path, input_channels, output_size, channels, num_blocks):
    """Convert a C++ model to a Python compatible format."""
    print(f"Converting C++ model {input_path} to Python format {output_path}")
    
    # Create a C++ wrapper model
    cpp_model = DDWRandWireResNetWrapper(input_channels, output_size, channels, num_blocks)
    cpp_model.load(input_path)
    
    # Create a Python model
    python_model = DDWRandWireResNet(input_channels, output_size, channels, num_blocks)
    
    # Transfer parameters (this is complex and depends on model structure)
    # This is a simplification - in practice, parameter mapping would need to be done manually
    print("WARNING: Direct C++ to Python conversion is not fully implemented")
    print("         The output model will have the C++ model architecture but random weights")
    
    # Save the Python model
    torch.save(python_model.state_dict(), output_path)
    print(f"Python model saved to {output_path}")
    
    return python_model

def export_to_torchscript(input_path, output_path, input_channels, output_size, channels, num_blocks, board_size):
    """Export a model to TorchScript format."""
    print(f"Exporting model {input_path} to TorchScript format {output_path}")
    
    # Check if input is a TorchScript model already
    try:
        # Try to load as TorchScript model
        model = torch.jit.load(input_path)
        print(f"Input is already a TorchScript model, saving to {output_path}")
        model.save(output_path)
        return model
    except Exception:
        # Not a TorchScript model, try other formats
        pass
    
    # Try to load as C++ model via wrapper
    try:
        model = DDWRandWireResNetWrapper(input_channels, output_size, channels, num_blocks)
        model.load(input_path)
        print("Successfully loaded as C++ model")
        
        # Export to TorchScript
        input_shape = [1, input_channels, board_size, board_size]
        model.export_to_torchscript(output_path, input_shape)
        print(f"Model exported to {output_path}")
        return model
    except Exception as e:
        print(f"Failed to load as C++ model: {e}")
    
    # Try to load as Python model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DDWRandWireResNet(input_channels, output_size, channels, num_blocks)
        model.load_state_dict(torch.load(input_path, map_location=device))
        model.to(device).eval()
        
        print("Successfully loaded as Python model")
        
        # Create input tensor for tracing
        dummy_input = torch.zeros((1, input_channels, board_size, board_size), device=device)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save the traced model
        traced_model.save(output_path)
        print(f"Model exported to {output_path}")
        return traced_model
    except Exception as e:
        print(f"Failed to load as Python model: {e}")
    
    print("ERROR: Could not load the input model in any supported format")
    return None

def main():
    args = parse_args()
    
    # Get default board size if not specified
    board_size = args.size if args.size > 0 else get_default_board_size(args.game)
    
    # Get input channels and output size
    input_channels = get_input_channels(args.game)
    output_size = get_output_size(args.game, board_size)
    
    print(f"Using model parameters:")
    print(f"  Game type: {args.game}")
    print(f"  Board size: {board_size}x{board_size}")
    print(f"  Input channels: {input_channels}")
    print(f"  Output size: {output_size}")
    print(f"  Channels: {args.channels}")
    print(f"  Blocks: {args.blocks}")
    
    # Perform the requested conversion
    if args.mode == "py2cpp":
        convert_python_to_cpp(args.input_model, args.output_model, 
                             input_channels, output_size, args.channels, args.blocks)
    elif args.mode == "cpp2py":
        convert_cpp_to_python(args.input_model, args.output_model,
                             input_channels, output_size, args.channels, args.blocks)
    elif args.mode == "export":
        export_to_torchscript(args.input_model, args.output_model,
                             input_channels, output_size, args.channels, args.blocks, board_size)
    
    print("Conversion completed!")

if __name__ == "__main__":
    main()