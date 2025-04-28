#!/usr/bin/env python3
"""
Model export utility for AlphaZero models.

This script exports trained PyTorch models to optimized formats (TorchScript, ONNX)
for faster inference in production environments.

Usage:
    python export_model.py [options]

Options:
    --model MODEL           Path to model file (.pt or .pth)
    --game {gomoku,chess,go}  Game type
    --size SIZE             Board size (default: depends on game)
    --format {torchscript,onnx,both}  Export format (default: both)
    --output-dir DIR        Output directory (default: exported_models)
    --quantize              Apply quantization to reduce model size
    --test                  Test exported model against original
    --variant               Use variant rules
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import pyalphazero as az
from alphazero.models import DDWRandWireResNet

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Model Export")
    parser.add_argument("--model", type=str, required=True,
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
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    return parser.parse_args()


def load_model(model_path, input_channels, action_size, device):
    """Load a PyTorch model."""
    model = DDWRandWireResNet(input_channels, action_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def export_to_torchscript(model, model_path, output_dir, quantize=False):
    """Export model to TorchScript format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model filename
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
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
        
        # Get model filename
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
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


def test_exported_model(original_model, exported_model_path, format_type, game_state):
    """Test exported model against the original model."""
    print(f"\nTesting exported {format_type} model...")
    
    # Get a tensor from the game state
    tensor = torch.FloatTensor(game_state.getEnhancedTensorRepresentation())
    tensor = tensor.unsqueeze(0).to(next(original_model.parameters()).device)
    
    # Get predictions from original model
    with torch.no_grad():
        original_policy, original_value = original_model(tensor)
        original_policy = torch.softmax(original_policy, dim=1)
    
    # Get predictions from exported model
    try:
        if format_type == "torchscript":
            # Load TorchScript model
            exported_model = torch.jit.load(exported_model_path)
            exported_model.eval()
            
            # Get predictions
            with torch.no_grad():
                exported_policy, exported_value = exported_model(tensor)
                exported_policy = torch.softmax(exported_policy, dim=1)
            
        elif format_type == "onnx":
            # Load ONNX model
            import onnxruntime as ort
            
            ort_session = ort.InferenceSession(exported_model_path)
            input_name = ort_session.get_inputs()[0].name
            
            # Run inference
            ort_inputs = {input_name: tensor.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Convert to PyTorch tensors
            exported_policy = torch.softmax(torch.tensor(ort_outputs[0]), dim=1)
            exported_value = torch.tensor(ort_outputs[1])
            
        # Calculate differences
        policy_diff = torch.abs(original_policy - exported_policy).mean().item()
        value_diff = torch.abs(original_value - exported_value).mean().item()
        
        print(f"Policy difference: {policy_diff:.6f}")
        print(f"Value difference: {value_diff:.6f}")
        
        # Run performance comparison
        print("\nPerformance comparison:")
        
        # Warm-up
        for _ in range(5):
            with torch.no_grad():
                original_model(tensor)
        
        # Test original model
        n_iters = 100
        start_time = time.time()
        for _ in range(n_iters):
            with torch.no_grad():
                original_model(tensor)
        original_time = (time.time() - start_time) / n_iters
        
        # Test exported model
        start_time = time.time()
        for _ in range(n_iters):
            if format_type == "torchscript":
                with torch.no_grad():
                    exported_model(tensor)
            elif format_type == "onnx":
                ort_session.run(None, ort_inputs)
        exported_time = (time.time() - start_time) / n_iters
        
        print(f"Original model: {original_time*1000:.2f} ms per inference")
        print(f"Exported model: {exported_time*1000:.2f} ms per inference")
        print(f"Speedup: {original_time/exported_time:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"Error testing exported model: {e}")
        return False


def main():
    args = parse_args()
    
    # Convert game type to enum
    game_type_map = {
        "gomoku": az.GameType.GOMOKU,
        "chess": az.GameType.CHESS,
        "go": az.GameType.GO
    }
    game_type = game_type_map[args.game]
    
    # Default board sizes
    if args.size <= 0:
        if args.game == "gomoku":
            board_size = 15
        elif args.game == "chess":
            board_size = 8
        elif args.game == "go":
            board_size = 19
        else:
            board_size = 15
    else:
        board_size = args.size
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create game state to get input shape
    game_state = az.createGameState(game_type, board_size, args.variant)
    tensor_rep = game_state.getEnhancedTensorRepresentation()
    input_channels = len(tensor_rep)
    action_size = game_state.getActionSpaceSize()
    
    print(f"Game type: {args.game}")
    print(f"Board size: {board_size}")
    print(f"Input channels: {input_channels}")
    print(f"Action space size: {action_size}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    try:
        model = load_model(args.model, input_channels, action_size, device)
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Export model based on requested format
    torchscript_path = None
    torchscript_quant_path = None
    onnx_path = None
    onnx_quant_path = None
    
    if args.format in ["torchscript", "both"]:
        torchscript_path, torchscript_quant_path = export_to_torchscript(
            model, args.model, args.output_dir, args.quantize
        )
    
    if args.format in ["onnx", "both"]:
        onnx_path, onnx_quant_path = export_to_onnx(
            model, args.model, args.output_dir, args.quantize
        )
    
    # Test exported models if requested
    if args.test:
        if torchscript_path:
            test_exported_model(model, torchscript_path, "torchscript", game_state)
        
        if torchscript_quant_path:
            test_exported_model(model, torchscript_quant_path, "torchscript", game_state)
        
        if onnx_path:
            test_exported_model(model, onnx_path, "onnx", game_state)
        
        if onnx_quant_path:
            test_exported_model(model, onnx_quant_path, "onnx", game_state)
    
    print("\nModel export completed successfully!")


if __name__ == "__main__":
    main()