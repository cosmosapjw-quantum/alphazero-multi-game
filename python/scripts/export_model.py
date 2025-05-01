#!/usr/bin/env python3
"""
Model export utility for AlphaZero models.

This script exports trained PyTorch models to optimized formats (TorchScript, ONNX)
for faster inference in production environments. It supports both Python models and 
the new C++ DDWRandWireResNet implementation for maximum performance.

Usage:
    python export_model.py [options]

Options:
    --model MODEL           Path to model file (.pt or .pth)
    --game {gomoku,chess,go}  Game type
    --size SIZE             Board size (default: depends on game)
    --format {torchscript,onnx,both}  Export format (default: torchscript)
    --output-dir DIR        Output directory (default: exported_models)
    --quantize              Apply quantization to reduce model size
    --test                  Test exported model against original
    --variant               Use variant rules
    --create-random         Create a random model if no model provided
    --use-cpp               Try to use C++ implementation for better performance
    --channels CHANNELS     Number of model channels (default: 64)
    --blocks BLOCKS         Number of random wire blocks (default: 8)
    --device DEVICE         Device to use (default: auto-detect)
    --timeout TIMEOUT       Timeout in seconds for model creation (default: 60)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the build directory for the C++ extension
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'pybind'))
sys.path.insert(0, build_dir)
# Add the root directory where the .so might also be copied
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)

import torch  # Import torch first
import _alphazero_cpp as az

# Try to import the models
try:
    # First try to import both Python and C++ implementations
    from alphazero.models import DDWRandWireResNet, DDWRandWireResNetWrapper, create_model
    CPP_MODEL_AVAILABLE = True
except ImportError:
    try:
        # If C++ implementation not available, try just the Python one
        from alphazero.models import DDWRandWireResNet, create_model
        CPP_MODEL_AVAILABLE = False
        print("C++ DDWRandWireResNet wrapper not available. Using Python implementation only.")
    except ImportError:
        # If even the Python implementation fails, we'll need to handle it in main()
        print("Warning: Failed to import model classes. Will attempt to create models directly when needed.")
        # Ensure CPP_MODEL_AVAILABLE is defined
        CPP_MODEL_AVAILABLE = False


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
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--create-random", action="store_true",
                        help="Create a random model if no model provided")
    parser.add_argument("--use-cpp", action="store_true",
                        help="Try to use C++ implementation for better performance")
    parser.add_argument("--channels", type=int, default=64,
                        help="Number of model channels (default: 64)")
    parser.add_argument("--blocks", type=int, default=8,
                        help="Number of random wire blocks (default: 8)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for model creation (default: 60)")
    return parser.parse_args()


def create_random_model(input_channels, action_size, channels=64, blocks=8, use_cpp=False, device="cpu", timeout_secs=60):
    """Create a random model with specified parameters."""
    print(f"Creating random model with {channels} channels and {blocks} blocks")
    
    # Add a timeout mechanism to prevent hangs
    import threading
    import signal
    
    # Flag to check if model creation is done
    creation_done = threading.Event()
    
    # Setup timeout handler
    def timeout_handler():
        if not creation_done.is_set():
            print(f"\nWARNING: Model creation taking longer than {timeout_secs} seconds!")
            print("You can:")
            print("  1. Keep waiting (model creation will continue)")
            print("  2. Ctrl+C to cancel and try again with smaller model (--channels and --blocks)")
            print("  3. Try with CPU instead of GPU (--device cpu)")
    
    # Start timeout timer
    timer = threading.Timer(timeout_secs, timeout_handler)
    timer.daemon = True
    timer.start()
    
    try:
        if use_cpp:
            try:
                print("Attempting to use C++ implementation...")
                if CPP_MODEL_AVAILABLE:
                    print("Using DDWRandWireResNetWrapper")
                    model = DDWRandWireResNetWrapper(input_channels, action_size, channels, blocks)
                else:
                    # Use C++ implementation directly
                    print("Using az.DDWRandWireResNetCpp directly")
                    model = az.DDWRandWireResNetCpp(input_channels, action_size, channels, blocks)
                print("C++ model initialized successfully")
                print("Using C++ DDWRandWireResNet implementation")
            except Exception as e:
                print(f"Error creating C++ model: {e}")
                # Fall back to Python implementation
                model = DDWRandWireResNet(input_channels, action_size, channels, blocks)
                print("Using Python DDWRandWireResNet implementation (fallback)")
        else:
            # Try Python implementation
            try:
                model = DDWRandWireResNet(input_channels, action_size, channels, blocks)
                print("Using Python DDWRandWireResNet implementation")
            except NameError:
                # Python implementation not available, use C++
                print("Python implementation not found, attempting C++ fallback...")
                model = az.DDWRandWireResNetCpp(input_channels, action_size, channels, blocks)
                print("Using C++ DDWRandWireResNet implementation (fallback)")
        
        # Cancel timer since model creation succeeded
        creation_done.set()
        timer.cancel()
        
        print(f"Moving model to device: {device}")
        model.to(device)
        print("Setting model to eval mode")
        model.eval()
        return model
    except Exception as e:
        creation_done.set()
        timer.cancel()
        if 'CUDA out of memory' in str(e):
            print(f"\nERROR: GPU out of memory. Try reducing model size (--channels and --blocks) or use CPU.")
        raise e


def load_model(model_path, input_channels, action_size, device, channels=64, blocks=8, use_cpp=False, create_random=False, timeout_secs=60):
    """Load a PyTorch model or create a random one if requested."""
    if model_path is None:
        if not create_random:
            raise ValueError("No model path provided and --create-random not specified")
        return create_random_model(input_channels, action_size, channels, blocks, use_cpp, device, timeout_secs)
    
    print(f"Loading model from {model_path}")
    
    # Add a timeout mechanism to prevent hangs
    import threading
    
    # Flag to check if model creation is done
    creation_done = threading.Event()
    
    # Setup timeout handler
    def timeout_handler():
        if not creation_done.is_set():
            print(f"\nWARNING: Model loading taking longer than {timeout_secs} seconds!")
            print("You can:")
            print("  1. Keep waiting (model loading will continue)")
            print("  2. Ctrl+C to cancel and try again with smaller model (--channels and --blocks)")
            print("  3. Try with CPU instead of GPU (--device cpu)")
    
    # Start timeout timer
    timer = threading.Timer(timeout_secs, timeout_handler)
    timer.daemon = True
    timer.start()
    
    try:
        if use_cpp:
            try:
                if CPP_MODEL_AVAILABLE:
                    model = DDWRandWireResNetWrapper(input_channels, action_size)
                else:
                    # Use C++ implementation directly
                    model = az.DDWRandWireResNetCpp(input_channels, action_size)
                print("Using C++ DDWRandWireResNet implementation")
            except Exception as e:
                print(f"Error creating C++ model: {e}")
                # Fall back to Python implementation
                model = DDWRandWireResNet(input_channels, action_size)
                print("Using Python DDWRandWireResNet implementation (fallback)")
        else:
            # Try Python implementation
            try:
                model = DDWRandWireResNet(input_channels, action_size)
                print("Using Python DDWRandWireResNet implementation")
            except NameError:
                # Python implementation not available, use C++
                model = az.DDWRandWireResNetCpp(input_channels, action_size)
                print("Using C++ DDWRandWireResNet implementation (fallback)")
        
        print(f"Loading state dict from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Cancel timer since model loading succeeded
        creation_done.set()
        timer.cancel()
        
        print(f"Moving model to device: {device}")
        model.to(device)
        print("Setting model to eval mode")
        model.eval()
        return model
    except Exception as e:
        creation_done.set()
        timer.cancel()
        if 'CUDA out of memory' in str(e):
            print(f"\nERROR: GPU out of memory. Try reducing model size or use CPU.")
        raise e


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
    device = args.device
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Adjust model size based on game and device
    if args.channels == 64 and args.blocks == 8:
        # Only apply automatic adjustment for default values
        is_large_game = args.game == "go" and board_size >= 19
        is_gpu = str(device).startswith("cuda")
        
        if is_large_game and not is_gpu:
            print("Large game detected on CPU. Reducing model size for faster creation.")
            args.channels = 32
            args.blocks = 4
        elif is_large_game and is_gpu:
            print("Large game detected on GPU. Using moderate model size.")
            args.channels = 48
            args.blocks = 6
        elif not is_gpu:
            print("Using CPU. Reducing model size for faster creation.")
            args.channels = 48
            args.blocks = 6
    
    # Generate a model name if creating a random model and no model path provided
    model_path = args.model
    if model_path is None and args.create_random:
        # Create a default filename for the random model
        model_path = f"random_model_{args.game}_{board_size}x{board_size}.pt"
        print(f"Will create random model and save to {model_path}")
    
    # Load the model
    try:
        start_time = time.time()
        model = load_model(
            model_path, 
            input_channels, 
            action_size, 
            device,
            channels=args.channels,
            blocks=args.blocks,
            use_cpp=args.use_cpp,
            create_random=args.create_random,
            timeout_secs=args.timeout
        )
        
        if args.model:
            print(f"Loaded model from {args.model}")
        else:
            # Save the randomly created model
            if args.create_random:
                os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
                print(f"Saving random model to {model_path}")
                torch.save(model.state_dict(), model_path)
                print(f"Saved random model to {model_path}")
        
        load_time = time.time() - start_time
        print(f"Model loading/creation took {load_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading/creating model: {e}")
        return
    
    # Export model based on requested format
    torchscript_path = None
    torchscript_quant_path = None
    onnx_path = None
    onnx_quant_path = None
    
    if args.format in ["torchscript", "both"]:
        print("\nExporting to TorchScript format...")
        start_time = time.time()
        torchscript_path, torchscript_quant_path = export_to_torchscript(
            model, model_path, args.output_dir, args.quantize
        )
        export_time = time.time() - start_time
        print(f"TorchScript export took {export_time:.2f} seconds")
    
    if args.format in ["onnx", "both"]:
        print("\nExporting to ONNX format...")
        start_time = time.time()
        onnx_path, onnx_quant_path = export_to_onnx(
            model, model_path, args.output_dir, args.quantize
        )
        export_time = time.time() - start_time
        print(f"ONNX export took {export_time:.2f} seconds")
    
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