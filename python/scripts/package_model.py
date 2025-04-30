#!/usr/bin/env python3
"""
Model distribution package utility for AlphaZero models.

This script creates distribution packages for trained AlphaZero models,
including metadata, versioning, and compatibility checks.

Usage:
    python package_model.py [options]

Options:
    --model MODEL           Path to model file
    --game {gomoku,chess,go}  Game type
    --size SIZE             Board size (default: depends on game)
    --version VERSION       Model version (default: 1.0.0)
    --output-dir DIR        Output directory (default: model_packages)
    --name NAME             Model name (default: derived from filename)
    --description DESC      Model description
    --export-formats LIST   Export formats (comma-separated list of: original,torchscript,onnx)
    --include-metadata      Include training metadata if available
    --quantize              Apply quantization to reduce model size
    --variant               Use variant rules for testing
"""

import os
import sys
import json
import time
import shutil
import argparse
from datetime import datetime
import torch
import _alphazero_cpp as az

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.models import DDWRandWireResNet


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Model Distribution Package")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model file")
    parser.add_argument("--game", type=str, required=True,
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--version", type=str, default="1.0.0",
                        help="Model version")
    parser.add_argument("--output-dir", type=str, default="model_packages",
                        help="Output directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Model name")
    parser.add_argument("--description", type=str, default=None,
                        help="Model description")
    parser.add_argument("--export-formats", type=str, default="original",
                        help="Export formats (comma-separated list)")
    parser.add_argument("--include-metadata", action="store_true",
                        help="Include training metadata if available")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply quantization to reduce model size")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules for testing")
    return parser.parse_args()


def copy_model_file(model_path, target_dir, target_name):
    """Copy the model file to the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    ext = os.path.splitext(model_path)[1]
    target_path = os.path.join(target_dir, f"{target_name}{ext}")
    shutil.copy2(model_path, target_path)
    return target_path


def get_model_size(model_path):
    """Get the size of a model file in bytes."""
    return os.path.getsize(model_path)


def export_torchscript(model, model_path, target_dir, target_name, quantize=False):
    """Export model to TorchScript format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Trace the model
        example_input = torch.randn(1, model.input_channels, 8, 8).to(next(model.parameters()).device)
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        ts_path = os.path.join(target_dir, f"{target_name}_torchscript.pt")
        traced_model.save(ts_path)
        print(f"Exported TorchScript model to {ts_path}")
        
        result = {
            "format": "torchscript",
            "path": ts_path,
            "size_bytes": get_model_size(ts_path)
        }
        
        # Quantization (if requested)
        if quantize:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Trace and export quantized model
            traced_quantized = torch.jit.trace(quantized_model, example_input)
            ts_quant_path = os.path.join(target_dir, f"{target_name}_torchscript_quantized.pt")
            traced_quantized.save(ts_quant_path)
            print(f"Exported quantized TorchScript model to {ts_quant_path}")
            
            result["quantized_path"] = ts_quant_path
            result["quantized_size_bytes"] = get_model_size(ts_quant_path)
            result["size_reduction_percent"] = 100 * (1 - result["quantized_size_bytes"] / result["size_bytes"])
        
        return result
        
    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")
        return None


def export_onnx(model, model_path, target_dir, target_name, quantize=False):
    """Export model to ONNX format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Prepare input
        example_input = torch.randn(1, model.input_channels, 8, 8).to(next(model.parameters()).device)
        
        # Export to ONNX
        onnx_path = os.path.join(target_dir, f"{target_name}.onnx")
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
        
        result = {
            "format": "onnx",
            "path": onnx_path,
            "size_bytes": get_model_size(onnx_path)
        }
        
        # Quantization (if requested)
        if quantize:
            try:
                import onnx
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                onnx_quant_path = os.path.join(target_dir, f"{target_name}_quantized.onnx")
                quantize_dynamic(
                    onnx_path,
                    onnx_quant_path,
                    weight_type=QuantType.QUInt8
                )
                print(f"Exported quantized ONNX model to {onnx_quant_path}")
                
                result["quantized_path"] = onnx_quant_path
                result["quantized_size_bytes"] = get_model_size(onnx_quant_path)
                result["size_reduction_percent"] = 100 * (1 - result["quantized_size_bytes"] / result["size_bytes"])
                
            except ImportError:
                print("ONNX quantization requires onnx and onnxruntime packages. Skipping quantization.")
        
        return result
    
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return None


def test_model(model, game_state):
    """Test the model on a game state to verify it works."""
    try:
        # Get a tensor from the game state
        tensor = torch.FloatTensor(game_state.getEnhancedTensorRepresentation())
        tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, value = model(tensor)
            policy = torch.softmax(policy_logits, dim=1)
        
        # Basic sanity checks
        assert policy.shape[1] == game_state.getActionSpaceSize(), "Policy shape mismatch"
        assert value.numel() == 1, "Value should be a single number"
        assert -1.0 <= value.item() <= 1.0, "Value should be between -1.0 and 1.0"
        
        # Test on a few more positions
        for _ in range(3):
            # Make a random legal move
            legal_moves = game_state.getLegalMoves()
            if not legal_moves or game_state.isTerminal():
                break
                
            import random
            action = random.choice(legal_moves)
            game_state.makeMove(action)
            
            # Test the model on the new position
            tensor = torch.FloatTensor(game_state.getEnhancedTensorRepresentation())
            tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)
            
            with torch.no_grad():
                policy_logits, value = model(tensor)
                policy = torch.softmax(policy_logits, dim=1)
            
            assert policy.shape[1] == game_state.getActionSpaceSize(), "Policy shape mismatch"
            assert value.numel() == 1, "Value should be a single number"
            assert -1.0 <= value.item() <= 1.0, "Value should be between -1.0 and 1.0"
        
        return True
    
    except Exception as e:
        print(f"Error testing model: {e}")
        return False


def package_model(args):
    """Package a model for distribution."""
    # Convert game type string to enum
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
    
    # Get model name
    if args.name:
        model_name = args.name
    else:
        model_basename = os.path.basename(args.model)
        model_name = os.path.splitext(model_basename)[0]
    
    # Create a standardized name for the model package
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"{args.game}_{board_size}x{board_size}_v{args.version}_{timestamp}"
    
    # Create output directory for this package
    package_dir = os.path.join(args.output_dir, package_name)
    os.makedirs(package_dir, exist_ok=True)
    
    # Create game state to get model input shape
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
        model = DDWRandWireResNet(input_channels, action_size)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Loaded model from {args.model}")
        
        # Test the model
        test_success = test_model(model, game_state)
        if not test_success:
            print("Warning: Model testing failed. Package may not work correctly.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Create metadata
    metadata = {
        "name": model_name,
        "description": args.description or f"{args.game.capitalize()} model trained using AlphaZero",
        "version": args.version,
        "game_type": args.game,
        "board_size": board_size,
        "variant_rules": args.variant,
        "created_at": datetime.now().isoformat(),
        "input_channels": input_channels,
        "action_size": action_size,
        "package_name": package_name,
        "model_architecture": "DDWRandWireResNet",
        "formats": []
    }
    
    # Get training metadata if requested and available
    if args.include_metadata:
        # Check for a metadata file with the same name as the model
        model_dir = os.path.dirname(args.model)
        model_basename = os.path.splitext(os.path.basename(args.model))[0]
        metadata_path = os.path.join(model_dir, f"{model_basename}_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    training_metadata = json.load(f)
                metadata["training_metadata"] = training_metadata
                print(f"Included training metadata from {metadata_path}")
            except Exception as e:
                print(f"Error loading training metadata: {e}")
    
    # Process export formats
    export_formats = [fmt.strip().lower() for fmt in args.export_formats.split(",")]
    
    # Always include the original model
    if "original" in export_formats:
        original_path = copy_model_file(args.model, package_dir, f"{model_name}_original")
        metadata["formats"].append({
            "format": "original",
            "path": os.path.basename(original_path),
            "size_bytes": get_model_size(original_path)
        })
    
    # Export to TorchScript if requested
    if "torchscript" in export_formats:
        torchscript_result = export_torchscript(
            model, args.model, package_dir, model_name, args.quantize
        )
        if torchscript_result:
            # Update paths to be relative to package
            torchscript_result["path"] = os.path.basename(torchscript_result["path"])
            if "quantized_path" in torchscript_result:
                torchscript_result["quantized_path"] = os.path.basename(torchscript_result["quantized_path"])
            metadata["formats"].append(torchscript_result)
    
    # Export to ONNX if requested
    if "onnx" in export_formats:
        onnx_result = export_onnx(
            model, args.model, package_dir, model_name, args.quantize
        )
        if onnx_result:
            # Update paths to be relative to package
            onnx_result["path"] = os.path.basename(onnx_result["path"])
            if "quantized_path" in onnx_result:
                onnx_result["quantized_path"] = os.path.basename(onnx_result["quantized_path"])
            metadata["formats"].append(onnx_result)
    
    # Save metadata
    metadata_path = os.path.join(package_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create a manifest file
    manifest_path = os.path.join(package_dir, "manifest.json")
    manifest = {
        "package_name": package_name,
        "game_type": args.game,
        "board_size": board_size,
        "model_name": model_name,
        "version": args.version,
        "files": [os.path.basename(metadata_path)]
    }
    
    # Add all files to manifest
    for format_info in metadata["formats"]:
        manifest["files"].append(format_info["path"])
        if "quantized_path" in format_info:
            manifest["files"].append(format_info["quantized_path"])
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["files"].append(os.path.basename(manifest_path))
    
    # Create a README.md file
    readme_path = os.path.join(package_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# {model_name} v{args.version}\n\n")
        f.write(f"{metadata['description']}\n\n")
        f.write("## Package Details\n\n")
        f.write(f"- Game: {args.game.capitalize()}\n")
        f.write(f"- Board Size: {board_size}x{board_size}\n")
        f.write(f"- Variant Rules: {'Yes' if args.variant else 'No'}\n")
        f.write(f"- Created: {metadata['created_at']}\n\n")
        f.write("## Available Formats\n\n")
        
        for format_info in metadata["formats"]:
            f.write(f"### {format_info['format'].capitalize()}\n")
            f.write(f"- File: {format_info['path']}\n")
            f.write(f"- Size: {format_info['size_bytes'] / (1024*1024):.2f} MB\n")
            
            if "quantized_path" in format_info:
                f.write(f"- Quantized File: {format_info['quantized_path']}\n")
                f.write(f"- Quantized Size: {format_info['quantized_size_bytes'] / (1024*1024):.2f} MB\n")
                f.write(f"- Size Reduction: {format_info['size_reduction_percent']:.1f}%\n")
            
            f.write("\n")
        
        f.write("## Usage\n\n")
        f.write("To use this model with the AlphaZero Multi-Game AI Engine:\n\n")
        f.write("```python\n")
        f.write("import _alphazero_cpp as az\n\n")
        f.write(f"# Create game state\n")
        f.write(f"game_state = az.createGameState(az.GameType.{args.game.upper()}, {board_size}, {str(args.variant).lower()})\n\n")
        f.write(f"# Load neural network\n")
        f.write(f"nn = az.createNeuralNetwork(\"path/to/{format_info['path']}\", az.GameType.{args.game.upper()}, {board_size})\n\n")
        f.write(f"# Create MCTS\n")
        f.write(f"tt = az.TranspositionTable(1048576, 1024)\n")
        f.write(f"mcts = az.ParallelMCTS(game_state, nn, tt, 4, 800)\n\n")
        f.write(f"# Run search\n")
        f.write(f"mcts.search()\n\n")
        f.write(f"# Get the best move\n")
        f.write(f"action = mcts.selectAction(False, 0.0)\n")
        f.write("```\n")
    
    manifest["files"].append(os.path.basename(readme_path))
    
    # Update manifest file with README
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Create a zip file of the package
    import zipfile
    package_zip_path = os.path.join(args.output_dir, f"{package_name}.zip")
    with zipfile.ZipFile(package_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in manifest["files"]:
            zipf.write(os.path.join(package_dir, file), file)
    
    print(f"\nModel package created successfully at {package_dir}")
    print(f"Package zip file created at {package_zip_path}")
    
    return True


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Package model
    success = package_model(args)
    
    if success:
        print("\nModel packaging completed successfully!")
    else:
        print("\nModel packaging failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()