#!/usr/bin/env python3
"""
Self-play data generation for AlphaZero training.

This script generates self-play games using the specified model with
support for GPU batching to significantly improve performance.

Usage:
    python self_play.py [options]

Options:
    --model MODEL         Path to model file
    --game {gomoku,chess,go}  Game type (default: gomoku)
    --size SIZE           Board size (default: depends on game)
    --num-games NUM       Number of games to generate (default: 100)
    --simulations SIMS    Number of MCTS simulations per move (default: 800)
    --threads THREADS     Number of threads (default: 4)
    --output-dir DIR      Output directory (default: data/games)
    --temperature TEMP    Initial temperature (default: 1.0)
    --temp-drop MOVE      Move to drop temperature (default: 30)
    --final-temp TEMP     Final temperature (default: 0.0)
    --dirichlet-alpha A   Dirichlet noise alpha (default: 0.03)
    --dirichlet-epsilon E Dirichlet noise weight (default: 0.25)
    --variant             Use variant rules (Renju, Chess960, Chinese)
    --batch-size SIZE     Batch size for neural network inference (default: 8)
    --batch-timeout MS    Timeout for batch completion in milliseconds (default: 100)
    --no-gpu              Disable GPU acceleration
    --no-batched-search   Disable batched MCTS search
    --fp16                Use FP16 precision (faster but less accurate)
    --export-model        Export PyTorch model to LibTorch format
    --create-random-model Create and export a random model if no model is provided
"""

import os
import sys
import argparse
import time
import json
import random
import torch
import numpy as np
import tempfile
import threading

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the build directory for the C++ extension
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'pybind'))
sys.path.insert(0, build_dir)

try:
    from alphazero.models import DDWRandWireResNet
    import _alphazero_cpp as az
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the project is properly built and the Python package is installed.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play with GPU Batching")
    parser.add_argument("--model", type=str, default="",
                        help="Path to model file")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to generate")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=12,
                        help="Number of threads")
    parser.add_argument("--output-dir", type=str, default="data/games",
                        help="Output directory for game records")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Initial temperature")
    parser.add_argument("--temp-drop", type=int, default=30,
                        help="Move number to drop temperature")
    parser.add_argument("--final-temp", type=float, default=0.0,
                        help="Final temperature")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.03,
                        help="Dirichlet noise alpha parameter")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Dirichlet noise weight")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    # GPU and batching related arguments
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for neural network inference")
    parser.add_argument("--batch-timeout", type=int, default=100,
                        help="Timeout for batch completion in milliseconds")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--no-batched-search", action="store_true",
                        help="Disable batched MCTS search")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision (faster but less accurate)")
    parser.add_argument("--export-model", action="store_true",
                        help="Export PyTorch model to LibTorch format")
    parser.add_argument("--create-random-model", action="store_true",
                        help="Create and export a random model if no model is provided")
    
    return parser.parse_args()


def export_pytorch_to_libtorch(model, input_shape, output_path):
    """Export a PyTorch model to LibTorch format for use with C++ API.
    
    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (batch_size, channels, height, width)
        output_path: Path to save the exported model
        
    Returns:
        Path to the exported model
    """
    print(f"Exporting PyTorch model to LibTorch format: {output_path}")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create a dummy input tensor
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        model = model.cuda()
        
    # Trace the model
    try:
        print("Tracing model...")
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Test the traced model
        print("Testing traced model...")
        traced_output = traced_model(dummy_input)
        model_output = model(dummy_input)
        
        # Check if outputs match
        policy_match = torch.allclose(traced_output[0], model_output[0], atol=1e-5)
        value_match = torch.allclose(traced_output[1], model_output[1], atol=1e-5)
        
        if not policy_match or not value_match:
            print("WARNING: Traced model outputs don't match original model!")
            print(f"  Policy match: {policy_match}")
            print(f"  Value match: {value_match}")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the traced model
        traced_model.save(output_path)
        print(f"Model successfully exported to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to export model: {e}")
        raise


def check_neural_network_gil_safety(nn):
    """Check if a neural network is safe to use with parallel processing.
    
    Args:
        nn: Neural network to check
        
    Returns:
        bool: True if the neural network is GIL-safe, False otherwise
    """
    if nn is None:
        # Random policy network is GIL-safe
        print("Using GIL-safe random policy network")
        return True
        
    if hasattr(nn, 'is_gil_safe') and nn.is_gil_safe():
        print("Using GIL-safe C++ neural network - parallel processing will be efficient.")
        return True
    else:
        print("\nWARNING: Using a Python neural network, which will limit parallelism due to the GIL.")
        print("For best performance, export your model to LibTorch format and use the C++ API.")
        print("Try running with the --export-model flag to automatically export your model.")
        return False


def benchmark_gil_impact(nn, num_threads=4):
    """Benchmark the impact of the GIL on parallel processing with this neural network.
    
    Args:
        nn: Neural network to benchmark
        num_threads: Number of threads to test
        
    Returns:
        float: Actual speedup achieved
    """
    if nn is None:
        print("Cannot benchmark - no neural network provided")
        return 0.0
        
    print(f"\nBenchmarking GIL impact with {num_threads} threads...")
    
    # Create a test game state
    game_state = az.createGameState(az.GameType.GOMOKU, 15, False)
    
    # Function to run in each thread
    def worker():
        start = time.time()
        for _ in range(100):
            nn.predict(game_state)
        return time.time() - start
    
    # Run in a single thread
    print("Running single-thread test...")
    single_thread_time = worker()
    
    # Run in multiple threads
    print(f"Running {num_threads}-thread test...")
    threads = []
    thread_times = [0] * num_threads
    
    for i in range(num_threads):
        threads.append(threading.Thread(
            target=lambda idx=i: thread_times.__setitem__(idx, worker())
        ))
    
    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    multi_thread_time = time.time() - start
    
    # Calculate speedup
    ideal_speedup = num_threads
    actual_speedup = single_thread_time * num_threads / multi_thread_time
    efficiency = actual_speedup / ideal_speedup * 100
    
    print(f"\nSingle thread time: {single_thread_time:.3f} s")
    print(f"Multi-thread time: {multi_thread_time:.3f} s for {num_threads} threads")
    print(f"Actual speedup: {actual_speedup:.2f}x (Ideal: {ideal_speedup:.2f}x)")
    print(f"Efficiency: {efficiency:.1f}%")
    
    if actual_speedup < ideal_speedup * 0.5:
        print("\nWARNING: Poor scaling detected - likely due to GIL issues.")
        print("Consider using a C++-based neural network for better parallel performance.")
    
    return actual_speedup


def create_random_model(input_channels, action_size, board_size):
    """Create a random model for testing and export.
    
    Args:
        input_channels: Number of input channels
        action_size: Size of action space
        board_size: Board size
        
    Returns:
        PyTorch model
    """
    print("Creating a random DDWRandWireResNet model...")
    model = DDWRandWireResNet(input_channels, action_size, channels=64, num_blocks=8)
    
    # Initialize randomly
    for param in model.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
            
    model.eval()
    return model


def create_neural_network(args, game_type, board_size):
    """Create a neural network for self-play with GPU batching support.
    
    Args:
        args: Command-line arguments
        game_type: Game type enum value
        board_size: Board size
        
    Returns:
        Neural network object
    """
    # Determine if GPU should be used
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    
    # Create a test game state to get input shape and action size for model
    game_state = az.createGameState(game_type, board_size, args.variant)
    tensor_rep = game_state.getEnhancedTensorRepresentation()
    input_channels = len(tensor_rep)
    action_size = game_state.getActionSpaceSize()
    
    # If export is requested but no model is provided, create a random model
    if args.export_model and not args.model and args.create_random_model:
        print("\nExport model requested but no model provided. Creating a random model...")
        # Create a random model
        model = create_random_model(input_channels, action_size, board_size)
        
        # Export to LibTorch
        # Create output directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Export location
        libtorch_path = f"models/random_model_{args.game}_{board_size}x{board_size}.pt"
        
        # Create input shape based on the game state
        input_shape = (1, input_channels, board_size, board_size)
        
        try:
            # Export the model
            libtorch_path = export_pytorch_to_libtorch(model, input_shape, libtorch_path)
            
            # Load the exported model with C++ API
            nn = az.createNeuralNetwork(libtorch_path, game_type, board_size, use_gpu)
            print(f"Created and exported random model to {libtorch_path}")
            
            # Verify GIL safety
            check_neural_network_gil_safety(nn)
            
            print("\nYou can now use this model for testing with:")
            print(f"  python self_play.py --model {libtorch_path}")
            
            return nn
        except Exception as e:
            print(f"Failed to export random model: {e}")
            print("Using random policy network instead")
            return None
            
    # If model path is provided, try to load it
    if args.model:
        try:
            # Try to load with the C++ API first (loads LibTorch models)
            nn = az.createNeuralNetwork(args.model, game_type, board_size, use_gpu)
            print(f"Loaded model from {args.model} (C++ API)")
            if use_gpu:
                print(f"Using GPU acceleration with device: {nn.getDeviceInfo()}")
                print(f"Average inference time: {nn.getInferenceTimeMs():.2f} ms")
                print(f"Batch size: {nn.getBatchSize()}")
            else:
                print(f"Using CPU: {nn.getDeviceInfo()}")
            
            # Verify GIL safety
            check_neural_network_gil_safety(nn)
            return nn
            
        except Exception as e:
            print(f"Failed to load model with C++ API: {e}")
            
            # Try to load with PyTorch
            try:
                # Create and load model
                device = torch.device("cuda" if use_gpu else "cpu")
                model = DDWRandWireResNet(input_channels, action_size)
                model.load_state_dict(torch.load(args.model, map_location=device))
                model = model.to(device)
                model.eval()
                
                # Export to LibTorch if requested or needed
                if args.export_model:
                    # Create temp model path if not explicitly exporting
                    libtorch_path = f"{os.path.splitext(args.model)[0]}_libtorch.pt"
                        
                    # Create input shape based on the game state
                    input_shape = (1, input_channels, board_size, board_size)
                    
                    # Export the model
                    libtorch_path = export_pytorch_to_libtorch(model, input_shape, libtorch_path)
                    
                    # Load the exported model with C++ API
                    nn = az.createNeuralNetwork(libtorch_path, game_type, board_size, use_gpu)
                    print(f"Loaded model via PyTorch export to {libtorch_path}")
                    
                    # Verify GIL safety
                    check_neural_network_gil_safety(nn)
                    return nn
                else:
                    # Create Python wrapper for PyTorch model
                    print("Using Python-backed neural network (performance will be limited)")
                    
                    # Create a custom Python-backed neural network
                    class PythonNeuralNetwork:
                        def __init__(self, model, device):
                            self.model = model
                            self.device = device
                            self.inference_times = []
                            
                        def predict(self, state):
                            # Convert state tensor to PyTorch tensor
                            state_tensor = torch.FloatTensor(state.getEnhancedTensorRepresentation())
                            state_tensor = state_tensor.unsqueeze(0).to(self.device)
                            
                            # Forward pass
                            start_time = time.time()
                            with torch.no_grad():
                                policy_logits, value = self.model(state_tensor)
                                policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                                value = value.item()
                            
                            # Record inference time
                            end_time = time.time()
                            self.inference_times.append((end_time - start_time) * 1000)  # ms
                            if len(self.inference_times) > 100:
                                self.inference_times.pop(0)
                            
                            return policy, value
                        
                        def getInferenceTimeMs(self):
                            if not self.inference_times:
                                return 0.0
                            return sum(self.inference_times) / len(self.inference_times)
                        
                        def getDeviceInfo(self):
                            if self.device.type == 'cuda':
                                return f"GPU: {torch.cuda.get_device_name(0)} (Python API)"
                            else:
                                return "CPU (Python API)"
                        
                        def getBatchSize(self):
                            return 1  # No batching in Python wrapper
                        
                        def getModelInfo(self):
                            return "PyTorch Model (Python API - Limited Performance)"
                        
                        def is_gil_safe(self):
                            return False  # Python models are not GIL-safe
                    
                    # Return Python-backed neural network
                    nn = PythonNeuralNetwork(model, device)
                    check_neural_network_gil_safety(nn)
                    return nn
                
            except Exception as e:
                print(f"Failed to load model with PyTorch: {e}")
                print("Using random policy network instead")
                return None
    
    # If no model path or loading failed, use random policy
    print("Using random policy network")
    return None


def run_self_play(args):
    """Run self-play games with GPU batching support.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
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
            board_size = 8  # Chess is always 8x8
        elif args.game == "go":
            board_size = 19  # Default to 19x19
        else:
            board_size = 15
    else:
        board_size = args.size
    
    # Check if we're just exporting a model
    if args.export_model and not args.model and args.create_random_model:
        # Create neural network (which will create and export a random model)
        neural_network = create_neural_network(args, game_type, board_size)
        print("\nModel export complete.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if GPU and batched search should be used
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    use_batched_search = use_gpu and not args.no_batched_search
    
    # Create neural network
    neural_network = create_neural_network(args, game_type, board_size)
    
    # Benchmark GIL impact if we have a neural network
    if neural_network is not None:
        benchmark_gil_impact(neural_network, min(4, args.threads))
    
    # Create self-play manager
    self_play = az.SelfPlayManager(
        neural_network, args.num_games, args.simulations, args.threads
    )
    
    # Set exploration parameters
    self_play.setExplorationParams(
        args.dirichlet_alpha,
        args.dirichlet_epsilon,
        args.temperature,
        args.temp_drop,
        args.final_temp
    )
    
    # Set output directory
    self_play.setSaveGames(True, args.output_dir)
    
    # Generate games
    print(f"\nGenerating {args.num_games} self-play games...")
    print(f"Game: {args.game.upper()}, Board size: {board_size}x{board_size}")
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"Threads: {args.threads}")
    print(f"Using GPU: {use_gpu}")
    print(f"Using batched search: {use_batched_search}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Performance monitoring variables
    start_time = time.time()
    last_check_time = start_time
    last_check_games = 0
    
    try:
        games = self_play.generateGames(game_type, board_size, args.variant)
        
        # Print periodic progress updates
        current_time = time.time()
        if current_time - last_check_time > 10:  # Update every 10 seconds
            games_completed = len(games) - last_check_games
            elapsed = current_time - last_check_time
            
            if games_completed > 0:
                print(f"Progress: {len(games)}/{args.num_games} games, " +
                      f"Rate: {games_completed / elapsed:.2f} games/sec")
            
            last_check_time = current_time
            last_check_games = len(games)
            
    except KeyboardInterrupt:
        print("\nSelf-play interrupted. Saving completed games...")
        # In case of interruption, we still process any completed games
        games = []  # Reset to empty list since we can't access interrupted games
    
    end_time = time.time()
    
    # Print results
    print(f"\nGenerated {len(games)} games in {end_time - start_time:.2f} seconds")
    
    # If no games were completed, exit
    if not games:
        print("No complete games were generated.")
        return
    
    # Calculate statistics
    total_moves = 0
    for game in games:
        total_moves += len(game.getMoves())
    
    # Handle case where games were completed
    if len(games) > 0:
        avg_moves_per_game = total_moves / len(games)
        avg_time_per_game = (end_time - start_time) / len(games)
        avg_moves_per_second = total_moves / (end_time - start_time)
        
        print(f"Total moves: {total_moves}")
        print(f"Average moves per game: {avg_moves_per_game:.1f}")
        print(f"Average time per game: {avg_time_per_game:.2f} seconds")
        print(f"Average moves per second: {avg_moves_per_second:.1f}")
        
        # Performance details if neural network is available
        if neural_network and hasattr(neural_network, 'getInferenceTimeMs'):
            inference_time = neural_network.getInferenceTimeMs()
            if inference_time > 0:
                print(f"Average neural network inference time: {inference_time:.2f} ms")
                percentage_time = inference_time * avg_moves_per_game * args.simulations / (avg_time_per_game * 1000) * 100
                print(f"Neural network time: {percentage_time:.1f}% of total")
        
        # Save a metadata file with the run information
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game": args.game,
            "board_size": board_size,
            "num_games": len(games),
            "simulations": args.simulations,
            "threads": args.threads,
            "temperature": args.temperature,
            "temp_drop": args.temp_drop,
            "final_temp": args.final_temp,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_epsilon": args.dirichlet_epsilon,
            "variant": args.variant,
            "model": args.model,
            "total_moves": total_moves,
            "avg_moves_per_game": avg_moves_per_game,
            "total_time": end_time - start_time,
            "avg_time_per_game": avg_time_per_game,
            "avg_moves_per_second": avg_moves_per_second,
            "use_gpu": use_gpu,
            "use_batched_search": use_batched_search,
            "batch_size": args.batch_size,
            "batch_timeout": args.batch_timeout,
            "fp16": args.fp16
        }
        
        # Add neural network information if available
        if neural_network:
            if hasattr(neural_network, 'getInferenceTimeMs'):
                metadata["inference_time_ms"] = neural_network.getInferenceTimeMs()
            if hasattr(neural_network, 'getDeviceInfo'):
                metadata["device_info"] = neural_network.getDeviceInfo()
            if hasattr(neural_network, 'getModelInfo'):
                metadata["model_info"] = neural_network.getModelInfo()
            if hasattr(neural_network, 'is_gil_safe'):
                metadata["is_gil_safe"] = neural_network.is_gil_safe()
        
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
    else:
        print("No statistics to calculate as no games were completed.")


if __name__ == "__main__":
    args = parse_args()
    run_self_play(args)