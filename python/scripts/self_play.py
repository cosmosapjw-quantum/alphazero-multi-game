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
"""

import os
import sys
import argparse
import time
import json
import random
import torch
import numpy as np

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
    
    return parser.parse_args()


def create_neural_network(args, game_type, board_size):
    """Create a neural network for self-play with GPU batching support."""
    # Determine if GPU should be used
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    
    # If model path is provided, try to load it
    if args.model:
        try:
            # Try to load with the C++ API first
            nn = az.createNeuralNetwork(args.model, game_type, board_size, use_gpu)
            print(f"Loaded model from {args.model} (C++ API)")
            if use_gpu:
                print(f"Using GPU acceleration with device: {nn.getDeviceInfo()}")
                print(f"Average inference time: {nn.getInferenceTimeMs():.2f} ms")
                print(f"Batch size: {nn.getBatchSize()}")
            else:
                print(f"Using CPU: {nn.getDeviceInfo()}")
            return nn
        except Exception as e:
            print(f"Failed to load model with C++ API: {e}")
            
            # Try to load with PyTorch
            try:
                # Create a test game state to get input shape
                game_state = az.createGameState(game_type, board_size, args.variant)
                tensor_rep = game_state.getEnhancedTensorRepresentation()
                input_channels = len(tensor_rep)
                action_size = game_state.getActionSpaceSize()
                
                # Create and load model
                device = torch.device("cuda" if use_gpu else "cpu")
                model = DDWRandWireResNet(input_channels, action_size)
                model.load_state_dict(torch.load(args.model, map_location=device))
                model = model.to(device)
                model.eval()
                
                # Enable FP16 if requested and GPU is available
                if args.fp16 and use_gpu and hasattr(torch.cuda, 'amp'):
                    print("Using FP16 precision for faster inference")
                
                # Create wrapper for the PyTorch model with batch support
                class TorchNeuralNetwork(az.NeuralNetwork):
                    def __init__(self, model, device, batch_size=8, use_fp16=False):
                        super().__init__()
                        self.model = model
                        self.device = device
                        self.batch_size = batch_size
                        self.use_fp16 = use_fp16 and device.type == 'cuda' and hasattr(torch.cuda, 'amp')
                        self.inference_times = []
                    
                    def predict(self, state):
                        # Convert state tensor to PyTorch tensor
                        state_tensor = torch.FloatTensor(state.getEnhancedTensorRepresentation())
                        state_tensor = state_tensor.unsqueeze(0).to(self.device)
                        
                        # Use FP16 if enabled
                        if self.use_fp16:
                            state_tensor = state_tensor.half()
                        
                        # Measure inference time
                        start_time = time.time()
                        
                        # Forward pass
                        with torch.no_grad():
                            if self.use_fp16:
                                with torch.cuda.amp.autocast():
                                    policy_logits, value = self.model(state_tensor)
                            else:
                                policy_logits, value = self.model(state_tensor)
                            
                            policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                            value = value.item()
                        
                        # Record inference time
                        end_time = time.time()
                        self.inference_times.append((end_time - start_time) * 1000)  # ms
                        if len(self.inference_times) > 100:
                            self.inference_times.pop(0)
                        
                        return policy, value
                    
                    def predictBatch(self, states, policies, values):
                        # Convert states to PyTorch tensor
                        batch_size = len(states)
                        state_tensors = []
                        
                        for i in range(batch_size):
                            state = states[i].get()
                            state_tensor = torch.FloatTensor(state.getEnhancedTensorRepresentation())
                            state_tensors.append(state_tensor)
                        
                        batch_tensor = torch.stack(state_tensors).to(self.device)
                        
                        # Use FP16 if enabled
                        if self.use_fp16:
                            batch_tensor = batch_tensor.half()
                        
                        # Measure inference time
                        start_time = time.time()
                        
                        # Forward pass
                        with torch.no_grad():
                            if self.use_fp16:
                                with torch.cuda.amp.autocast():
                                    policy_logits, value_tensor = self.model(batch_tensor)
                            else:
                                policy_logits, value_tensor = self.model(batch_tensor)
                            
                            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
                            value_list = value_tensor.squeeze(-1).cpu().numpy()
                        
                        # Record inference time
                        end_time = time.time()
                        self.inference_times.append((end_time - start_time) * 1000 / batch_size)  # ms per sample
                        if len(self.inference_times) > 100:
                            self.inference_times.pop(0)
                        
                        # Clear existing data
                        policies.clear()
                        values.clear()
                        
                        # Fill output vectors
                        for i in range(batch_size):
                            policies.append(policy_probs[i].tolist())
                            values.append(float(value_list[i]))
                    
                    def isGpuAvailable(self):
                        return torch.cuda.is_available() and self.device.type == 'cuda'
                    
                    def getDeviceInfo(self):
                        if self.isGpuAvailable():
                            gpu_name = torch.cuda.get_device_name(0)
                            precision = "FP16" if self.use_fp16 else "FP32"
                            return f"GPU: {gpu_name} ({precision})"
                        else:
                            return "CPU"
                    
                    def getInferenceTimeMs(self):
                        if not self.inference_times:
                            return 0.0
                        return sum(self.inference_times) / len(self.inference_times)
                    
                    def getBatchSize(self):
                        return self.batch_size
                    
                    def getModelInfo(self):
                        return f"PyTorch DDWRandWireResNet ({self.model.num_nodes} nodes)"
                    
                    def getModelSizeBytes(self):
                        return sum(p.numel() * (2 if self.use_fp16 else 4) for p in self.model.parameters())
                    
                    def benchmark(self, numIterations=100, batchSize=16):
                        if not self.isGpuAvailable():
                            print("Benchmarking skipped - GPU not available")
                            return
                        
                        # Create a dummy state
                        state = states[0].get() if len(states) > 0 else None
                        if state is None:
                            print("Cannot benchmark - no valid state available")
                            return
                        
                        # Warmup
                        for _ in range(10):
                            self.predict(state)
                        
                        # Single inference benchmark
                        start_time = time.time()
                        for _ in range(numIterations):
                            self.predict(state)
                        end_time = time.time()
                        single_time = (end_time - start_time) * 1000 / numIterations
                        
                        print(f"Single inference: {single_time:.2f} ms")
                        
                        # Batch inference benchmark
                        test_states = [state] * batchSize
                        test_policies = []
                        test_values = []
                        
                        start_time = time.time()
                        for _ in range(numIterations // batchSize + 1):
                            self.predictBatch(test_states, test_policies, test_values)
                        end_time = time.time()
                        batch_time = (end_time - start_time) * 1000 / numIterations
                        
                        print(f"Batch inference: {batch_time:.2f} ms per sample (batch size: {batchSize})")
                        print(f"Speedup: {single_time / batch_time:.2f}x")
                    
                    def enableDebugMode(self, enable):
                        pass
                
                nn = TorchNeuralNetwork(model, device, args.batch_size, args.fp16)
                print(f"Loaded model from {args.model} (PyTorch)")
                if device.type == 'cuda':
                    print(f"Using GPU acceleration with device: {torch.cuda.get_device_name(0)}")
                    if args.fp16:
                        print("Using FP16 precision for faster inference")
                else:
                    print("Using CPU")
                return nn
            except Exception as e:
                print(f"Failed to load model with PyTorch: {e}")
                print("Using random policy network instead")
                return None
    
    # If no model path or loading failed, use random policy
    print("Using random policy network")
    return None


def run_self_play(args):
    """Run self-play games with GPU batching support."""
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if GPU and batched search should be used
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    use_batched_search = use_gpu and not args.no_batched_search
    
    # Create neural network
    neural_network = create_neural_network(args, game_type, board_size)
    
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
        
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
    else:
        print("No statistics to calculate as no games were completed.")


if __name__ == "__main__":
    args = parse_args()
    run_self_play(args)