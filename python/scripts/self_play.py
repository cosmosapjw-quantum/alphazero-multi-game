#!/usr/bin/env python3
"""
Self-play data generation for AlphaZero training.

This script generates self-play games using the specified model.

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
"""

import os
import sys
import argparse
import time
import json
import random
import torch
import numpy as np
import pyalphazero as az
from alphazero.models import DDWRandWireResNet

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play")
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
    parser.add_argument("--threads", type=int, default=4,
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
    return parser.parse_args()


def create_neural_network(args, game_type, board_size):
    """Create a neural network for self-play."""
    # If model path is provided, try to load it
    if args.model:
        try:
            # Try to load with the C++ API first
            nn = az.createNeuralNetwork(args.model, game_type, board_size)
            print(f"Loaded model from {args.model} (C++ API)")
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
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = DDWRandWireResNet(input_channels, action_size)
                model.load_state_dict(torch.load(args.model, map_location=device))
                model = model.to(device)
                model.eval()
                
                # Create wrapper for the PyTorch model
                class TorchNeuralNetwork(az.NeuralNetwork):
                    def __init__(self, model, device):
                        super().__init__()
                        self.model = model
                        self.device = device
                    
                    def predict(self, state):
                        # Convert state tensor to PyTorch tensor
                        state_tensor = torch.FloatTensor(state.getEnhancedTensorRepresentation())
                        state_tensor = state_tensor.unsqueeze(0).to(self.device)
                        
                        # Forward pass
                        with torch.no_grad():
                            policy_logits, value = self.model(state_tensor)
                            policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                            value = value.item()
                        
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
                        
                        # Forward pass
                        with torch.no_grad():
                            policy_logits, value_tensor = self.model(batch_tensor)
                            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
                            value_list = value_tensor.squeeze(-1).cpu().numpy()
                        
                        # Set output
                        for i in range(batch_size):
                            policies[i] = policy_probs[i].tolist()
                            values[i] = value_list[i]
                    
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
                        return 16
                    
                    def getModelInfo(self):
                        return "PyTorch DDWRandWireResNet"
                    
                    def getModelSizeBytes(self):
                        return sum(p.numel() * 4 for p in self.model.parameters())
                    
                    def benchmark(self, numIterations=100, batchSize=16):
                        pass
                    
                    def enableDebugMode(self, enable):
                        pass
                    
                    def printModelSummary(self):
                        pass
                
                nn = TorchNeuralNetwork(model, device)
                print(f"Loaded model from {args.model} (PyTorch)")
                return nn
            except Exception as e:
                print(f"Failed to load model with PyTorch: {e}")
                print("Using random policy network instead")
                return None
    
    # If no model path or loading failed, use random policy
    print("Using random policy network")
    return None


def run_self_play(args):
    """Run self-play games."""
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
    
    # Set progress callback
    def progress_callback(game, move, total_games, total_moves):
        sys.stdout.write(f"\rGame {game+1}/{total_games}, Move {move+1}")
        sys.stdout.flush()
    
    # Generate games
    print(f"Generating {args.num_games} self-play games...")
    print(f"Game type: {args.game}")
    print(f"Board size: {board_size}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Threads: {args.threads}")
    print(f"Output directory: {args.output_dir}")
    
    start_time = time.time()
    games = self_play.generateGames(game_type, board_size, args.variant)
    end_time = time.time()
    
    # Print results
    print(f"\nGenerated {len(games)} games in {end_time - start_time:.2f} seconds")
    
    # Calculate some statistics
    total_moves = 0
    for game in games:
        total_moves += len(game.getMoveHistory())
    
    print(f"Total moves: {total_moves}")
    print(f"Average moves per game: {total_moves / len(games):.1f}")
    print(f"Average time per game: {(end_time - start_time) / len(games):.2f} seconds")
    print(f"Average moves per second: {total_moves / (end_time - start_time):.1f}")
    
    # Save a metadata file with the run information
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "game": args.game,
        "board_size": board_size,
        "num_games": args.num_games,
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
        "avg_moves_per_game": total_moves / len(games),
        "total_time": end_time - start_time,
        "avg_time_per_game": (end_time - start_time) / len(games),
        "avg_moves_per_second": total_moves / (end_time - start_time)
    }
    
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {os.path.join(args.output_dir, 'metadata.json')}")


if __name__ == "__main__":
    args = parse_args()
    run_self_play(args)