#!/usr/bin/env python3
"""
Evaluation script for AlphaZero models.

This script evaluates AlphaZero models by running tournaments and calculating ELO ratings.

Usage:
    python evaluate.py [options]

Options:
    --model-a MODEL     Path to first model file (required)
    --model-b MODEL     Path to second model file (optional, uses random policy if not provided)
    --game {gomoku,chess,go}  Game type (default: gomoku)
    --size SIZE         Board size (default: depends on game)
    --games NUM         Number of games to play (default: 100)
    --simulations SIMS  Number of MCTS simulations per move (default: 800)
    --threads THREADS   Number of threads (default: 4)
    --output-dir DIR    Output directory for results (default: results)
    --temperature TEMP  Temperature for move selection (default: 0.1)
    --time-control SEC  Time per move in seconds (default: none)
    --variant           Use variant rules (Renju, Chess960, Chinese)
    --swap              Swap colors/sides halfway through games
    --debug             Enable debug output
"""

import os
import sys
import argparse
import json
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyalphazero as az
from alphazero.models import DDWRandWireResNet
from alphazero.utils.elo import EloRating, calculate_elo_change

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Model Evaluation")
    parser.add_argument("--model-a", type=str, required=True,
                        help="Path to first model file")
    parser.add_argument("--model-b", type=str, default="",
                        help="Path to second model file (optional)")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for move selection")
    parser.add_argument("--time-control", type=float, default=None,
                        help="Time per move in seconds (overrides simulations)")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--swap", action="store_true",
                        help="Swap colors/sides halfway through games")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def create_neural_network(model_path, game_type, board_size):
    """Create a neural network wrapper for the AI."""
    if not model_path:
        print("Using random policy (no model provided)")
        return None
    
    try:
        # Try to load with the C++ API first
        nn = az.createNeuralNetwork(model_path, game_type, board_size)
        print(f"Loaded model from {model_path} (C++ API)")
        return nn
    except Exception as e:
        print(f"Failed to load model with C++ API: {e}")
        
        # Try to load with PyTorch
        try:
            # Create a test game state to get input shape
            game_state = az.createGameState(game_type, board_size, False)
            tensor_rep = game_state.getEnhancedTensorRepresentation()
            input_channels = len(tensor_rep)
            action_size = game_state.getActionSpaceSize()
            
            # Create and load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DDWRandWireResNet(input_channels, action_size)
            model.load_state_dict(torch.load(model_path, map_location=device))
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
            
            nn = TorchNeuralNetwork(model, device)
            print(f"Loaded model from {model_path} (PyTorch)")
            return nn
        except Exception as e:
            print(f"Failed to load model with PyTorch: {e}")
            print("Using random policy network instead")
            return None


def play_game(game_state, mcts_a, mcts_b, args, game_num=0, start_player_a=True):
    """Play a single game between two MCTS instances."""
    history = []
    move_times = []
    current_player_a = start_player_a
    
    # For time-based control
    time_per_move = args.time_control
    
    while not game_state.isTerminal():
        start_time = time.time()
        
        # Select the MCTS for the current player
        mcts = mcts_a if current_player_a else mcts_b
        
        # Run search based on time or simulations
        if time_per_move is not None:
            # Time-based search
            end_time = start_time + time_per_move
            mcts.setNumSimulations(10000000)  # Very high number
            
            def progress_callback(current, total):
                if time.time() >= end_time:
                    # Stop search when time is up
                    mcts.setAbort(True)
            
            mcts.setProgressCallback(progress_callback)
            mcts.search()
            mcts.setAbort(False)  # Reset abort flag
        else:
            # Simulation-based search
            mcts.search()
        
        # Select action
        action = mcts.selectAction(True, args.temperature)
        
        # Record move and timing
        end_time = time.time()
        move_time = end_time - start_time
        move_times.append(move_time)
        
        # Convert action to human-readable form
        move_str = game_state.actionToString(action)
        
        # Store history
        history.append({
            'action': action,
            'move_str': move_str,
            'player': 'A' if current_player_a else 'B',
            'move_time': move_time,
            'game_player': game_state.getCurrentPlayer()
        })
        
        if args.debug:
            print(f"Game {game_num+1}: Player {'A' if current_player_a else 'B'} plays {move_str} in {move_time:.2f}s")
        
        # Make move
        game_state.makeMove(action)
        
        # Update MCTS trees
        mcts_a.updateWithMove(action)
        mcts_b.updateWithMove(action)
        
        # Switch player
        current_player_a = not current_player_a
    
    # Get game result
    result = game_state.getGameResult()
    
    # Calculate scores based on who played which color/side
    score_a = 0
    score_b = 0
    
    if result == az.GameResult.WIN_PLAYER1:
        # Player 1 (Black/White in Chess) won
        if start_player_a:
            # A started as Player 1
            score_a = 1
        else:
            # B started as Player 1
            score_b = 1
    elif result == az.GameResult.WIN_PLAYER2:
        # Player 2 (White/Black in Chess) won
        if start_player_a:
            # A started as Player 1, so B won
            score_b = 1
        else:
            # B started as Player 1, so A won
            score_a = 1
    else:
        # Draw
        score_a = 0.5
        score_b = 0.5
    
    # Prepare game record
    game_record = {
        'moves': history,
        'result': str(result),
        'score_a': score_a,
        'score_b': score_b,
        'move_times': move_times,
        'avg_move_time': sum(move_times) / len(move_times) if move_times else 0,
        'total_moves': len(history)
    }
    
    return game_record


def run_evaluation(args):
    """Run an evaluation tournament between two models."""
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
            board_size = 9  # Default to 9x9 for evaluation (faster than 19x19)
        else:
            board_size = 15
    else:
        board_size = args.size
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the neural networks
    nn_a = create_neural_network(args.model_a, game_type, board_size)
    nn_b = create_neural_network(args.model_b, game_type, board_size)
    
    # Create ELO tracker for rating calculation
    elo_tracker = EloRating(initial_rating=1500.0, k_factor=32.0)
    
    # Create transposition tables (separate for each model)
    tt_a = az.TranspositionTable(1048576, 1024)
    tt_b = az.TranspositionTable(1048576, 1024)
    
    # Game results
    wins_a = 0
    wins_b = 0
    draws = 0
    game_records = []
    
    # Model identifiers
    model_a_id = args.model_a if args.model_a else "RandomPolicy"
    model_b_id = args.model_b if args.model_b else "RandomPolicy"
    
    # Adjust short name
    model_a_short = os.path.splitext(os.path.basename(model_a_id))[0] if args.model_a else "Random"
    model_b_short = os.path.splitext(os.path.basename(model_b_id))[0] if args.model_b else "Random"
    
    print(f"Evaluating {model_a_short} vs {model_b_short}")
    print(f"Game type: {args.game}")
    print(f"Board size: {board_size}")
    print(f"Number of games: {args.games}")
    print(f"Simulations per move: {args.simulations}")
    print(f"Time control: {args.time_control}s per move" if args.time_control else "No time control")
    print(f"Temperature: {args.temperature}")
    print(f"Variant rules: {'Yes' if args.variant else 'No'}")
    
    # Run games
    for game_num in tqdm(range(args.games), desc="Playing games"):
        # Determine starting player for this game
        if args.swap:
            # If swap is enabled, alternate starting player every game
            start_player_a = (game_num % 2 == 0)
        else:
            # Otherwise, player A always starts
            start_player_a = True
        
        # Create a new game state
        game_state = az.createGameState(game_type, board_size, args.variant)
        
        # Create MCTS for both players
        mcts_a = az.ParallelMCTS(
            game_state, nn_a, tt_a, 
            args.threads, args.simulations
        )
        
        mcts_b = az.ParallelMCTS(
            game_state, nn_b, tt_b, 
            args.threads, args.simulations
        )
        
        # Set MCTS parameters
        mcts_a.setCPuct(1.5)
        mcts_a.setFpuReduction(0.0)
        
        mcts_b.setCPuct(1.5)
        mcts_b.setFpuReduction(0.0)
        
        # Play the game
        game_record = play_game(game_state, mcts_a, mcts_b, args, game_num, start_player_a)
        
        # Update statistics
        if game_record['score_a'] == 1:
            wins_a += 1
        elif game_record['score_b'] == 1:
            wins_b += 1
        else:
            draws += 1
        
        # Add game number and starting player info
        game_record['game_num'] = game_num
        game_record['start_player_a'] = start_player_a
        
        # Store game record
        game_records.append(game_record)
        
        # Update ELO ratings
        elo_tracker.add_game_result(
            player_id=model_a_id,
            opponent_id=model_b_id,
            score=game_record['score_a']
        )
    
    # Calculate final statistics
    total_games = wins_a + wins_b + draws
    win_rate_a = wins_a / total_games * 100
    win_rate_b = wins_b / total_games * 100
    draw_rate = draws / total_games * 100
    
    # Get final ELO ratings
    elo_a = elo_tracker.get_rating(model_a_id)
    elo_b = elo_tracker.get_rating(model_b_id)
    
    # Create results summary
    results = {
        'model_a': model_a_id,
        'model_b': model_b_id,
        'model_a_short': model_a_short,
        'model_b_short': model_b_short,
        'game_type': args.game,
        'board_size': board_size,
        'variant_rules': args.variant,
        'games_played': total_games,
        'wins_a': wins_a,
        'wins_b': wins_b,
        'draws': draws,
        'win_rate_a': win_rate_a,
        'win_rate_b': win_rate_b,
        'draw_rate': draw_rate,
        'elo_a': elo_a,
        'elo_b': elo_b,
        'simulations': args.simulations,
        'threads': args.threads,
        'temperature': args.temperature,
        'time_control': args.time_control,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'settings': vars(args)
    }
    
    # Calculate average move time
    total_move_times = []
    for record in game_records:
        total_move_times.extend(record['move_times'])
    
    results['avg_move_time'] = sum(total_move_times) / len(total_move_times) if total_move_times else 0
    results['total_moves'] = len(total_move_times)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"  {model_a_short}: {wins_a} wins ({win_rate_a:.1f}%), ELO: {elo_a:.1f}")
    print(f"  {model_b_short}: {wins_b} wins ({win_rate_b:.1f}%), ELO: {elo_b:.1f}")
    print(f"  Draws: {draws} ({draw_rate:.1f}%)")
    print(f"  Total games: {total_games}")
    print(f"  Average move time: {results['avg_move_time']:.3f}s")
    print(f"  Total moves: {results['total_moves']}")
    
    # Save results
    result_file = os.path.join(args.output_dir, f"{args.game}_{model_a_short}_vs_{model_b_short}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {result_file}")
    
    # Generate a visual summary
    generate_results_plot(results, args.output_dir)
    
    return results


def generate_results_plot(results, output_dir):
    """Generate a visual summary of evaluation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Win rates pie chart
    labels = [f"{results['model_a_short']} Wins", f"{results['model_b_short']} Wins", "Draws"]
    sizes = [results['wins_a'], results['wins_b'], results['draws']]
    colors = ['#5DA5DA', '#FAA43A', '#60BD68']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    ax1.set_title('Game Outcomes')
    
    # ELO comparison
    models = [results['model_a_short'], results['model_b_short']]
    elos = [results['elo_a'], results['elo_b']]
    
    ax2.bar(models, elos, color=['#5DA5DA', '#FAA43A'])
    ax2.set_title('ELO Ratings')
    ax2.set_ylabel('ELO')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add ELO values as text
    for i, v in enumerate(elos):
        ax2.text(i, v + 5, f"{v:.1f}", ha='center')
    
    # Add overall title
    plt.suptitle(f"Evaluation Results: {results['game_type'].capitalize()} (Board Size: {results['board_size']})")
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f"{results['game_type']}_{results['model_a_short']}_vs_{results['model_b_short']}_results.png")
    plt.savefig(plot_file)
    print(f"Results plot saved to {plot_file}")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)