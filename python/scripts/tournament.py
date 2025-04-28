#!/usr/bin/env python3
"""
Tournament script for AlphaZero models.

This script runs a round-robin tournament between multiple AlphaZero models
to assess relative strengths and calculate ELO ratings.

Usage:
    python tournament.py [options]

Options:
    --models DIR            Directory containing model files (required)
    --game {gomoku,chess,go}  Game type (default: gomoku)
    --size SIZE             Board size (default: depends on game)
    --games NUM             Number of games per pairing (default: 10)
    --simulations SIMS      Number of MCTS simulations per move (default: 800)
    --threads THREADS       Number of threads (default: 4)
    --output-dir DIR        Output directory for results (default: tournament_results)
    --temperature TEMP      Temperature for move selection (default: 0.1)
    --time-control SEC      Time per move in seconds (default: none)
    --variant               Use variant rules (Renju, Chess960, Chinese)
    --elo-k K               K-factor for ELO calculations (default: 32)
    --initial-elo ELO       Initial ELO rating (default: 1500)
    --include-random        Include a random player in the tournament
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pyalphazero as az
from alphazero.utils.elo import EloRating

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Tournament")
    parser.add_argument("--models", type=str, required=True,
                        help="Directory containing model files")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games per pairing")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads")
    parser.add_argument("--output-dir", type=str, default="tournament_results",
                        help="Output directory for results")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for move selection")
    parser.add_argument("--time-control", type=float, default=None,
                        help="Time per move in seconds (overrides simulations)")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--elo-k", type=float, default=32.0,
                        help="K-factor for ELO calculations")
    parser.add_argument("--initial-elo", type=float, default=1500.0,
                        help="Initial ELO rating")
    parser.add_argument("--include-random", action="store_true",
                        help="Include a random player in the tournament")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID to use (for multi-GPU systems)")
    return parser.parse_args()


def create_neural_network(model_path, game_type, board_size, gpu_id=0):
    """Create a neural network wrapper for the AI."""
    if not model_path:
        print("Using random policy")
        return None
    
    try:
        # Try to load with the C++ API first
        nn = az.createNeuralNetwork(model_path, game_type, board_size, True)
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
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{gpu_id}")
            else:
                device = torch.device("cpu")
                
            # Import model class
            from alphazero.models import DDWRandWireResNet
            
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
            
            nn = TorchNeuralNetwork(model, device)
            print(f"Loaded model from {model_path} (PyTorch)")
            return nn
        except Exception as e:
            print(f"Failed to load model with PyTorch: {e}")
            print("Using random policy network instead")
            return None


def play_game(game_state, mcts_a, mcts_b, args, game_num=0, start_player_a=True, model_a_name="", model_b_name=""):
    """Play a single game between two MCTS instances."""
    history = []
    move_times = []
    current_player_a = start_player_a
    
    # For time-based control
    time_per_move = args.time_control
    
    # Track game progress
    move_num = 0
    
    while not game_state.isTerminal():
        start_time = time.time()
        
        # Select the MCTS for the current player
        mcts = mcts_a if current_player_a else mcts_b
        current_model = model_a_name if current_player_a else model_b_name
        
        # Print current move info
        print(f"\rGame {game_num+1}, Move {move_num+1}: {current_model} thinking...", end="")
        
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
            'model': model_a_name if current_player_a else model_b_name,
            'move_time': move_time,
            'game_player': game_state.getCurrentPlayer(),
            'move_num': move_num
        })
        
        # Make move
        game_state.makeMove(action)
        
        # Update MCTS trees
        mcts_a.updateWithMove(action)
        mcts_b.updateWithMove(action)
        
        # Switch player
        current_player_a = not current_player_a
        move_num += 1
    
    print()  # New line after the game finishes
    
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
            print(f"Game {game_num+1}: {model_a_name} wins as Player 1")
        else:
            # B started as Player 1
            score_b = 1
            print(f"Game {game_num+1}: {model_b_name} wins as Player 1")
    elif result == az.GameResult.WIN_PLAYER2:
        # Player 2 (White/Black in Chess) won
        if start_player_a:
            # A started as Player 1, so B won
            score_b = 1
            print(f"Game {game_num+1}: {model_b_name} wins as Player 2")
        else:
            # B started as Player 1, so A won
            score_a = 1
            print(f"Game {game_num+1}: {model_a_name} wins as Player 2")
    else:
        # Draw
        score_a = 0.5
        score_b = 0.5
        print(f"Game {game_num+1}: Draw between {model_a_name} and {model_b_name}")
    
    # Prepare game record
    game_record = {
        'moves': history,
        'result': str(result),
        'score_a': score_a,
        'score_b': score_b,
        'model_a': model_a_name,
        'model_b': model_b_name,
        'start_player_a': start_player_a,
        'move_times': move_times,
        'avg_move_time': sum(move_times) / len(move_times) if move_times else 0,
        'total_moves': len(history)
    }
    
    return game_record


def play_match(model_a_path, model_b_path, args, game_num_offset=0):
    """Play a match between two models."""
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
            board_size = 9  # Default to 9x9 for tournament (faster than 19x19)
        else:
            board_size = 15
    else:
        board_size = args.size
    
    # Get model names
    model_a_name = os.path.basename(model_a_path) if model_a_path else "Random"
    model_b_name = os.path.basename(model_b_path) if model_b_path else "Random"
    
    print(f"\nPlaying match: {model_a_name} vs {model_b_name}")
    print(f"Games: {args.games}, Simulations: {args.simulations}, Threads: {args.threads}")
    
    # Load the neural networks
    nn_a = create_neural_network(model_a_path, game_type, board_size, args.gpu_id)
    nn_b = create_neural_network(model_b_path, game_type, board_size, args.gpu_id)
    
    # Create transposition tables (separate for each model)
    tt_a = az.TranspositionTable(1048576, 1024)
    tt_b = az.TranspositionTable(1048576, 1024)
    
    # Game results
    wins_a = 0
    wins_b = 0
    draws = 0
    game_records = []
    
    # Run games
    for game_num in range(args.games):
        # Determine starting player for this game
        # Alternate who goes first to ensure fairness
        start_player_a = (game_num % 2 == 0)
        
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
        game_record = play_game(
            game_state, mcts_a, mcts_b, args, 
            game_num + game_num_offset, start_player_a,
            model_a_name, model_b_name
        )
        
        # Update statistics
        if game_record['score_a'] == 1:
            wins_a += 1
        elif game_record['score_b'] == 1:
            wins_b += 1
        else:
            draws += 1
        
        # Store game record
        game_records.append(game_record)
    
    # Match results
    match_results = {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'model_a_path': model_a_path,
        'model_b_path': model_b_path,
        'games_played': args.games,
        'wins_a': wins_a,
        'wins_b': wins_b,
        'draws': draws,
        'score_a': wins_a + 0.5 * draws,
        'score_b': wins_b + 0.5 * draws,
        'game_records': game_records
    }
    
    return match_results


def run_tournament(args):
    """Run a round-robin tournament between multiple models."""
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find model files in the specified directory
    model_files = []
    for file in os.listdir(args.models):
        if file.endswith(".pt") or file.endswith(".pth"):
            model_files.append(os.path.join(args.models, file))
    
    if not model_files:
        print(f"No model files found in {args.models}")
        return
    
    print(f"Found {len(model_files)} model files:")
    for model in model_files:
        print(f"  {os.path.basename(model)}")
    
    # Add a random player if requested
    player_names = [os.path.basename(model) for model in model_files]
    if args.include_random:
        model_files.append("")  # Empty string represents random player
        player_names.append("Random")
    
    # Create a tournament schedule
    matches = []
    for i, model_a in enumerate(model_files):
        for j, model_b in enumerate(model_files):
            if i < j:  # Only play each pairing once
                matches.append((model_a, model_b))
    
    print(f"\nTournament will consist of {len(matches)} matches with {args.games} games each")
    print(f"Total games to be played: {len(matches) * args.games}")
    
    # Initialize ELO ratings
    elo_tracker = EloRating(initial_rating=args.initial_elo, k_factor=args.elo_k)
    
    # Results storage
    all_results = []
    game_num_offset = 0
    
    # Play all matches
    for match_num, (model_a, model_b) in enumerate(matches):
        print(f"\nMatch {match_num+1}/{len(matches)}")
        
        # Play the match
        match_results = play_match(model_a, model_b, args, game_num_offset)
        game_num_offset += args.games
        
        # Store results
        all_results.append(match_results)
        
        # Update ELO ratings
        model_a_name = match_results['model_a']
        model_b_name = match_results['model_b']
        
        # Calculate expected score
        expected_a = elo_tracker.get_expected_score(model_a_name, model_b_name)
        expected_b = 1.0 - expected_a
        
        # Update ratings
        actual_a = match_results['score_a'] / args.games
        elo_tracker.add_match_result(model_a_name, model_b_name, actual_a)
        
        # Print current match results
        print(f"\nMatch results: {model_a_name} vs {model_b_name}")
        print(f"  Score: {match_results['score_a']}-{match_results['score_b']}")
        print(f"  Win rate: {match_results['score_a']/args.games*100:.1f}%")
        print(f"  Expected score: {expected_a*100:.1f}%")
        
        # Save incremental results
        save_tournament_results(all_results, elo_tracker, args)
    
    # Final results
    return all_results, elo_tracker


def save_tournament_results(results, elo_tracker, args):
    """Save tournament results to file."""
    # Game type and board size
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
            board_size = 9
        else:
            board_size = 15
    else:
        board_size = args.size
    
    # Prepare summary data
    summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'game_type': args.game,
        'board_size': board_size,
        'games_per_match': args.games,
        'simulations': args.simulations,
        'threads': args.threads,
        'variant_rules': args.variant,
        'time_control': args.time_control,
        'temperature': args.temperature,
        'elo_settings': {
            'k_factor': args.elo_k,
            'initial_rating': args.initial_elo
        },
        'matches': results,
        'elo_ratings': {}
    }
    
    # Add ELO ratings
    players = set()
    for match in results:
        players.add(match['model_a'])
        players.add(match['model_b'])
    
    # Get final ELO ratings
    for player in players:
        summary['elo_ratings'][player] = elo_tracker.get_rating(player)
    
    # Add player ranking
    ranked_players = sorted(
        [(player, summary['elo_ratings'][player]) for player in players],
        key=lambda x: x[1],
        reverse=True
    )
    summary['player_ranking'] = [
        {'player': player, 'elo': elo, 'rank': i+1}
        for i, (player, elo) in enumerate(ranked_players)
    ]
    
    # Save to file
    results_file = os.path.join(args.output_dir, f"tournament_results_{int(time.time())}.json")
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate summary visualization
    generate_tournament_visuals(summary, args.output_dir)
    
    return summary


def generate_tournament_visuals(summary, output_dir):
    """Generate visualizations of tournament results."""
    # 1. ELO ratings bar chart
    plt.figure(figsize=(12, 6))
    
    # Sort players by ELO
    players = [(p['player'], p['elo']) for p in summary['player_ranking']]
    players.sort(key=lambda x: x[1], reverse=True)
    
    # Plot ELO ratings
    plt.barh([p[0] for p in players], [p[1] for p in players], color='skyblue')
    plt.xlabel('ELO Rating')
    plt.ylabel('Player')
    plt.title('Tournament Results: ELO Ratings')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add ELO values as text
    for i, (player, elo) in enumerate(players):
        plt.text(elo + 5, i, f"{elo:.1f}", va='center')
    
    plt.tight_layout()
    elo_chart_file = os.path.join(output_dir, f"tournament_elo_ratings.png")
    plt.savefig(elo_chart_file, dpi=300)
    print(f"ELO ratings chart saved to {elo_chart_file}")
    
    # 2. Match results matrix
    player_names = [p[0] for p in players]
    n_players = len(player_names)
    
    # Create matrix of results
    results_matrix = np.zeros((n_players, n_players))
    games_matrix = np.zeros((n_players, n_players))
    
    for match in summary['matches']:
        model_a = match['model_a']
        model_b = match['model_b']
        
        idx_a = player_names.index(model_a)
        idx_b = player_names.index(model_b)
        
        # Fill in the matrix (store win percentages)
        score_a = match['score_a']
        total_games = match['games_played']
        win_rate_a = score_a / total_games
        
        results_matrix[idx_a, idx_b] = win_rate_a
        results_matrix[idx_b, idx_a] = 1 - win_rate_a
        
        games_matrix[idx_a, idx_b] = total_games
        games_matrix[idx_b, idx_a] = total_games
    
    # Plot the matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(results_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Win Rate')
    
    # Add text annotations
    for i in range(n_players):
        for j in range(n_players):
            if i != j and games_matrix[i, j] > 0:
                plt.text(j, i, f"{results_matrix[i, j]:.2f}",
                         ha="center", va="center",
                         color="black" if 0.3 < results_matrix[i, j] < 0.7 else "white")
            elif i == j:
                plt.text(j, i, "X", ha="center", va="center")
    
    # Set labels
    plt.xticks(range(n_players), player_names, rotation=45, ha="right")
    plt.yticks(range(n_players), player_names)
    plt.title('Tournament Results: Win Rates Matrix')
    
    plt.tight_layout()
    matrix_file = os.path.join(output_dir, f"tournament_results_matrix.png")
    plt.savefig(matrix_file, dpi=300)
    print(f"Results matrix saved to {matrix_file}")


if __name__ == "__main__":
    args = parse_args()
    results, elo_tracker = run_tournament(args)
    
    # Print final rankings
    print("\nFinal Tournament Rankings:")
    print("=========================")
    
    # Get all players
    players = set()
    for match in results:
        players.add(match['model_a'])
        players.add(match['model_b'])
    
    # Sort by ELO rating
    ranked_players = sorted(
        [(player, elo_tracker.get_rating(player)) for player in players],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Print rankings
    for i, (player, elo) in enumerate(ranked_players):
        print(f"{i+1}. {player}: {elo:.1f} ELO")