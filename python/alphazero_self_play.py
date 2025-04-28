#!/usr/bin/env python3
"""
AlphaZero Self-Play Data Generation Script

This script runs the C++ AlphaZero implementation to generate self-play game data
for training the neural network.
"""

import os
import subprocess
import argparse
import time
import json
import random
import shutil
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

def run_self_play_game(args: Dict[str, Any], game_id: int) -> str:
    """Run a single self-play game using the C++ executable"""
    # Create output path for this game
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args["output_dir"], f"game_{game_id:05d}_{timestamp}.json")
    
    # Build command
    cmd = [
        args["executable_path"],
        "--game", "gomoku",
        "--board-size", str(args["board_size"]),
        "--simulations", str(args["simulations"]),
        "--threads", str(args["threads"]),
        "--selfplay",
        "--output", output_path
    ]
    
    if args["model_path"]:
        cmd.extend(["--model", args["model_path"]])
    
    if args["use_gpu"]:
        cmd.append("--use-gpu")
    
    if args["use_renju"]:
        cmd.append("--use-renju")
    
    # Set temperature parameters
    cmd.extend([
        "--temperature-init", str(args["temp_init"]),
        "--temperature-final", str(args["temp_final"]),
        "--temperature-threshold", str(args["temp_threshold"])
    ])
    
    # Run the command
    try:
        start_time = time.time()
        print(f"Starting game {game_id}...")
        
        # Run subprocess with output capture
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if output file was created
        if os.path.exists(output_path):
            print(f"Game {game_id} completed in {duration:.1f}s")
            return output_path
        else:
            print(f"Game {game_id} failed: Output file not created")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Stdout: {result.stdout}")
            print(f"  Stderr: {result.stderr}")
            return ""
        
    except subprocess.CalledProcessError as e:
        print(f"Game {game_id} failed with error code {e.returncode}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return ""

def collect_game_stats(output_paths: List[str]) -> Dict[str, Any]:
    """Collect statistics from game records"""
    stats = {
        "total_games": len(output_paths),
        "valid_games": 0,
        "total_moves": 0,
        "min_moves": float('inf'),
        "max_moves": 0,
        "avg_moves": 0,
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0
    }
    
    for path in output_paths:
        if not path:
            continue
        
        try:
            with open(path, 'r') as f:
                game_data = json.load(f)
            
            stats["valid_games"] += 1
            moves = len(game_data["moves"])
            stats["total_moves"] += moves
            stats["min_moves"] = min(stats["min_moves"], moves)
            stats["max_moves"] = max(stats["max_moves"], moves)
            
            if game_data["result"] == 1:
                stats["player1_wins"] += 1
            elif game_data["result"] == 2:
                stats["player2_wins"] += 1
            else:
                stats["draws"] += 1
        
        except Exception as e:
            print(f"Error reading game file {path}: {e}")
    
    if stats["valid_games"] > 0:
        stats["avg_moves"] = stats["total_moves"] / stats["valid_games"]
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Data Generation Script")
    parser.add_argument("--executable", type=str, default="../build/bin/alphazero_cli",
                       help="Path to AlphaZero executable")
    parser.add_argument("--output-dir", type=str, default="self_play_games",
                       help="Output directory for game records")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of games to generate")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--model", type=str, default="",
                       help="Path to neural network model")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU for neural network inference")
    parser.add_argument("--board-size", type=int, default=15,
                       help="Board size")
    parser.add_argument("--simulations", type=int, default=800,
                       help="Number of MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=2,
                       help="Number of threads per game")
    parser.add_argument("--use-renju", action="store_true",
                       help="Use Renju rules")
    parser.add_argument("--temp-init", type=float, default=1.0,
                       help="Initial temperature")
    parser.add_argument("--temp-final", type=float, default=0.1,
                       help="Final temperature")
    parser.add_argument("--temp-threshold", type=int, default=30,
                       help="Move threshold for temperature change")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if executable exists
    if not os.path.exists(args.executable):
        print(f"Error: Executable not found at {args.executable}")
        return
    
    # Prepare arguments for self-play games
    run_args = {
        "executable_path": args.executable,
        "output_dir": args.output_dir,
        "board_size": args.board_size,
        "simulations": args.simulations,
        "threads": args.threads,
        "model_path": args.model,
        "use_gpu": args.use_gpu,
        "use_renju": args.use_renju,
        "temp_init": args.temp_init,
        "temp_final": args.temp_final,
        "temp_threshold": args.temp_threshold
    }
    
    # Run self-play games in parallel
    output_paths = []
    
    print(f"Generating {args.games} self-play games using {args.workers} workers")
    print(f"  Model: {args.model if args.model else 'Random policy (no model)'}")
    print(f"  Board size: {args.board_size}")
    print(f"  Simulations: {args.simulations}")
    print(f"  Threads per game: {args.threads}")
    print(f"  Rules: {'Renju' if args.use_renju else 'Standard Gomoku'}")
    print(f"  Temperature: {args.temp_init} -> {args.temp_final} at move {args.temp_threshold}")
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_self_play_game, run_args, i) 
                  for i in range(args.games)]
        
        for future in futures:
            output_path = future.result()
            output_paths.append(output_path)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Collect statistics
    stats = collect_game_stats(output_paths)
    
    print("\nSelf-play completed")
    print(f"  Total time: {total_duration:.1f}s")
    print(f"  Games per second: {stats['valid_games'] / total_duration:.2f}")
    print(f"  Valid games: {stats['valid_games']} / {stats['total_games']}")
    print(f"  Results: Black wins: {stats['player1_wins']}, White wins: {stats['player2_wins']}, Draws: {stats['draws']}")
    print(f"  Moves per game: Min: {stats['min_moves']}, Max: {stats['max_moves']}, Avg: {stats['avg_moves']:.1f}")
    
    # Save statistics to a JSON file
    stats_path = os.path.join(args.output_dir, "self_play_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    main()