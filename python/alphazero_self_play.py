#!/usr/bin/env python3
"""
AlphaZero Self-Play Data Generation Script

This script generates self-play games for training the AlphaZero neural network.
It uses the C++ AlphaZero implementation for game simulation and MCTS.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import multiprocessing
import glob
from datetime import datetime
from tqdm import tqdm

def run_self_play_process(args):
    """Run a single self-play process using the C++ executable"""
    process_id, executable, output_dir, game, board_size, simulations, threads, model, use_gpu, use_renju = args
    
    # Create process-specific output directory
    process_dir = os.path.join(output_dir, f"process_{process_id}")
    os.makedirs(process_dir, exist_ok=True)
    
    # Construct command
    cmd = [
        executable,
        "--game", game,
        "--board-size", str(board_size),
        "--simulations", str(simulations),
        "--threads", str(threads),
        "--selfplay",
        "--output", os.path.join(process_dir, f"game_{{ }}.json"),
        "--temperature-init", "1.0",
        "--temperature-final", "0.25",
        "--temperature-threshold", "30"
    ]
    
    # Add model if provided
    if model:
        cmd.extend(["--model", model])
    
    # Add GPU option if requested
    if use_gpu:
        cmd.append("--use-gpu")
    
    # Add Renju rules option if requested
    if use_renju:
        cmd.append("--use-renju")
    
    # Run the command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for completion with timeout
        stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout
        
        if process.returncode != 0:
            print(f"Process {process_id} failed with code {process.returncode}")
            print(f"stderr: {stderr}")
            return False
        
        return True
    except subprocess.TimeoutExpired:
        print(f"Process {process_id} timed out and will be killed")
        process.kill()
        process.wait()
        return False
    except Exception as e:
        print(f"Process {process_id} failed with exception: {e}")
        return False

def collect_games(output_dir, final_dir):
    """Collect all generated games into a single directory"""
    os.makedirs(final_dir, exist_ok=True)
    
    # Find all process directories
    process_dirs = glob.glob(os.path.join(output_dir, "process_*"))
    
    # Collect and rename games
    total_games = 0
    for process_dir in process_dirs:
        games = glob.glob(os.path.join(process_dir, "*.json"))
        for game_file in games:
            # Extract timestamp from the JSON file
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                    timestamp = game_data.get('timestamp', datetime.now().strftime("%Y%m%dT%H%M%SZ"))
                    game_id = total_games
                    
                    # Create a new filename
                    new_filename = f"game_{game_id:04d}_{timestamp}.json"
                    new_filepath = os.path.join(final_dir, new_filename)
                    
                    # Copy the file
                    with open(new_filepath, 'w') as out_f:
                        json.dump(game_data, out_f)
                    
                    total_games += 1
            except Exception as e:
                print(f"Error processing {game_file}: {e}")
    
    print(f"Collected {total_games} games in {final_dir}")
    return total_games

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Data Generation")
    parser.add_argument("--executable", required=True, help="Path to AlphaZero CLI executable")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated games")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--game", default="gomoku", choices=["gomoku", "chess", "go"], help="Game type")
    parser.add_argument("--board-size", type=int, default=15, help="Board size")
    parser.add_argument("--simulations", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=2, help="Threads per game")
    parser.add_argument("--model", help="Path to neural network model")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for neural network")
    parser.add_argument("--use-renju", action="store_true", help="Use Renju rules for Gomoku")
    args = parser.parse_args()
    
    # Check if executable exists
    if not os.path.isfile(args.executable):
        print(f"Error: Executable not found at {args.executable}")
        return 1
    
    # Create temp output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(args.output_dir, f"temp_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create final output directory
    final_dir = os.path.join(args.output_dir, f"games_{args.game}_{timestamp}")
    os.makedirs(final_dir, exist_ok=True)
    
    # Determine number of games per worker
    num_workers = min(args.workers, args.games)
    games_per_worker = [args.games // num_workers] * num_workers
    # Distribute remaining games
    for i in range(args.games % num_workers):
        games_per_worker[i] += 1
    
    print(f"Generating {args.games} {args.game} games with {num_workers} workers")
    print(f"Game distribution: {games_per_worker}")
    
    # Create process pool
    with multiprocessing.Pool(num_workers) as pool:
        # Prepare arguments for each worker
        worker_args = []
        for i in range(num_workers):
            worker_args.append((
                i,
                args.executable,
                temp_dir,
                args.game,
                args.board_size,
                args.simulations,
                args.threads,
                args.model,
                args.use_gpu,
                args.use_renju
            ))
        
        # Run workers
        results = list(tqdm(
            pool.imap_unordered(run_self_play_process, worker_args),
            total=num_workers,
            desc="Self-play workers"
        ))
    
    # Collect games into final directory
    print("Collecting games...")
    num_collected = collect_games(temp_dir, final_dir)
    
    # Check if we got the expected number of games
    if num_collected < args.games:
        print(f"Warning: Expected {args.games} games, but collected only {num_collected}")
    
    # Create summary file
    summary = {
        "date": datetime.now().isoformat(),
        "game": args.game,
        "board_size": args.board_size,
        "num_games": num_collected,
        "simulations": args.simulations,
        "threads": args.threads,
        "model": args.model,
        "use_gpu": args.use_gpu,
        "use_renju": args.use_renju
    }
    
    with open(os.path.join(final_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Self-play completed. Generated {num_collected} games.")
    print(f"Games saved to: {final_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())