#!/usr/bin/env python3
"""
AlphaZero Self-Play Orchestrator

This script orchestrates self-play using the optimized C++ implementation while providing
a Python interface for management and monitoring. It handles:

1. Loading/creating models
2. Converting models to optimized TorchScript format
3. Running C++ self-play processes
4. Monitoring performance and resource utilization
5. Collecting and organizing game records

Usage:
    python orchestrate_selfplay.py [options]

Options:
    --model MODEL               Path to model file (PyTorch, LibTorch, or TorchScript)
    --game {gomoku,chess,go}    Game type (default: gomoku)
    --size SIZE                 Board size (default: depends on game)
    --num-games NUM             Number of games to generate (default: 100)
    --simulations SIMS          Number of MCTS simulations per move (default: 800)
    --threads THREADS           Number of threads (default: auto-detect)
    --processes PROCS           Number of separate C++ processes (default: 1)
    --output-dir DIR            Output directory (default: data/games)
    --batch-size SIZE           Batch size for neural network inference (default: 16)
    --batch-timeout MS          Timeout for batch completion in milliseconds (default: 10)
    --temperature TEMP          Initial temperature (default: 1.0)
    --temp-drop MOVE            Move to drop temperature (default: 30)
    --final-temp TEMP           Final temperature (default: 0.0)
    --dirichlet-alpha A         Dirichlet noise alpha (default: 0.03)
    --dirichlet-epsilon E       Dirichlet noise weight (default: 0.25)
    --variant                   Use variant rules (Renju, Chess960, Chinese)
    --no-gpu                    Disable GPU acceleration
    --fp16                      Use FP16 precision (faster but less accurate)
    --monitor-interval SEC      Interval in seconds between monitoring updates (default: 5)
    --create-random-model       Create and use a random model if none is provided
    --use-cpp-binary            Use the C++ binary instead of Python wrapper (fastest)
    --profile                   Enable performance profiling
"""

import os
import sys
import argparse
import time
import json
import random
import numpy as np
import subprocess
import threading
import signal
import psutil
import datetime
import glob
import torch
import re
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get build directory for the C++ extension
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '../..'))
build_dir = os.path.join(project_dir, 'build', 'src', 'pybind')
cpp_binary_path = os.path.join(project_dir, 'build', 'src', 'selfplay', 'self_play')

# Add build directory to path
sys.path.insert(0, build_dir)

# Attempt to import required modules
try:
    import _alphazero_cpp as az
    from alphazero.models import DDWRandWireResNet, DDWRandWireResNetWrapper, create_model, export_to_torchscript
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the project is properly built and the Python package is installed.")
    sys.exit(1)

# Global variables for monitoring
processes = []
running = True
total_games_completed = 0
total_moves_generated = 0
start_time = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Orchestrator")
    
    parser.add_argument("--model", type=str, default="",
                        help="Path to model file (PyTorch, LibTorch, or TorchScript)")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to generate")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--threads", type=int, default=0,
                        help="Number of threads per process (0 for auto-detect)")
    parser.add_argument("--processes", type=int, default=1,
                        help="Number of separate C++ processes")
    parser.add_argument("--output-dir", type=str, default="data/games",
                        help="Output directory for game records")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for neural network inference")
    parser.add_argument("--batch-timeout", type=int, default=10,
                        help="Timeout for batch completion in milliseconds")
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
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision (faster but less accurate)")
    parser.add_argument("--monitor-interval", type=int, default=5,
                        help="Interval in seconds between monitoring updates")
    parser.add_argument("--create-random-model", action="store_true",
                        help="Create and use a random model if none is provided")
    parser.add_argument("--use-cpp-binary", action="store_true",
                        help="Use the C++ binary instead of Python wrapper (fastest)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling")
    
    # Advanced MCTS options
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="Exploration constant")
    parser.add_argument("--fpu-reduction", type=float, default=0.1,
                        help="First play urgency reduction")
    parser.add_argument("--virtual-loss", type=int, default=3,
                        help="Virtual loss amount")
    parser.add_argument("--no-tt", action="store_true",
                        help="Disable transposition table in MCTS")
    parser.add_argument("--progressive-widening", action="store_true",
                        help="Use progressive widening in MCTS")
    
    # Performance optimization options
    parser.add_argument("--optimize-batch", action="store_true",
                        help="Dynamically optimize batch parameters")
    parser.add_argument("--cache-size", type=int, default=2097152,
                        help="Size of transposition table cache (default: 2M entries)")
    parser.add_argument("--optimize-threads", action="store_true",
                        help="Dynamically optimize thread allocation")
    parser.add_argument("--compact-size", type=int, default=0,
                        help="Compact board representation size (0 for no compaction)")
    parser.add_argument("--pin-threads", action="store_true",
                        help="Pin threads to CPU cores")
    
    return parser.parse_args()

def get_default_board_size(game_type):
    """Get default board size for a game."""
    if game_type == "chess":
        return 8
    elif game_type == "go":
        return 19
    else:  # gomoku
        return 15

def get_input_channels(game_type):
    """Get number of input channels for a game."""
    if game_type == "chess":
        return 14  # 6 piece types x 2 colors + auxiliary channels
    elif game_type == "go":
        return 8   # Current player stones, opponent stones, history, and auxiliary channels
    else:  # gomoku
        return 8   # Current player stones, opponent stones, history, and auxiliary channels

def get_output_size(game_type, board_size):
    """Get output size (policy dimension) for a game."""
    if game_type == "chess":
        return 64 * 73  # 64 squares, 73 possible moves per square (max)
    elif game_type == "go":
        return board_size * board_size + 1  # +1 for pass move
    else:  # gomoku
        return board_size * board_size

def create_random_model(game_type, board_size):
    """Create a random model for testing."""
    print("Creating a random DDWRandWireResNet model...")
    
    input_channels = get_input_channels(game_type)
    output_size = get_output_size(game_type, board_size)
    
    # Create a random C++ model with DDWRandWireResNetWrapper
    model = create_model(input_channels, output_size, channels=64, num_blocks=8)
    
    # Create output directories if needed
    os.makedirs("models", exist_ok=True)
    game_dir = os.path.join("models", game_type)
    os.makedirs(game_dir, exist_ok=True)
    
    # Generate a filename
    model_path = os.path.join(game_dir, f"random_model_{game_type}_{board_size}x{board_size}.pt")
    ts_model_path = os.path.join(game_dir, f"random_model_{game_type}_{board_size}x{board_size}.torchscript.pt")
    
    # Save both the model and its TorchScript version
    model.save(model_path)
    model.export_to_torchscript(ts_model_path, [1, input_channels, board_size, board_size])
    
    print(f"Random model saved to {model_path}")
    print(f"TorchScript model saved to {ts_model_path}")
    
    return ts_model_path

def prepare_model(args, game_type, board_size):
    """Prepare the model for self-play, converting to TorchScript if needed."""
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    
    # If no model provided, create a random one if requested
    if not args.model:
        if args.create_random_model:
            return create_random_model(game_type, board_size)
        else:
            print("No model provided. Using random play.")
            return ""
    
    model_path = args.model
    
    # Check if it's already a TorchScript model (faster check first)
    if model_path.endswith('.torchscript.pt') or model_path.endswith('.pt'):
        try:
            # Try to load as TorchScript model
            torch.jit.load(model_path)
            print(f"Model {model_path} is already in TorchScript format")
            return model_path
        except Exception:
            # Not a TorchScript model, convert it
            pass
    
    # Convert to TorchScript format
    print(f"Converting model {model_path} to TorchScript format for C++ inference")
    
    # Create output path
    base_name = os.path.basename(model_path)
    dir_name = os.path.dirname(model_path)
    
    # Add torchscript suffix if not present
    if not base_name.endswith('.torchscript.pt'):
        ts_filename = os.path.splitext(base_name)[0] + '.torchscript.pt'
    else:
        ts_filename = base_name
    
    output_path = os.path.join(dir_name, ts_filename)
    
    try:
        # Get model parameters
        input_channels = get_input_channels(game_type)
        output_size = get_output_size(game_type, board_size)
        
        # Try to load with PyTorch
        device = torch.device("cuda" if use_gpu else "cpu")
        
        try:
            # Try as Python DDWRandWireResNet
            model = DDWRandWireResNet(input_channels, output_size)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device).eval()
            
            # Create input tensor for tracing
            dummy_input = torch.zeros((1, input_channels, board_size, board_size), device=device)
            
            # Trace the model
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Save the traced model
            traced_model.save(output_path)
            print(f"Converted PyTorch model to TorchScript format: {output_path}")
        except Exception as e:
            print(f"Failed to load as Python model: {e}")
            
            # Try as C++ DDWRandWireResNetWrapper
            try:
                model = DDWRandWireResNetWrapper(input_channels, output_size)
                model.load(model_path)
                model.export_to_torchscript(output_path, [1, input_channels, board_size, board_size])
                print(f"Converted C++ model to TorchScript format: {output_path}")
            except Exception as e2:
                print(f"Failed to load as C++ model: {e2}")
                print("Using original model path, but inference may be slower")
                return model_path
        
        return output_path
    
    except Exception as e:
        print(f"Error converting model: {e}")
        print("Using original model path, but inference may be slower")
        return model_path

def start_cpp_process(args, proc_idx):
    """Start a C++ self-play process using the standalone binary."""
    global total_games_completed, total_moves_generated
    
    # Calculate number of games for this process
    games_per_process = max(1, args.num_games // args.processes)
    if proc_idx == args.processes - 1:
        # Last process gets any remaining games
        games_per_process += args.num_games % args.processes
    
    # Create a unique output directory for this process
    proc_output_dir = os.path.join(args.output_dir, f"proc_{proc_idx}")
    os.makedirs(proc_output_dir, exist_ok=True)
    
    # Adjust batch timeout based on batch size for better utilization
    # Larger batch sizes need more time to fill
    adjusted_batch_timeout = args.batch_timeout
    if args.optimize_batch:
        adjusted_batch_timeout = max(20, min(100, args.batch_size // 2))
    
    # Optimize thread allocation if requested
    adjusted_threads = args.threads
    if args.optimize_threads:
        # Adjust threads based on the processor count and batch size
        cpu_count = psutil.cpu_count(logical=False)
        if args.batch_size <= 8:
            # For small batches, use more threads for parallelism
            adjusted_threads = max(1, cpu_count - 1)
        elif args.batch_size <= 32:
            # For medium batches, balance threads
            adjusted_threads = max(1, cpu_count // 2)
        else:
            # For large batches, fewer threads to avoid contention
            adjusted_threads = max(1, cpu_count // 4)
    
    # Optimize cache size
    cache_size = args.cache_size
    if args.no_tt:
        cache_size = 1  # Effectively disable TT
    
    # Build command line arguments
    cmd = [
        cpp_binary_path,
        f"--game={args.game}",
        f"--size={args.size}",
        f"--num-games={games_per_process}",
        f"--simulations={args.simulations}",
        f"--threads={adjusted_threads}",
        f"--output-dir={proc_output_dir}",
        f"--batch-size={args.batch_size}",
        f"--batch-timeout={adjusted_batch_timeout}",
        f"--temperature={args.temperature}",
        f"--temp-drop={args.temp_drop}",
        f"--final-temp={args.final_temp}",
        f"--dirichlet-alpha={args.dirichlet_alpha}",
        f"--dirichlet-epsilon={args.dirichlet_epsilon}",
        f"--c-puct={args.c_puct}",
        f"--fpu-reduction={args.fpu_reduction}",
        f"--virtual-loss={args.virtual_loss}",
        f"--cache-size={cache_size}"
    ]
    
    # Add optional arguments
    if args.model:
        cmd.append(f"--model={args.model}")
    if args.variant:
        cmd.append("--variant")
    if args.no_gpu:
        cmd.append("--no-gpu")
    if args.fp16:
        cmd.append("--fp16")
    if args.no_tt:
        cmd.append("--no-tt")
    if args.progressive_widening:
        cmd.append("--progressive-widening")
    if args.pin_threads:
        cmd.append("--pin-threads")
    if args.compact_size > 0:
        cmd.append(f"--compact-size={args.compact_size}")
    # Set thread affinity for this process
    if args.pin_threads and args.processes > 1:
        start_core = proc_idx * (psutil.cpu_count(logical=False) // args.processes)
        end_core = (proc_idx + 1) * (psutil.cpu_count(logical=False) // args.processes) - 1
        cmd.append(f"--thread-affinity={start_core}-{end_core}")
    
    # Open log file
    log_path = os.path.join(proc_output_dir, "selfplay.log")
    log_file = open(log_path, "w")
    
    # Start the process
    print(f"Starting process {proc_idx} with {games_per_process} games, {adjusted_threads} threads, batch={args.batch_size}, timeout={adjusted_batch_timeout}ms")
    proc = subprocess.Popen(
        cmd, 
        stdout=log_file, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Thread function to monitor process completion
    def monitor_process():
        proc.wait()
        log_file.close()
        
        # Read log to update statistics
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
                
                # Extract completed games count
                games_match = re.search(r"Generated (\d+) games", log_content)
                if games_match:
                    games = int(games_match.group(1))
                    with statistics_lock:
                        global total_games_completed
                        total_games_completed += games
                
                # Extract total moves count
                moves_match = re.search(r"Total moves: (\d+)", log_content)
                if moves_match:
                    moves = int(moves_match.group(1))
                    with statistics_lock:
                        global total_moves_generated
                        total_moves_generated += moves
            
            # Move game files to main output directory
            for game_file in glob.glob(os.path.join(proc_output_dir, "*.json")):
                # Skip metadata files
                if "metadata_" in os.path.basename(game_file):
                    continue
                
                # Move game file
                dest_path = os.path.join(args.output_dir, os.path.basename(game_file))
                shutil.move(game_file, dest_path)
        except Exception as e:
            print(f"Error processing log file: {e}")
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_process)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return proc, monitor_thread, log_file

def start_py_cpp_selfplay(args, model_path):
    """Start self-play using Python wrapper for C++ code."""
    global total_games_completed, total_moves_generated
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get game type
    game_type_map = {
        "gomoku": az.GameType.GOMOKU,
        "chess": az.GameType.CHESS,
        "go": az.GameType.GO
    }
    game_type = game_type_map[args.game]
    
    # Use GPU if available
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    
    # Create neural network from model path
    if model_path:
        try:
            print(f"Loading model from {model_path}")
            neural_network = az.createNeuralNetwork(model_path, game_type, args.size, use_gpu)
            print(f"Model loaded: {neural_network.getDeviceInfo()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random policy network")
            neural_network = None
    else:
        print("No model specified. Using random policy network.")
        neural_network = None
    
    # Calculate games per process
    games_per_process = max(1, args.num_games // args.processes)
    processes_info = []
    
    for proc_idx in range(args.processes):
        # Last process gets any remaining games
        if proc_idx == args.processes - 1:
            proc_games = games_per_process + args.num_games % args.processes
        else:
            proc_games = games_per_process
        
        # Create a unique output directory for this process
        proc_output_dir = os.path.join(args.output_dir, f"proc_{proc_idx}")
        os.makedirs(proc_output_dir, exist_ok=True)
        
        # Thread function to run self-play for this process
        def run_selfplay_process(proc_idx, num_games, output_dir):
            # Create SelfPlayManager
            self_play = az.SelfPlayManager(
                neural_network,
                num_games,
                args.simulations,
                args.threads
            )
            
            # Configure batch settings
            self_play.setBatchConfig(args.batch_size, args.batch_timeout)
            
            # Set exploration parameters
            self_play.setExplorationParams(
                args.dirichlet_alpha,
                args.dirichlet_epsilon,
                args.temperature,
                args.temp_drop,
                args.final_temp
            )
            
            # Configure saving
            self_play.setSaveGames(True, output_dir)
            
            # Configure MCTS
            mcts_config = {
                'useBatchedMCTS': True,
                'batchSize': args.batch_size,
                'batchTimeoutMs': args.batch_timeout,
                'searchMode': 'BATCHED',
                'fpuReduction': args.fpu_reduction,
                'cPuct': args.c_puct,
                'virtualLoss': args.virtual_loss,
                'useFmapCache': not args.no_tt,
                'useProgressiveWidening': args.progressive_widening
            }
            self_play.setMctsConfig(mcts_config)
            
            # Progress callback
            last_update_time = time.time()
            
            def progress_callback(game_id, move_num, total_games, total_moves):
                nonlocal last_update_time
                now = time.time()
                
                # Update only once per second to reduce overhead
                if now - last_update_time > 1.0:
                    with statistics_lock:
                        global total_games_completed, total_moves_generated
                        total_games_completed = game_id
                        total_moves_generated = total_moves
                    last_update_time = now
            
            self_play.setProgressCallback(progress_callback)
            
            # Generate games
            print(f"Process {proc_idx}: Starting self-play with {num_games} games")
            start_time = time.time()
            games = self_play.generateGames(game_type, args.size, args.variant)
            end_time = time.time()
            
            # Print results
            duration = end_time - start_time
            print(f"Process {proc_idx}: Completed {len(games)} games in {duration:.2f} seconds")
            
            # Move game files to main output directory
            for game_file in glob.glob(os.path.join(output_dir, "*.json")):
                # Skip metadata files
                if "metadata_" in os.path.basename(game_file):
                    continue
                
                # Move game file
                dest_path = os.path.join(args.output_dir, os.path.basename(game_file))
                shutil.move(game_file, dest_path)
            
            print(f"Process {proc_idx}: Self-play completed")
        
        # Start thread for this process
        thread = threading.Thread(
            target=run_selfplay_process,
            args=(proc_idx, proc_games, proc_output_dir)
        )
        thread.daemon = True
        thread.start()
        
        processes_info.append((thread, proc_output_dir))
    
    return processes_info

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) to gracefully terminate processes."""
    global running
    print("Interrupt received, shutting down processes...")
    running = False
    
    # Terminate all child processes
    for proc, _, _ in processes:
        if proc.poll() is None:  # If process is still running
            try:
                proc.terminate()
            except:
                pass
    
    sys.exit(0)

def monitor_resources(args):
    """Monitor CPU, GPU, and memory usage."""
    global running, total_games_completed, total_moves_generated, start_time
    
    # Check if Nvidia GPU is available
    nvidia_smi_available = False
    try:
        subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'])
        nvidia_smi_available = True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Initialize statistics
    peak_cpu_usage = 0.0
    peak_gpu_usage = 0.0
    peak_memory_usage = 0
    
    while running:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Calculate statistics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        # Update peak values
        peak_cpu_usage = max(peak_cpu_usage, cpu_percent)
        peak_memory_usage = max(peak_memory_usage, memory_usage)
        
        # Get GPU stats if available
        gpu_percent = 0.0
        if nvidia_smi_available:
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
                    universal_newlines=True
                )
                gpu_percent = float(output.strip().split()[0])
                peak_gpu_usage = max(peak_gpu_usage, gpu_percent)
            except:
                pass
        
        # Calculate rates
        games_per_second = total_games_completed / elapsed_time if elapsed_time > 0 else 0
        moves_per_second = total_moves_generated / elapsed_time if elapsed_time > 0 else 0
        
        # Clear terminal line and print status
        print(f"\r{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"Games: {total_games_completed}/{args.num_games} | "
              f"Moves: {total_moves_generated} | "
              f"Rate: {games_per_second:.2f} games/s, {moves_per_second:.1f} moves/s | "
              f"CPU: {cpu_percent:.1f}% | "
              f"GPU: {gpu_percent:.1f}% | "
              f"Mem: {memory_usage:.0f} MB", end="")
        
        # Write to log file periodically
        if int(elapsed_time) % 60 == 0:
            with open(os.path.join(args.output_dir, "monitoring.log"), "a") as f:
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
                       f"{elapsed_time:.1f},{total_games_completed},{total_moves_generated},"
                       f"{games_per_second:.2f},{moves_per_second:.1f},"
                       f"{cpu_percent:.1f},{gpu_percent:.1f},{memory_usage:.0f}\n")
        
        # Check if all games are completed
        if total_games_completed >= args.num_games:
            print("\nAll games completed!")
            running = False
            break
        
        # Wait for next update
        time.sleep(args.monitor_interval)
    
    # Save final statistics
    try:
        metadata = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": elapsed_time,
            "games_completed": total_games_completed,
            "total_moves": total_moves_generated,
            "avg_moves_per_game": total_moves_generated / total_games_completed if total_games_completed > 0 else 0,
            "games_per_second": games_per_second,
            "moves_per_second": moves_per_second,
            "peak_cpu_usage_percent": peak_cpu_usage,
            "peak_gpu_usage_percent": peak_gpu_usage,
            "peak_memory_usage_mb": peak_memory_usage,
            "args": vars(args)
        }
        
        with open(os.path.join(args.output_dir, "final_statistics.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving final statistics: {e}")

def merge_game_files(args):
    """Merge game files from all processes into the main output directory."""
    print("Merging game files from all processes...")
    
    # Count files before starting
    initial_count = len(glob.glob(os.path.join(args.output_dir, "*.json")))
    proc_count = 0
    
    # Find all process directories
    for proc_dir in glob.glob(os.path.join(args.output_dir, "proc_*")):
        # Get all game files
        game_files = glob.glob(os.path.join(proc_dir, "*.json"))
        for game_file in game_files:
            # Skip metadata files
            if "metadata_" in os.path.basename(game_file):
                continue
            
            # Move game file to main output directory
            dest_path = os.path.join(args.output_dir, os.path.basename(game_file))
            shutil.move(game_file, dest_path)
            proc_count += 1
    
    # Count files after merging
    final_count = len(glob.glob(os.path.join(args.output_dir, "*.json")))
    
    print(f"Merged {proc_count} files from process directories")
    print(f"Final game count: {initial_count} (initial) + {proc_count} (processes) = {final_count}")

def main():
    global processes, running, start_time, statistics_lock
    
    # Parse command line arguments
    args = parse_args()
    
    # Set default board size if not specified
    if args.size <= 0:
        args.size = get_default_board_size(args.game)
    
    # Auto-detect thread count if not specified and not optimizing
    if args.threads <= 0 and not args.optimize_threads:
        args.threads = max(1, psutil.cpu_count(logical=False))
        
        # Adjust based on process count
        args.threads = max(1, args.threads // args.processes)
        
        print(f"Auto-detected {args.threads} threads per process")
    
    # Auto-optimize processes if not specified
    if args.processes == 1 and args.use_cpp_binary:
        gpu_available = torch.cuda.is_available() and not args.no_gpu
        
        if gpu_available:
            # With GPU, use fewer processes with more threads to optimize GPU utilization
            optimal_processes = max(1, torch.cuda.device_count())
            args.processes = optimal_processes
            if args.threads <= 0:
                args.threads = max(1, psutil.cpu_count(logical=False) // optimal_processes)
        else:
            # For CPU-only mode, use more processes with fewer threads
            optimal_processes = max(1, psutil.cpu_count(logical=False) // 2)
            args.processes = min(optimal_processes, 4)  # Limit to 4 processes to avoid oversubscription
            if args.threads <= 0:
                args.threads = max(1, psutil.cpu_count(logical=False) // optimal_processes)
        
        print(f"Auto-configured to use {args.processes} processes with {args.threads} threads each")

    # Optimize batch size if not specified
    if args.optimize_batch and args.use_cpp_binary:
        gpu_available = torch.cuda.is_available() and not args.no_gpu
        
        if gpu_available:
            # For GPU, use larger batches to maximize throughput
            # Scale batch size with model complexity and GPU memory
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            if args.game == "go":
                # Go has larger state representation
                args.batch_size = min(256, max(32, int(gpu_mem * 16)))
            elif args.game == "chess":
                # Chess has medium state complexity
                args.batch_size = min(256, max(32, int(gpu_mem * 24)))
            else:  # gomoku
                args.batch_size = min(256, max(32, int(gpu_mem * 32)))
            
            # Adjust batch timeout based on batch size
            args.batch_timeout = max(20, min(100, args.batch_size // 2))
        else:
            # For CPU-only, use smaller batches to reduce latency
            args.batch_size = 16
            args.batch_timeout = 5
        
        print(f"Auto-optimized batch parameters: size={args.batch_size}, timeout={args.batch_timeout}ms")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize threading lock for statistics
    statistics_lock = threading.Lock()
    
    # Prepare the model (convert to TorchScript if needed)
    model_path = prepare_model(args, args.game, args.size)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Record start time
    start_time = time.time()
    
    # Initialize monitoring log
    with open(os.path.join(args.output_dir, "monitoring.log"), "w") as f:
        f.write("timestamp,elapsed_seconds,games_completed,total_moves,games_per_second,moves_per_second,"
               "cpu_percent,gpu_percent,memory_mb\n")
    
    # Start self-play processes
    if args.use_cpp_binary and os.path.exists(cpp_binary_path):
        # Start C++ processes with binary
        print(f"Starting {args.processes} C++ self-play processes")
        for i in range(args.processes):
            proc, thread, log_file = start_cpp_process(args, i)
            processes.append((proc, thread, log_file))
    else:
        # Use Python wrapper for C++ code
        print(f"Starting {args.processes} Python-wrapped self-play processes")
        processes_info = start_py_cpp_selfplay(args, model_path)
    
    # Start resource monitoring
    monitor_thread = threading.Thread(target=monitor_resources, args=(args,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Wait for all processes to complete
        if args.use_cpp_binary:
            for _, thread, _ in processes:
                thread.join()
        else:
            for thread, _ in processes_info:
                thread.join()
        
        # Signal monitoring thread to stop
        running = False
        monitor_thread.join(timeout=2)
        
        # Make sure all game files are merged
        merge_game_files(args)
        
        # Print final statistics
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Self-play completed in {elapsed_time:.2f} seconds")
        print(f"Generated {total_games_completed} games with {total_moves_generated} total moves")
        print(f"Average moves per game: {total_moves_generated / total_games_completed:.1f}" if total_games_completed > 0 else "Average moves per game: N/A")
        print(f"Performance: {total_games_completed / elapsed_time:.2f} games/sec, {total_moves_generated / elapsed_time:.1f} moves/sec" if elapsed_time > 0 else "Performance: N/A")
        print("="*60)
        
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        print("\nInterrupted by user. Shutting down...")
        running = False
        
        # Terminate all child processes
        for proc, _, _ in processes:
            if hasattr(proc, 'poll') and proc.poll() is None:  # If process is still running
                try:
                    proc.terminate()
                except:
                    pass
    
    except Exception as e:
        print(f"\nError during self-play: {e}")
    
    finally:
        # Clean up open log files
        for proc, _, log_file in processes:
            if hasattr(log_file, 'close'):
                try:
                    log_file.close()
                except:
                    pass

if __name__ == "__main__":
    main()