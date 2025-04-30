#!/usr/bin/env python3
"""
Benchmarking script for AlphaZero models.

This script benchmarks AlphaZero model performance across different parameters
and hardware configurations.

Usage:
    python benchmark.py [options]

Options:
    --model MODEL           Path to model file
    --game {gomoku,chess,go}  Game type (default: gomoku)
    --size SIZE             Board size (default: depends on game)
    --simulations SIMS      Number of MCTS simulations per move (default: 800)
    --batch-size BATCH      Batch size for neural network inference (default: 16)
    --threads THREADS       Number of threads (default: 4)
    --iterations ITER       Number of benchmark iterations (default: 10)
    --output-file FILE      Output file for results (default: benchmark_results.json)
    --use-gpu               Use GPU for inference
    --variant               Use variant rules
"""

import os
import sys
import argparse
import json
import time
import timeit
import threading
import numpy as np
import torch
import _alphazero_cpp as az
from alphazero.models import DDWRandWireResNet

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Model Benchmarking")
    parser.add_argument("--model", type=str, default="",
                        help="Path to model file")
    parser.add_argument("--game", type=str, default="gomoku",
                        choices=["gomoku", "chess", "go"],
                        help="Game type")
    parser.add_argument("--size", type=int, default=0,
                        help="Board size (0 for default)")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for neural network inference")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--output-file", type=str, default="benchmark_results.json",
                        help="Output file for results")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for inference")
    parser.add_argument("--variant", action="store_true",
                        help="Use variant rules")
    parser.add_argument("--thread-scaling", action="store_true",
                        help="Benchmark scaling with different thread counts")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress output")
    return parser.parse_args()


def create_neural_network(args, game_type, board_size):
    """Create a neural network for benchmarking."""
    if args.model:
        try:
            # Try to load with the C++ API first
            nn = az.createNeuralNetwork(args.model, game_type, board_size, args.use_gpu)
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
                device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
                model = DDWRandWireResNet(input_channels, action_size)
                model.load_state_dict(torch.load(args.model, map_location=device))
                model = model.to(device)
                model.eval()
                
                # Create wrapper for the PyTorch model
                class TorchNeuralNetwork(az.NeuralNetwork):
                    def __init__(self, model, device, batch_size=16):
                        super().__init__()
                        self.model = model
                        self.device = device
                        self.batch_size = batch_size
                    
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
                        return self.batch_size
                    
                    def getModelInfo(self):
                        return "PyTorch DDWRandWireResNet"
                    
                    def getModelSizeBytes(self):
                        return sum(p.numel() * 4 for p in self.model.parameters())
                
                nn = TorchNeuralNetwork(model, device, args.batch_size)
                print(f"Loaded model from {args.model} (PyTorch)")
                return nn
            except Exception as e:
                print(f"Failed to load model with PyTorch: {e}")
                print("Using random policy network instead")
    
    # Fallback to random policy
    print("Using random policy network")
    return None


def benchmark_single_inference(nn, game_state):
    """Benchmark single inference speed."""
    # Warm-up
    for _ in range(5):
        nn.predict(game_state)
    
    # Benchmark
    start_time = time.time()
    iterations = 100
    
    for _ in range(iterations):
        nn.predict(game_state)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time / iterations  # Return average time per inference


def benchmark_batch_inference(nn, game_state, batch_size):
    """Benchmark batch inference speed."""
    # Create a batch of game states
    states = [game_state.clone() for _ in range(batch_size)]
    states_refs = [az.reference_wrapper(s) for s in states]
    
    # Containers for outputs
    policies = []
    values = []
    
    # Warm-up
    nn.predictBatch(states_refs, policies, values)
    
    # Benchmark
    start_time = time.time()
    iterations = 20
    
    for _ in range(iterations):
        policies = []
        values = []
        nn.predictBatch(states_refs, policies, values)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time / iterations  # Return average time per batch inference


def benchmark_mcts_search(mcts, iterations):
    """Benchmark MCTS search speed."""
    # Warm-up
    mcts.search()
    
    # Reset progress callback to avoid output spam
    mcts.setProgressCallback(None)
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(iterations):
        mcts.search()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time / iterations  # Return average time per search


def benchmark_thread_scaling(nn, game_state, tt, max_threads, simulations):
    """Benchmark MCTS performance scaling with different thread counts."""
    results = []
    
    for num_threads in range(1, max_threads + 1):
        print(f"Testing with {num_threads} threads...")
        
        # Create MCTS with current thread count
        mcts = az.ParallelMCTS(
            game_state, nn, tt,
            num_threads, simulations
        )
        
        # Reset progress callback to avoid output spam
        mcts.setProgressCallback(None)
        
        # Warm-up
        mcts.search()
        
        # Benchmark
        start_time = time.time()
        iterations = 3
        
        for _ in range(iterations):
            mcts.search()
        
        end_time = time.time()
        avg_search_time = (end_time - start_time) / iterations
        
        # Calculate nodes per second
        nodes_per_second = simulations / avg_search_time
        
        results.append({
            "threads": num_threads,
            "avg_search_time": avg_search_time,
            "nodes_per_second": nodes_per_second,
            "efficiency": nodes_per_second / (num_threads * (results[0]["nodes_per_second"] / 1)) if results else 1.0
        })
    
    return results


def benchmark_memory_usage(mcts, game_state):
    """Benchmark memory usage during MCTS search."""
    # Initial memory usage
    initial_usage = mcts.getMemoryUsage()
    
    # Run search to populate the tree
    mcts.search()
    
    # Final memory usage
    final_usage = mcts.getMemoryUsage()
    
    # Transposition table memory usage
    tt_memory = 0
    if hasattr(mcts, "getTranspositionTable"):
        tt = mcts.getTranspositionTable()
        if tt:
            tt_memory = tt.getMemoryUsageBytes()
    
    return {
        "initial_bytes": initial_usage,
        "final_bytes": final_usage,
        "growth_bytes": final_usage - initial_usage,
        "transposition_table_bytes": tt_memory,
        "initial_mb": initial_usage / (1024 * 1024),
        "final_mb": final_usage / (1024 * 1024),
        "growth_mb": (final_usage - initial_usage) / (1024 * 1024),
        "transposition_table_mb": tt_memory / (1024 * 1024)
    }


def run_benchmark(args):
    """Run the complete benchmark suite."""
    print("Starting AlphaZero benchmark...")
    
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
            board_size = 9  # Use smaller board for benchmarking
        else:
            board_size = 15
    else:
        board_size = args.size
    
    print(f"Game type: {args.game}")
    print(f"Board size: {board_size}")
    print(f"Simulations: {args.simulations}")
    print(f"Batch size: {args.batch_size}")
    print(f"Threads: {args.threads}")
    print(f"Iterations: {args.iterations}")
    print(f"Device: {'GPU' if args.use_gpu else 'CPU'}")
    
    # Create game state for benchmark
    game_state = az.createGameState(game_type, board_size, args.variant)
    
    # Create neural network
    nn = create_neural_network(args, game_type, board_size)
    
    # Create transposition table
    tt = az.TranspositionTable(1048576, 1024)
    
    # Create MCTS
    mcts = az.ParallelMCTS(
        game_state, nn, tt,
        args.threads, args.simulations
    )
    
    # Set MCTS parameters
    mcts.setCPuct(1.5)
    mcts.setFpuReduction(0.0)
    
    # Collect system information
    system_info = {
        "cpu": "Unknown",
        "memory": "Unknown",
        "gpu": "None",
        "operating_system": "Unknown"
    }
    
    # Try to get CPU info
    try:
        if sys.platform == "linux" or sys.platform == "linux2":
            # Linux
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    system_info["cpu"] = line.split(": ")[1]
                    break
        elif sys.platform == "darwin":
            # macOS
            import subprocess
            system_info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode("utf-8")
        elif sys.platform == "win32":
            # Windows
            import platform
            system_info["cpu"] = platform.processor()
    except Exception as e:
        print(f"Failed to get CPU info: {e}")
    
    # Try to get memory info
    try:
        if sys.platform == "linux" or sys.platform == "linux2":
            # Linux
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            for line in meminfo.split("\n"):
                if "MemTotal" in line:
                    mem_total = int(line.split()[1]) // 1024  # Convert from KB to MB
                    system_info["memory"] = f"{mem_total} MB"
                    break
        elif sys.platform == "darwin":
            # macOS
            import subprocess
            mem_total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) // (1024 * 1024)  # Convert from B to MB
            system_info["memory"] = f"{mem_total} MB"
        elif sys.platform == "win32":
            # Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong)
                ]
            memory_status = MEMORYSTATUS()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
            mem_total = memory_status.dwTotalPhys // (1024 * 1024)  # Convert from B to MB
            system_info["memory"] = f"{mem_total} MB"
    except Exception as e:
        print(f"Failed to get memory info: {e}")
    
    # Try to get GPU info
    try:
        if torch.cuda.is_available():
            system_info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        print(f"Failed to get GPU info: {e}")
    
    # Get OS info
    try:
        system_info["operating_system"] = f"{sys.platform}"
        if sys.platform == "linux" or sys.platform == "linux2":
            # Linux
            with open("/etc/os-release", "r") as f:
                os_release = f.read()
            for line in os_release.split("\n"):
                if line.startswith("PRETTY_NAME="):
                    system_info["operating_system"] = line.split("=")[1].strip('"')
                    break
        elif sys.platform == "darwin":
            # macOS
            import platform
            system_info["operating_system"] = f"macOS {platform.mac_ver()[0]}"
        elif sys.platform == "win32":
            # Windows
            import platform
            system_info["operating_system"] = platform.platform()
    except Exception as e:
        print(f"Failed to get OS info: {e}")
    
    # Dictionary to store benchmark results
    results = {
        "system_info": system_info,
        "benchmark_config": {
            "game": args.game,
            "board_size": board_size,
            "simulations": args.simulations,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "iterations": args.iterations,
            "use_gpu": args.use_gpu,
            "variant_rules": args.variant
        },
        "neural_network": {
            "model_path": args.model,
            "device": "GPU" if args.use_gpu and nn and nn.isGpuAvailable() else "CPU",
            "device_info": nn.getDeviceInfo() if nn else "None"
        },
        "results": {}
    }
    
    # Add model info if available
    if nn:
        results["neural_network"]["model_size_bytes"] = nn.getModelSizeBytes()
        results["neural_network"]["model_size_mb"] = nn.getModelSizeBytes() / (1024 * 1024)
        results["neural_network"]["model_info"] = nn.getModelInfo()
    
    print("\nRunning neural network inference benchmark...")
    
    # 1. Single inference benchmark
    if nn:
        single_inference_times = []
        for i in range(args.iterations):
            time_per_inference = benchmark_single_inference(nn, game_state)
            single_inference_times.append(time_per_inference)
            if not args.no_progress:
                print(f"Single inference iteration {i+1}/{args.iterations}: {time_per_inference:.6f}s")
        
        results["results"]["single_inference"] = {
            "times": single_inference_times,
            "mean": np.mean(single_inference_times),
            "median": np.median(single_inference_times),
            "min": np.min(single_inference_times),
            "max": np.max(single_inference_times),
            "std": np.std(single_inference_times)
        }
        
        print(f"Single inference: {results['results']['single_inference']['mean']:.6f}s (±{results['results']['single_inference']['std']:.6f}s)")
        
        # 2. Batch inference benchmark
        batch_inference_times = []
        for i in range(args.iterations):
            time_per_batch = benchmark_batch_inference(nn, game_state, args.batch_size)
            batch_inference_times.append(time_per_batch)
            if not args.no_progress:
                print(f"Batch inference iteration {i+1}/{args.iterations}: {time_per_batch:.6f}s")
        
        results["results"]["batch_inference"] = {
            "batch_size": args.batch_size,
            "times": batch_inference_times,
            "mean": np.mean(batch_inference_times),
            "median": np.median(batch_inference_times),
            "min": np.min(batch_inference_times),
            "max": np.max(batch_inference_times),
            "std": np.std(batch_inference_times),
            "time_per_sample": np.mean(batch_inference_times) / args.batch_size
        }
        
        print(f"Batch inference (batch_size={args.batch_size}): {results['results']['batch_inference']['mean']:.6f}s "
              f"({results['results']['batch_inference']['time_per_sample']:.6f}s per sample)")
    
    print("\nRunning MCTS search benchmark...")
    
    # 3. MCTS search benchmark
    search_times = []
    for i in range(args.iterations):
        time_per_search = benchmark_mcts_search(mcts, 1)
        search_times.append(time_per_search)
        if not args.no_progress:
            print(f"MCTS search iteration {i+1}/{args.iterations}: {time_per_search:.6f}s")
    
    results["results"]["mcts_search"] = {
        "times": search_times,
        "mean": np.mean(search_times),
        "median": np.median(search_times),
        "min": np.min(search_times),
        "max": np.max(search_times),
        "std": np.std(search_times),
        "nodes_per_second": args.simulations / np.mean(search_times)
    }
    
    print(f"MCTS search ({args.simulations} simulations): {results['results']['mcts_search']['mean']:.6f}s "
          f"({results['results']['mcts_search']['nodes_per_second']:.1f} nodes/s)")
    
    # 4. Thread scaling benchmark (optional)
    if args.thread_scaling:
        print("\nRunning thread scaling benchmark...")
        max_threads = min(16, os.cpu_count() or 4)
        scaling_results = benchmark_thread_scaling(nn, game_state, tt, max_threads, args.simulations)
        
        results["results"]["thread_scaling"] = scaling_results
        
        print("Thread scaling results:")
        for r in scaling_results:
            print(f"  {r['threads']} threads: {r['avg_search_time']:.6f}s "
                  f"({r['nodes_per_second']:.1f} nodes/s, efficiency: {r['efficiency']:.2f})")
    
    # 5. Memory usage benchmark
    print("\nRunning memory usage benchmark...")
    memory_results = benchmark_memory_usage(mcts, game_state)
    results["results"]["memory_usage"] = memory_results
    
    print(f"Memory usage: Initial: {memory_results['initial_mb']:.2f} MB, "
          f"Final: {memory_results['final_mb']:.2f} MB, "
          f"Growth: {memory_results['growth_mb']:.2f} MB")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {args.output_file}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)