#!/usr/bin/env python3
"""
Simple test script for AlphaZero Python bindings.
This loads dependencies with RTLD_GLOBAL before importing the module.
"""

import os
import sys
import ctypes

def load_library(lib_path):
    """Load a library with RTLD_GLOBAL."""
    try:
        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        print(f"Successfully loaded {lib_path}")
        return lib
    except Exception as e:
        print(f"Failed to load {lib_path}: {e}")
        return None

# Load dependencies with RTLD_GLOBAL
print("Loading dependencies...")
fmt_lib = load_library("/usr/lib/x86_64-linux-gnu/libfmt.so.8.1.1")
c10_lib = load_library("/opt/libtorch/lib/libc10.so")
torch_cpu_lib = load_library("/opt/libtorch/lib/libtorch_cpu.so")
torch_lib = load_library("/opt/libtorch/lib/libtorch.so")

# Directory with the AlphaZero module
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/src/pybind'))
sys.path.append(build_dir)

# Find the module
module_files = [f for f in os.listdir(build_dir) 
               if f.startswith('alphazero.cpython') and f.endswith('.so')]

if not module_files:
    print("Could not find any AlphaZero modules in build directory.")
    sys.exit(1)

module_path = os.path.join(build_dir, module_files[0])
print(f"Found module: {module_path}")

# Try to import the module - this will only work with the correct Python version
try:
    print("Attempting to import module...")
    import alphazero
    
    print("\nAvailable game types:")
    print(f"  GOMOKU = {alphazero.GameType.GOMOKU}")
    print(f"  CHESS = {alphazero.GameType.CHESS}")
    print(f"  GO = {alphazero.GameType.GO}")
    
    print("\nCreating a Gomoku game state...")
    gomoku = alphazero.createGameState(alphazero.GameType.GOMOKU, 9, False)
    print(f"Board size: {gomoku.getBoardSize()}")
    print(f"Action space size: {gomoku.getActionSpaceSize()}")
    print(f"Current player: {gomoku.getCurrentPlayer()}")
    
    print("\nMaking a move at the center...")
    center = gomoku.getBoardSize() // 2
    move = center * gomoku.getBoardSize() + center
    gomoku.makeMove(move)
    print(f"Board after move:\n{gomoku.toString()}")
    
    print("\nSuccess! Python bindings are working correctly.")
    
except ImportError as e:
    print(f"Failed to import module: {e}")
    print("\nThis is likely due to a Python version mismatch.")
    print(f"Module was built for Python {module_files[0].split('.')[1].replace('cpython-', '')}")
    print(f"Your Python version is {sys.version_info.major}.{sys.version_info.minor}")
    print("\nPlease try running this script with the correct Python version.")
    
except Exception as e:
    print(f"Error: {e}") 