#!/usr/bin/env python3

import os
import sys
import ctypes

# Load libraries with RTLD_GLOBAL to ensure symbols are globally visible
def load_library(lib_path):
    try:
        return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Failed to load {lib_path}: {e}")
        return None

# Try to load the fmt library first
fmt_lib = load_library("/usr/lib/x86_64-linux-gnu/libfmt.so.8.1.1")
if fmt_lib:
    print("Successfully loaded fmt library")
else:
    print("Failed to load fmt library")

# Try to load the torch libraries in the correct order
torch_lib_dir = "/opt/libtorch/lib"
torch_libs = [
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
]

for lib in torch_libs:
    lib_path = os.path.join(torch_lib_dir, lib)
    if load_library(lib_path):
        print(f"Successfully loaded {lib}")
    else:
        print(f"Failed to load {lib}")

# Now try to import the PyAlphaZero module
try:
    # Add the build directory to the path
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/src/pybind'))
    sys.path.append(build_dir)
    
    # Import the module directly using the actual module name
    module_files = [f for f in os.listdir(build_dir) if f.startswith('alphazero.cpython') and f.endswith('.so')]
    
    if not module_files:
        print("Could not find any AlphaZero Python modules in build directory")
        sys.exit(1)
    
    module_path = os.path.join(build_dir, module_files[0])
    print(f"Loading module from: {module_path}")
    
    # The module name should be 'alphazero' without the extension
    # We'll directly import the module without creating a new name
    import alphazero
    
    # Print out available functions
    print("\nAvailable functions:", [name for name in dir(alphazero) if not name.startswith('__')])
    
    # Try to create a simple game state to test
    try:
        print("\nTrying to create a Gomoku game state...")
        gomoku_state = alphazero.createGameState(alphazero.GameType.GOMOKU, 9, False)
        print(f"Game type: {gomoku_state.getGameType()}")
        print(f"Board size: {gomoku_state.getBoardSize()}")
        print(f"Current player: {gomoku_state.getCurrentPlayer()}")
        print("Success! Python bindings are working correctly.")
    except Exception as e:
        print(f"Error creating game state: {e}")
    
except ImportError as e:
    print(f"Error importing AlphaZero module: {e}")
    print("\nYou can still use the C++ implementation directly:")
    print("  - Run C++ tests: ./build/tests/core_tests")
    print("  - Use CLI app: ./build/bin/alphazero_cli_app --game gomoku --simulations 800 --threads 4")
    sys.exit(1) 