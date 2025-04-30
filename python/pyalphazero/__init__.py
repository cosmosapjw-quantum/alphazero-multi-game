import os
import sys
import ctypes
import importlib.util

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Need to load dependencies with RTLD_GLOBAL flag to resolve symbols
torch_lib_path = "/opt/libtorch/lib/libtorch.so"
fmt_lib_path = os.path.join(current_dir, "lib/libfmt.so.8.1.1")

# Try to load the dependencies with RTLD_GLOBAL flag
try:
    if os.path.exists(torch_lib_path):
        ctypes.CDLL(torch_lib_path, mode=ctypes.RTLD_GLOBAL)
    if os.path.exists(fmt_lib_path):
        ctypes.CDLL(fmt_lib_path, mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    print(f"Warning: Failed to preload dependencies: {e}")

# Path to the .so file
so_path = os.path.join(current_dir, "__init__.so")

# Load the module from the .so file
try:
    spec = importlib.util.spec_from_file_location("_pyalphazero", so_path)
    _pyalphazero = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_pyalphazero)
    
    # Import all symbols from the module
    from _pyalphazero import *
    
    # Clean up
    del ctypes, importlib, spec, _pyalphazero
except ImportError as e:
    print(f"Error importing AlphaZero module: {e}")
    print("The C++ extension module couldn't be loaded.")
    print("Please follow the instructions in README_PYTHON_BINDINGS.md to fix this issue.")
    print("")
    print("You can still use the C++ implementation directly:")
    print("  - Run C++ tests: ./build/tests/core_tests")
    print("  - Use CLI app: ./build/bin/alphazero_cli_app --game gomoku --simulations 800 --threads 4") 