# Python Bindings for AlphaZero

## Current Status
The C++ implementation of AlphaZero is working correctly, as evidenced by the passing C++ tests.

## Python Binding Issues
We are currently encountering several issues with the Python bindings:

1. Missing symbols from the fmt library:
   ```
   undefined symbol: _ZN3fmt2v86detail14snprintf_floatIeEEiT_iNS1_11float_specsERNS1_6bufferIcEE
   ```

2. PyTorch library conflicts:
   ```
   /home/cosmos/.local/lib/python3.10/site-packages/torch/lib/libshm.so: undefined symbol: _ZN3c105utils9str_errorEi
   ```

3. Python version mismatch: The module was built for Python 3.10, but you might be using a different version.

These issues are related to how the different libraries are compiled and linked, and conflicts between system libraries and PyTorch libraries.

## Solutions Tried

1. Explicitly linking with fmt library in CMakeLists.txt:
   ```cmake
   target_link_libraries(alphazero_py PRIVATE fmt::fmt)
   ```
   
2. Preloading fmt library with RTLD_GLOBAL before importing the module:
   ```python
   import ctypes
   ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libfmt.so.8.1.1", mode=ctypes.RTLD_GLOBAL)
   ```

3. Rebuilding with -fvisibility=default flag:
   ```bash
   cmake .. -DCMAKE_CXX_FLAGS="-fvisibility=default"
   ```

4. Copying the system fmt library into the Python package directory.

5. Preloading PyTorch libraries in the correct order.

## Using the C++ Implementation
Until the Python binding issues are resolved, we recommend using the C++ implementation directly. You can:

1. Run the C++ tests to verify implementation correctness:
   ```
   ./build/tests/core_tests
   ./build/tests/game_tests
   ./build/tests/mcts_tests
   ```

2. To build the CLI application (which is currently not built):
   ```
   cd build
   make alphazero_cli_app
   ```

## Quick Test
We've provided a test script that tries to load the Python bindings with the correct libraries preloaded:
```bash
# Run with the specific Python version the module was built for
python3.10 python/test_alphazero.py
```

Note: This might still fail due to PyTorch library conflicts.

## Next Steps for Python Bindings
To fully resolve the Python binding issues, you will need to:

1. Set up a clean Python environment with matching versions:
   - Use Python 3.10 (the version the module was built for)
   - Install PyTorch in that environment (preferably matching the libtorch version used in C++)
   - Ensure fmt library is installed and properly linked

2. Rebuild the Python bindings with the following modifications:
   - Ensure fmt library is statically linked into the module
   - Use Python 3.10 for building the module
   - Update the build flags to ensure all symbols are exported

3. Consider using SWIG or a different approach for generating Python bindings if pybind11 continues to cause issues

## Workaround for Testing
For testing purposes, you can use the following approach:

1. Create a Python script that preloads all necessary libraries before importing the module:
   ```python
   import ctypes
   
   # Load dependencies with RTLD_GLOBAL
   ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libfmt.so.8.1.1", mode=ctypes.RTLD_GLOBAL)
   ctypes.CDLL("/opt/libtorch/lib/libc10.so", mode=ctypes.RTLD_GLOBAL)
   ctypes.CDLL("/opt/libtorch/lib/libtorch_cpu.so", mode=ctypes.RTLD_GLOBAL)
   ctypes.CDLL("/opt/libtorch/lib/libtorch.so", mode=ctypes.RTLD_GLOBAL)
   
   # Import the module
   import sys
   sys.path.append('/path/to/build/dir')
   import alphazero
   ```

2. Make sure to use Python 3.10 to match the version the module was built for. 