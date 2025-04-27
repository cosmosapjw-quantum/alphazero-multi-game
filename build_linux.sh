#!/bin/bash

# Clean build script for alphazero-multi-game in Linux WSL
# This script avoids path conflicts with Windows Anaconda

# Exit on any error
set -e

echo "=== Setting up clean build environment ==="

# Clean PATH - remove Anaconda paths
export PATH=$(echo $PATH | tr ":" "\n" | grep -v "anaconda3" | tr "\n" ":")

# Unset variables that might interfere with build
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset CPATH
unset CMAKE_INCLUDE_PATH
unset CMAKE_LIBRARY_PATH
unset CONDA_PREFIX
unset CONDA_EXE

echo "=== Creating build directory ==="
rm -rf build
mkdir -p build
cd build

echo "=== Running CMake ==="
# Use system paths only and disable LibTorch explicitly
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBTORCH_OFF=ON

echo "=== Building project ==="
# Build with clean environment
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin cmake --build . --parallel

echo "=== Building examples ==="
# Try to build each example individually to avoid failing the whole build if one fails
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin cmake --build . --target gomoku_example || echo "Failed to build gomoku_example"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin cmake --build . --target chess_example || echo "Failed to build chess_example"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin cmake --build . --target go_example || echo "Failed to build go_example"

echo "=== Build completed! ==="
echo
echo "Any available examples will be in the build/bin directory."
echo "Check for successful builds above." 