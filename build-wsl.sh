#!/bin/bash

# This script isolates the build environment from Windows/Anaconda paths
# and helps resolve pthread conflicts in WSL environments

# Save original environment variables
ORIGINAL_PATH="$PATH"
ORIGINAL_CPATH="${CPATH:-}"
ORIGINAL_LIBRARY_PATH="${LIBRARY_PATH:-}"
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
ORIGINAL_INCLUDE="${INCLUDE:-}"
ORIGINAL_C_INCLUDE_PATH="${C_INCLUDE_PATH:-}"
ORIGINAL_CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}"

# Filter out Windows/Anaconda paths from environment
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files|cosmo/AppData" | tr '\n' ':' | sed 's/:$//')
export CPATH=$(echo "$CPATH" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files|cosmo/AppData" | tr '\n' ':' | sed 's/:$//')
export LIBRARY_PATH=$(echo "$LIBRARY_PATH" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files|cosmo/AppData" | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files|cosmo/AppData" | tr '\n' ':' | sed 's/:$//')
export INCLUDE=$(echo "$INCLUDE" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files|cosmo/AppData" | tr '\n' ':' | sed 's/:$//')
export C_INCLUDE_PATH=$(echo "$C_INCLUDE_PATH" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files|cosmo/AppData" | tr '\n' ':' | sed 's/:$//')
export CPLUS_INCLUDE_PATH=$(echo "$CPLUS_INCLUDE_PATH" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files|cosmo/AppData" | tr '\n' ':' | sed 's/:$//')

# Unset Conda-related variables
unset CONDA_PREFIX
unset CONDA_PYTHON_EXE
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_SHLVL

# Define additional flags to work around pthread conflicts
export CXXFLAGS="-Wno-error -DWIN32_LEAN_AND_MEAN -D__MINGW64__ -I$(pwd)/include $CXXFLAGS"
export CFLAGS="-Wno-error -DWIN32_LEAN_AND_MEAN -D__MINGW64__ -I$(pwd)/include $CFLAGS"

# Clean the build directory
echo "Cleaning build directory..."
rm -rf build
mkdir -p build

# Run CMake to configure the project
echo "Configuring with CMake..."
cd build
cmake -DCMAKE_CXX_FLAGS="$CXXFLAGS" ..

# Build the project
echo "Building the project..."
make

# Restore original environment
export PATH="$ORIGINAL_PATH"
export CPATH="$ORIGINAL_CPATH"
export LIBRARY_PATH="$ORIGINAL_LIBRARY_PATH"
export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
export INCLUDE="$ORIGINAL_INCLUDE"
export C_INCLUDE_PATH="$ORIGINAL_C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$ORIGINAL_CPLUS_INCLUDE_PATH"

echo "Build process completed." 