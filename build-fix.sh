#!/bin/bash

# Save original environment variables
ORIGINAL_PATH="$PATH"
ORIGINAL_CPATH="${CPATH:-}"
ORIGINAL_LIBRARY_PATH="${LIBRARY_PATH:-}"
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
ORIGINAL_INCLUDE="${INCLUDE:-}"
ORIGINAL_C_INCLUDE_PATH="${C_INCLUDE_PATH:-}"
ORIGINAL_CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}"

# Remove any Anaconda/Windows paths from environment
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "anaconda\|cosmo" | tr '\n' ':' | sed 's/:$//')
export CPATH=$(echo "$CPATH" | tr ':' '\n' | grep -v "anaconda\|cosmo" | tr '\n' ':' | sed 's/:$//')
export LIBRARY_PATH=$(echo "$LIBRARY_PATH" | tr ':' '\n' | grep -v "anaconda\|cosmo" | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "anaconda\|cosmo" | tr '\n' ':' | sed 's/:$//')
export INCLUDE=$(echo "$INCLUDE" | tr ':' '\n' | grep -v "anaconda\|cosmo" | tr '\n' ':' | sed 's/:$//')
export C_INCLUDE_PATH=$(echo "$C_INCLUDE_PATH" | tr ':' '\n' | grep -v "anaconda\|cosmo" | tr '\n' ':' | sed 's/:$//')
export CPLUS_INCLUDE_PATH=$(echo "$CPLUS_INCLUDE_PATH" | tr ':' '\n' | grep -v "anaconda\|cosmo" | tr '\n' ':' | sed 's/:$//')

# Unset any potential conda-related variables
unset CONDA_PREFIX
unset CONDA_PYTHON_EXE
unset CONDA_DEFAULT_ENV
unset CONDA_EXE
unset CONDA_SHLVL

# Clean the build directory if it exists
echo "Cleaning build directory..."
rm -rf build
mkdir -p build

# Run CMake with proper flags
echo "Configuring with CMake..."
cd build
cmake -DCMAKE_CXX_FLAGS="-Wno-error" ..

# Build the project
echo "Building the project..."
make VERBOSE=1

# Restore original environment
export PATH="$ORIGINAL_PATH"
export CPATH="$ORIGINAL_CPATH"
export LIBRARY_PATH="$ORIGINAL_LIBRARY_PATH"
export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
export INCLUDE="$ORIGINAL_INCLUDE"
export C_INCLUDE_PATH="$ORIGINAL_C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$ORIGINAL_CPLUS_INCLUDE_PATH"

echo "Build process completed." 