#!/bin/bash

# Save the original PATH
ORIGINAL_PATH="$PATH"

# Filter out Anaconda paths from PATH
# This prevents conflict between Linux and Windows (Anaconda) headers
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "anaconda3" | tr '\n' ':' | sed 's/:$//')

# Add our include path for types.h
export CXXFLAGS="$CXXFLAGS -I$(pwd)/include"

echo "Building with cleaner PATH that excludes Anaconda..."
echo "PATH=$PATH"

# Run CMake
mkdir -p build
cd build
cmake ..
make

# Restore the original PATH
export PATH="$ORIGINAL_PATH" 