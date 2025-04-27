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

# Compile and run test
echo "Compiling test_build.cpp..."
g++ -I./include test_build.cpp -o test_build -pthread

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the test..."
    ./test_build
else
    echo "Compilation failed!"
fi

# Restore original environment
export PATH="$ORIGINAL_PATH"
export CPATH="$ORIGINAL_CPATH"
export LIBRARY_PATH="$ORIGINAL_LIBRARY_PATH"
export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
export INCLUDE="$ORIGINAL_INCLUDE"
export C_INCLUDE_PATH="$ORIGINAL_C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$ORIGINAL_CPLUS_INCLUDE_PATH" 