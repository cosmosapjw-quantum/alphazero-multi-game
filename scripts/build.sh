#!/bin/bash

# build.sh - Build script for AlphaZero
set -e

# Default build variables
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_PREFIX=""
ENABLE_GPU=ON
BUILD_TESTS=ON
ENABLE_PYTHON=ON
BUILD_EXAMPLES=ON
USE_NINJA=OFF
RUN_TESTS=OFF
VERBOSE=OFF
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
TORCH_DIR=""

# Function to display an error message and exit
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift
            shift
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift
            shift
            ;;
        --disable-gpu)
            ENABLE_GPU=OFF
            shift
            ;;
        --disable-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --disable-python)
            ENABLE_PYTHON=OFF
            shift
            ;;
        --disable-examples)
            BUILD_EXAMPLES=OFF
            shift
            ;;
        --torch-dir)
            TORCH_DIR="$2"
            shift
            shift
            ;;
        --use-ninja)
            USE_NINJA=ON
            shift
            ;;
        --run-tests)
            RUN_TESTS=ON
            shift
            ;;
        --verbose)
            VERBOSE=ON
            shift
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift
            shift
            ;;
        --help)
            echo "AlphaZero Build Script"
            echo "======================"
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug                 Build in Debug mode"
            echo "  --release               Build in Release mode (default)"
            echo "  --build-dir DIR         Build directory (default: build)"
            echo "  --install-prefix DIR    Installation prefix"
            echo "  --disable-gpu           Disable GPU support"
            echo "  --disable-tests         Disable building tests"
            echo "  --disable-python        Disable Python bindings"
            echo "  --disable-examples      Disable building examples"
            echo "  --torch-dir DIR         LibTorch installation directory"
            echo "  --use-ninja             Use Ninja build system"
            echo "  --run-tests             Run tests after building"
            echo "  --verbose               Verbose output"
            echo "  --jobs N                Number of parallel build jobs (default: auto)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for required tools
command -v cmake >/dev/null 2>&1 || error_exit "CMake is required but not installed."
if [ "$USE_NINJA" = "ON" ]; then
    command -v ninja >/dev/null 2>&1 || error_exit "Ninja is required with --use-ninja but not installed."
fi

# Check CUDA if GPU is enabled
if [ "$ENABLE_GPU" = "ON" ]; then
    command -v nvcc >/dev/null 2>&1 || error_exit "NVCC (CUDA Toolkit) is required with GPU support but not found."
    echo "Found CUDA:"
    nvcc --version
fi

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Set CMAKE_GENERATOR for Ninja
GENERATOR_OPTION=""
if [ "$USE_NINJA" = "ON" ]; then
    GENERATOR_OPTION="-G Ninja"
fi

# Set build type
CMAKE_OPTIONS="$GENERATOR_OPTION -DCMAKE_BUILD_TYPE=$BUILD_TYPE"

# Set other options
CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_ENABLE_GPU=$ENABLE_GPU"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_BUILD_TESTS=$BUILD_TESTS"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_ENABLE_PYTHON=$ENABLE_PYTHON"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_BUILD_EXAMPLES=$BUILD_EXAMPLES"

# Add installation prefix if specified
if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
fi

# Add Torch directory if specified
if [ -n "$TORCH_DIR" ]; then
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DTORCH_DIR=$TORCH_DIR"
fi

# Configure
echo "Configuring with options: $CMAKE_OPTIONS"
cmake .. $CMAKE_OPTIONS || error_exit "CMake configuration failed."

# Build
echo "Building with $NUM_JOBS jobs..."
if [ "$USE_NINJA" = "ON" ]; then
    ninja -j $NUM_JOBS || error_exit "Build failed."
else
    if [ "$VERBOSE" = "ON" ]; then
        make VERBOSE=1 -j $NUM_JOBS || error_exit "Build failed."
    else
        make -j $NUM_JOBS || error_exit "Build failed."
    fi
fi

# Run tests if requested
if [ "$RUN_TESTS" = "ON" ]; then
    echo "Running tests..."
    ctest --output-on-failure
fi

# Install if prefix is specified
if [ -n "$INSTALL_PREFIX" ]; then
    echo "Installing to $INSTALL_PREFIX..."
    if [ "$USE_NINJA" = "ON" ]; then
        ninja install || error_exit "Installation failed."
    else
        make install || error_exit "Installation failed."
    fi
fi

echo "Build completed successfully!"

# Print summary
echo ""
echo "Build Summary:"
echo "  Build Type: $BUILD_TYPE"
echo "  Build Directory: $BUILD_DIR"
if [ -n "$INSTALL_PREFIX" ]; then
    echo "  Install Prefix: $INSTALL_PREFIX"
fi
echo "  GPU Support: $ENABLE_GPU"
echo "  Python Bindings: $ENABLE_PYTHON"
echo "  Examples: $BUILD_EXAMPLES"
echo "  Tests: $BUILD_TESTS"

# Print run instructions
echo ""
echo "To run the CLI:"
echo "  $BUILD_DIR/bin/alphazero_cli --game gomoku --simulations 800 --threads 4"
if [ "$BUILD_TESTS" = "ON" ]; then
    echo ""
    echo "To run all tests:"
    echo "  cd $BUILD_DIR && ctest"
fi
if [ "$BUILD_EXAMPLES" = "ON" ]; then
    echo ""
    echo "To run the Gomoku example:"
    echo "  $BUILD_DIR/bin/examples/gomoku_example"
fi