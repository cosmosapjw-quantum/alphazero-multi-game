#!/bin/bash

# Exit on error
set -e

# Parse command line arguments
BUILD_TYPE="Release"
BUILD_DIR="build"
BUILD_TESTS=ON
BUILD_PYTHON=OFF
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift
            shift
            ;;
        --with-python)
            BUILD_PYTHON=ON
            shift
            ;;
        --without-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Create build directory if it doesn't exist
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_TESTS=${BUILD_TESTS} \
    -DBUILD_PYTHON_BINDINGS=${BUILD_PYTHON}

# Build
echo "Building with ${NUM_JOBS} jobs..."
cmake --build . --parallel ${NUM_JOBS}

echo "Build complete. Executables are in ${BUILD_DIR}/bin/"