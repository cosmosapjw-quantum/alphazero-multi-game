#!/bin/bash

#============================================================================
# AlphaZero Build Script
# 
# This script configures and builds the AlphaZero project with CMake.
#============================================================================

set -e

# Check if running in Anaconda/Miniconda environment and handle it properly
if [ -n "$CONDA_PREFIX" ]; then
    echo "Detected Anaconda/Miniconda environment: $CONDA_PREFIX"
    echo "Setting up environment to avoid library conflicts..."
    
    # Save current conda environment variables
    CONDA_BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
    
    # Unset Conda compiler variables to prioritize system ones
    unset CC CXX FC F77 F90 F95
    
    # Force using system libraries for compilation
    export CXXFLAGS="-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"
    export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
    export CMAKE_PREFIX_PATH="/usr:$CMAKE_PREFIX_PATH"
fi

# ANSI color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#============================================================================
# Configuration variables
#============================================================================
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
TORCH_DIR=""
CMAKE_OPTIONS=""

# Get number of CPU cores for parallel build
if command -v nproc >/dev/null 2>&1; then
    NUM_JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    NUM_JOBS=$(sysctl -n hw.ncpu)
else
    NUM_JOBS=4
fi

#============================================================================
# Helper functions
#============================================================================

# Print error message and exit
function error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Print warning message
function warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Print info message
function info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

# Print success message
function success() {
    echo -e "${GREEN}$1${NC}"
}

# Print help message
function print_help() {
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
}

# Check if LibTorch is available
function check_libtorch() {
    if [ -n "$TORCH_DIR" ] && [ ! -d "$TORCH_DIR" ]; then
        error_exit "Specified LibTorch directory does not exist: $TORCH_DIR"
    fi
    
    if [ -z "$TORCH_DIR" ] && [ "$ENABLE_PYTHON" = "ON" ]; then
        warning "No LibTorch directory specified. Will attempt to use system-installed LibTorch."
        warning "If build fails, use --torch-dir or --disable-python."
    fi
}

# Print LibTorch installation instructions
function print_libtorch_instructions() {
    echo ""
    echo "NOTE: To use PyTorch (LibTorch), you need to:"
    echo ""
    echo "  1. Install LibTorch and specify its location with --torch-dir"
    echo "     Example: ./build.sh --torch-dir=/path/to/libtorch"
    echo ""
    echo "  2. Download and extract LibTorch:"
    echo "     For CUDA 12.4 support:"
    echo "     wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu124.zip"
    echo "     unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu124.zip"
    echo ""
    echo "     For CUDA 12.1:"
    echo "     wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    echo "     unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip"
    echo ""
    echo "     For CPU-only:"
    echo "     wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
    echo "     unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip"
    echo ""
    echo "     Then run build with: ./build.sh --torch-dir=$(pwd)/libtorch"
    echo ""
    echo "  3. Or disable Python bindings: ./build.sh --disable-python"
}

#============================================================================
# Parse command line arguments
#============================================================================
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
        --build-dir=*)
            BUILD_DIR="${key#*=}"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --install-prefix=*)
            INSTALL_PREFIX="${key#*=}"
            shift
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
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
        --torch-dir=*)
            TORCH_DIR="${key#*=}"
            shift
            ;;
        --torch-dir)
            TORCH_DIR="$2"
            shift 2
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
        --jobs=*)
            NUM_JOBS="${key#*=}"
            shift
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            warning "Unknown option: $key"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

#============================================================================
# Check dependencies and environment
#============================================================================

# Check for required tools
command -v cmake >/dev/null 2>&1 || error_exit "CMake is required but not installed."
if [ "$USE_NINJA" = "ON" ]; then
    command -v ninja >/dev/null 2>&1 || error_exit "Ninja is required with --use-ninja but not installed."
fi

# Check CUDA if GPU is enabled
if [ "$ENABLE_GPU" = "ON" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        info "Found CUDA:"
        nvcc --version
    else
        warning "NVCC (CUDA Toolkit) is not found. GPU support will be disabled."
        warning "Install CUDA Toolkit if you need GPU support."
        ENABLE_GPU=OFF
    fi
fi

# Check PyTorch/LibTorch
check_libtorch

#============================================================================
# Configure CMake build options
#============================================================================

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Set CMake generator
if [ "$USE_NINJA" = "ON" ]; then
    CMAKE_OPTIONS="$CMAKE_OPTIONS -G Ninja"
fi

# Set build type
CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_BUILD_TYPE=$BUILD_TYPE"

# Let's skip the additional flags for now
# if [ -n "$CONDA_PREFIX" ]; then
#     CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_CXX_FLAGS=..."
# fi

# Set AlphaZero options
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
else
    # If not specified via --torch-dir, try to hint from Conda env if possible
    CONDA_TORCH_CMAKE_DIR="$CONDA_PREFIX/lib/python*/site-packages/torch/share/cmake/Torch"
    if [ -d "$(eval echo $CONDA_TORCH_CMAKE_DIR)" ]; then
      CMAKE_OPTIONS="$CMAKE_OPTIONS -DTorch_DIR=$(eval echo $CONDA_TORCH_CMAKE_DIR)"
      info "Using Torch from Conda environment: $(eval echo $CONDA_TORCH_CMAKE_DIR)"
    fi
fi

# Add CUDA-related options when CUDA is available
if [ "$ENABLE_GPU" = "ON" ] && command -v nvcc >/dev/null 2>&1; then
    # Get CUDA version and directory
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    # Find CUDA installation directory
    if [ -d "/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR" ]; then
        CUDA_DIR="/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR"
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_DIR="/usr/local/cuda"
    else
        error_exit "Could not find CUDA installation directory"
    fi
    
    info "Using CUDA installation at: $CUDA_DIR"
    
    # Add CUDA options
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_DIR"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_CUDA_COMPILER=$CUDA_DIR/bin/nvcc"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDA_INCLUDE_DIRS=$CUDA_DIR/include"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDA_CUDART_LIBRARY=$CUDA_DIR/lib64/libcudart.so"
    
    # Set CUDA architectures based on available GPU
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DTORCH_CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;9.0"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_CUDA_ARCHITECTURES=70;75;80;86;90"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_CUDA_STANDARD=17"
    
    # Check for cuDNN - more thorough check for different versions and locations
    CUDNN_LIB_PATH=""

    # Find any libcudnn.so version in standard locations
    if [ -f "/usr/lib/x86_64-linux-gnu/libcudnn.so" ]; then
        CUDNN_LIB_PATH="/usr/lib/x86_64-linux-gnu/libcudnn.so"
    elif ls /usr/lib/x86_64-linux-gnu/libcudnn.so.* >/dev/null 2>&1; then
        CUDNN_LIB_PATH=$(ls /usr/lib/x86_64-linux-gnu/libcudnn.so.* | head -1)
    elif [ -f "$CUDA_DIR/lib64/libcudnn.so" ]; then
        CUDNN_LIB_PATH="$CUDA_DIR/lib64/libcudnn.so"
    elif ls $CUDA_DIR/lib64/libcudnn.so.* >/dev/null 2>&1; then
        CUDNN_LIB_PATH=$(ls $CUDA_DIR/lib64/libcudnn.so.* | head -1)
    fi

    # If not found in standard places, search recursively
    if [ -z "$CUDNN_LIB_PATH" ]; then
        info "Searching for cuDNN library..."
        for dir in "/usr/local" "/usr" "$CUDA_DIR"; do
            if [ -d "$dir" ]; then
                FOUND_LIB=$(find "$dir" -name "libcudnn.so*" | head -1)
                if [ -n "$FOUND_LIB" ]; then
                    CUDNN_LIB_PATH="$FOUND_LIB"
                    info "Found cuDNN at $CUDNN_LIB_PATH"
                    break
                fi
            fi
        done
    fi
    
    # Now find the header
    CUDNN_INCLUDE_DIR=""
    if [ -f "/usr/include/cudnn.h" ]; then
        CUDNN_INCLUDE_DIR="/usr/include"
    elif [ -f "/usr/include/cuda/cudnn.h" ]; then
        CUDNN_INCLUDE_DIR="/usr/include/cuda"
    elif [ -f "/usr/local/include/cudnn.h" ]; then
        CUDNN_INCLUDE_DIR="/usr/local/include"
    elif [ -f "$CUDA_DIR/include/cudnn.h" ]; then
        CUDNN_INCLUDE_DIR="$CUDA_DIR/include"
    fi

    # If header not found in standard places, search recursively
    if [ -z "$CUDNN_INCLUDE_DIR" ] && [ -n "$CUDNN_LIB_PATH" ]; then
        info "Searching for cuDNN headers..."
        for dir in "/usr/local" "/usr" "$CUDA_DIR"; do
            if [ -d "$dir" ]; then
                FOUND_INCLUDE=$(find "$dir" -name "cudnn.h" | head -1)
                if [ -n "$FOUND_INCLUDE" ]; then
                    CUDNN_INCLUDE_DIR=$(dirname "$FOUND_INCLUDE")
                    info "Found cuDNN header at $CUDNN_INCLUDE_DIR"
                    break
                fi
            fi
        done
    fi
    
    if [ -n "$CUDNN_LIB_PATH" ] && [ -n "$CUDNN_INCLUDE_DIR" ]; then
        # Always signal intent to use CUDNN to AlphaZero if found
        CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_USE_CUDNN=ON"
        # Remove TORCH_USE_CUDNN as it's unused and ignored by Conda Torch CMake
        # CMAKE_OPTIONS="$CMAKE_OPTIONS -DTORCH_USE_CUDNN=ON" 

        # Only pass explicit system CUDNN paths if NOT in Conda env, 
        # otherwise let Torch find its own bundled/compatible one.
        if [ -z "$CONDA_PREFIX" ]; then
            CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDNN_LIBRARY=$CUDNN_LIB_PATH"
            CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR"
            info "Passing system cuDNN paths to CMake: Library=$CUDNN_LIB_PATH, Include=$CUDNN_INCLUDE_DIR"
        else
            info "In Conda environment, letting Torch find its own cuDNN (still passing -DTORCH_USE_CUDNN=ON)."
        fi
    else
        CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_USE_CUDNN=OFF"
        # TORCH_USE_CUDNN is implicitly OFF if CUDNN is not found
        warning "cuDNN library or headers not found, will build without cuDNN support"
        warning "Please make sure cuDNN is installed and accessible"
    fi
fi

# Force using system Python instead of Anaconda
if [ "$ENABLE_PYTHON" = "ON" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_PATH=$(which python3)
        CMAKE_OPTIONS="$CMAKE_OPTIONS -DPYTHON_EXECUTABLE=$PYTHON_PATH"
        CMAKE_OPTIONS="$CMAKE_OPTIONS -DPython_EXECUTABLE=$PYTHON_PATH"
        
        # Force using system libstdc++ instead of conda's
        export CMAKE_CXX_FLAGS="-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
        
        # Check for pybind11
        if python3 -c "import pybind11" >/dev/null 2>&1; then
            PYBIND11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')
            CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_DOWNLOAD_DEPENDENCIES=OFF" 
            CMAKE_OPTIONS="$CMAKE_OPTIONS -Dpybind11_DIR=$PYBIND11_DIR"
            info "Using pybind11 from: $PYBIND11_DIR"
        else
            warning "pybind11 not found, will download it during configuration"
            CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_DOWNLOAD_DEPENDENCIES=ON"
        fi
    else
        warning "Python 3 not found, will disable Python bindings"
        ENABLE_PYTHON=OFF
        CMAKE_OPTIONS="$CMAKE_OPTIONS -DALPHAZERO_ENABLE_PYTHON=OFF"
    fi
fi

#============================================================================
# Run CMake configuration
#============================================================================
info "Configuring with options: $CMAKE_OPTIONS"

# Construct the path to Torch CMake config within the provided TORCH_DIR or discovered system paths
TORCH_CMAKE_CONFIG_DIR=""
SYSTEM_TORCH_FOUND=0

# Construct the path to Torch CMake config within the provided TORCH_DIR
TORCH_CMAKE_CONFIG_DIR=""
if [ -n "$TORCH_DIR" ]; then
    # User specified TORCH_DIR takes precedence
    TORCH_CMAKE_CONFIG_DIR="${TORCH_DIR}/share/cmake/Torch"
    if [ -d "$TORCH_CMAKE_CONFIG_DIR" ]; then
        # Prepend the correct Torch_DIR definition for find_package
        CMAKE_OPTIONS="-DTorch_DIR=${TORCH_CMAKE_CONFIG_DIR} $CMAKE_OPTIONS"
        info "Explicitly setting Torch_DIR for CMake from --torch-dir: ${TORCH_CMAKE_CONFIG_DIR}"
        SYSTEM_TORCH_FOUND=1 # Flag that we found Torch here
    else
        warning "Torch CMake config directory not found at specified --torch-dir: ${TORCH_CMAKE_CONFIG_DIR}"
    fi
elif [ -n "$CONDA_PREFIX" ]; then
    # Try finding Torch in Conda environment if TORCH_DIR is not specified
    CONDA_TORCH_CMAKE_DIR_PATTERN="$CONDA_PREFIX/lib/python*/site-packages/torch/share/cmake/Torch"
    MATCHING_CONDA_TORCH_DIRS=$(eval echo $CONDA_TORCH_CMAKE_DIR_PATTERN)
    # Check if the pattern expanded to an existing directory
    if [ -d "$MATCHING_CONDA_TORCH_DIRS" ]; then
        TORCH_CMAKE_CONFIG_DIR="$MATCHING_CONDA_TORCH_DIRS"
        CMAKE_OPTIONS="-DTorch_DIR=${TORCH_CMAKE_CONFIG_DIR} $CMAKE_OPTIONS"
        info "Using Torch from Conda environment: ${TORCH_CMAKE_CONFIG_DIR}"
        SYSTEM_TORCH_FOUND=1 # Flag that we found Torch here
    fi
fi

# If Torch wasn't found via --torch-dir or Conda, search system paths
if [ "$SYSTEM_TORCH_FOUND" -eq 0 ]; then
    info "Searching for system-installed LibTorch..."
    SYSTEM_TORCH_PATHS=(
        "/usr/local/lib/python*/dist-packages/torch/share/cmake/Torch"
        "/usr/lib/python*/dist-packages/torch/share/cmake/Torch"
        "/opt/pytorch/torch/share/cmake/Torch" # Common location for some installations
    )
    for pattern in "${SYSTEM_TORCH_PATHS[@]}"; do
        MATCHING_SYSTEM_TORCH_DIRS=$(eval echo $pattern)
        # Check if the pattern expanded to an existing directory
        if [ -d "$MATCHING_SYSTEM_TORCH_DIRS" ]; then
            TORCH_CMAKE_CONFIG_DIR="$MATCHING_SYSTEM_TORCH_DIRS"
            CMAKE_OPTIONS="-DTorch_DIR=${TORCH_CMAKE_CONFIG_DIR} $CMAKE_OPTIONS"
            info "Found system Torch_DIR for CMake: ${TORCH_CMAKE_CONFIG_DIR}"
            SYSTEM_TORCH_FOUND=1
            break # Found it, no need to check further
        fi
    done

    if [ "$SYSTEM_TORCH_FOUND" -eq 0 ]; then
        warning "Could not automatically detect LibTorch installation (checked --torch-dir, Conda env, and system paths)."
        warning "CMake will attempt find_package(Torch). If it fails, provide --torch-dir."
        # Let CMake try to find it using its default search paths
    fi
fi

cmake .. $CMAKE_OPTIONS || error_exit "CMake configuration failed."

#============================================================================
# Build the project
#============================================================================
info "Building with $NUM_JOBS jobs..."
if [ "$USE_NINJA" = "ON" ]; then
    ninja -j $NUM_JOBS || error_exit "Build failed."
else
    if [ "$VERBOSE" = "ON" ]; then
        make VERBOSE=1 -j $NUM_JOBS || error_exit "Build failed."
    else
        make -j $NUM_JOBS || error_exit "Build failed."
    fi
fi

#============================================================================
# Run tests if requested
#============================================================================
if [ "$RUN_TESTS" = "ON" ]; then
    info "Running tests..."
    ctest --output-on-failure
fi

#============================================================================
# Install if prefix is specified
#============================================================================
if [ -n "$INSTALL_PREFIX" ]; then
    info "Installing to $INSTALL_PREFIX..."
    if [ "$USE_NINJA" = "ON" ]; then
        ninja install || error_exit "Installation failed."
    else
        make install || error_exit "Installation failed."
    fi
fi

#============================================================================
# Print summary
#============================================================================
success "Build completed successfully!"

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

#============================================================================
# Print run instructions
#============================================================================
echo ""
echo "To run the CLI:"
echo "  $BUILD_DIR/bin/alphazero_cli_app --game gomoku --simulations 800 --threads 4"
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

# Print LibTorch instructions if needed
if [ "$ENABLE_PYTHON" = "ON" ] && [ -z "$TORCH_DIR" ] && ! cmake --find-package -DNAME=Torch -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST >/dev/null 2>&1; then
    print_libtorch_instructions
fi