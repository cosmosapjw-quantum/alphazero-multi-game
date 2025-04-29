#!/bin/bash
set -e

# Create a Docker container with all the required dependencies
echo "===> Creating Docker build container..."

# Remove any existing build directory to start clean
echo "===> Cleaning previous build artifacts..."
sudo rm -rf build

sudo docker run --rm -v "$(pwd):/src" --name alphazero-test-build -w /src nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 bash -c '
# Install dependencies
apt-get update && \
apt-get install -y --no-install-recommends \
    build-essential git wget unzip ca-certificates \
    libopenblas-dev ninja-build pkg-config \
    python3-dev python3-pip \
    nlohmann-json3-dev \
    pybind11-dev \
    cmake \
    patch && \
apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Create a backup of the original CMakeLists.txt
cp CMakeLists.txt CMakeLists.txt.original

# Directly modify the CMakeLists.txt to disable installation and export commands
sed -i '"'"'s/EXPORT  alphazeroTargets/# EXPORT disabled/g'"'"' CMakeLists.txt
sed -i '"'"'/install(EXPORT alphazeroTargets/,+3d'"'"' CMakeLists.txt
sed -i '"'"'/include(GNUInstallDirs)/d'"'"' CMakeLists.txt
sed -i '"'"'/include(CMakePackageConfigHelpers)/d'"'"' CMakeLists.txt
sed -i '"'"'/configure_package_config_file/,+4d'"'"' CMakeLists.txt
sed -i '"'"'/install(TARGETS/,+5d'"'"' CMakeLists.txt
sed -i '"'"'/if(TARGET alphazero_gui)/,+3d'"'"' CMakeLists.txt
sed -i '"'"'/install(DIRECTORY/,+1d'"'"' CMakeLists.txt

# Build the tests with a clean build directory
echo "===> Building tests..."
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DALPHAZERO_ENABLE_GPU=ON \
    -DALPHAZERO_ENABLE_PYTHON=OFF \
    -DALPHAZERO_BUILD_TESTS=ON \
    -DALPHAZERO_BUILD_EXAMPLES=OFF
    
make -j$(nproc)

# Run the tests
echo "===> Running tests..."
ctest -V || echo "CTest execution failed, but continuing to run individual tests"

# Find and show all test executables
echo ""
echo "===> Test Executables:"
find_cmd="find . -name \"*_tests\" -type f -executable"
if $find_cmd | grep -q .; then
  echo "Found test executables:"
  $find_cmd
  
  echo -e "\n===> Running Test Executables Directly:"
  for test in $($find_cmd); do
    echo -e "\nRunning: $test"
    $test
  done
else
  echo "No test executables found."
  
  # Look for any executables that might be tests
  echo -e "\n===> Looking for any executables in the tests directory:"
  find ./tests -type f -executable
fi

# Restore original CMakeLists.txt
cd /src
mv CMakeLists.txt.original CMakeLists.txt

echo -e "\n===> Build directories content:"
find build -name "*test*" | sort
'

echo "===> Tests completed!" 