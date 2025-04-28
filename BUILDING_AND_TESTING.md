# Building and Testing the AlphaZero Multi-Game AI Engine

This guide provides detailed instructions for building, testing, and running the AlphaZero Multi-Game AI Engine. The implementation supports Gomoku, Chess, and Go, with this version primarily focused on Gomoku.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building the Project](#building-the-project)
3. [Running Tests](#running-tests)
4. [Using the Command-Line Interface](#using-the-command-line-interface)
5. [Running Examples](#running-examples)
6. [Self-Play and Training](#self-play-and-training)
7. [Docker Support](#docker-support)
8. [GPU Acceleration](#gpu-acceleration)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Prerequisites

### Required Dependencies

- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- CUDA Toolkit 11.0+ (for GPU support)
- cuDNN 8+ (for optimal neural network performance)
- PyTorch 2.0+ (both the C++ LibTorch and Python PyTorch)
- Python 3.8+ (for Python bindings and training tools)

### Optional Dependencies

- Ninja build system (faster builds)
- GoogleTest (for running tests, automatically downloaded if not found)
- TensorBoard (for visualizing training metrics)

### Installing Dependencies on Ubuntu

```bash
# Install essential build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev python3-pip

# Install CUDA and cuDNN
# Please follow NVIDIA's official instructions for CUDA 12.6 and cuDNN 8
# https://developer.nvidia.com/cuda-downloads
# https://developer.nvidia.com/cudnn

# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install LibTorch (C++ PyTorch)
# Download the appropriate version from https://pytorch.org/get-started/locally/
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip -d /opt/
export Torch_DIR=/opt/libtorch

# Install other Python dependencies
pip3 install numpy matplotlib tqdm tensorboard
```

### Installing Dependencies on Windows

We recommend using [vcpkg](https://github.com/microsoft/vcpkg) for dependency management on Windows:

```powershell
# Clone vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat

# Install dependencies
./vcpkg install libtorch:x64-windows nlohmann-json:x64-windows gtest:x64-windows

# Set environment variables
set Torch_DIR=C:/vcpkg/installed/x64-windows/share/libtorch
```

For CUDA on Windows, install the CUDA Toolkit and cuDNN following NVIDIA's official instructions.

## Building the Project

### Using the Build Script

The easiest way to build the project is to use the provided build script:

```bash
# Clone the repository
git clone https://github.com/yourusername/alphazero.git
cd alphazero

# Make the build script executable
chmod +x scripts/build.sh

# Build in release mode
./scripts/build.sh --release

# Build with specific options
./scripts/build.sh --debug --disable-gpu --torch-dir /path/to/libtorch
```

### Manual Build with CMake

If you prefer to use CMake directly:

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DTORCH_DIR=/path/to/libtorch

# Build
make -j$(nproc)
```

### Build Options

The following CMake options are available:

- `ALPHAZERO_ENABLE_GPU`: Enable GPU support (default: ON)
- `ALPHAZERO_BUILD_TESTS`: Build tests (default: ON)
- `ALPHAZERO_ENABLE_PYTHON`: Build Python bindings (default: ON)
- `ALPHAZERO_BUILD_EXAMPLES`: Build example programs (default: ON)
- `TORCH_DIR`: Path to LibTorch installation

## Running Tests

After building the project with tests enabled, you can run them:

```bash
# Run all tests
cd build
ctest

# Run specific test suites
./tests/core_tests
./tests/game_tests
./tests/mcts_tests
./tests/nn_tests
./tests/integration_tests
```

The tests use GoogleTest and should provide detailed output about passed and failed tests.

### Test Coverage

To build with test coverage information:

```bash
./scripts/build.sh --debug --coverage
cd build
make test_coverage
```

This will generate coverage reports in the `build/coverage` directory.

## Using the Command-Line Interface

The AlphaZero CLI provides a command-line interface for playing against the AI and running benchmarks:

```bash
# Play Gomoku against the AI
./build/bin/alphazero_cli --game gomoku --board-size 15 --simulations 800 --threads 4

# Play with Renju rules
./build/bin/alphazero_cli --game gomoku --board-size 15 --simulations 800 --threads 4 --use-renju

# Use a specific neural network model
./build/bin/alphazero_cli --game gomoku --model path/to/model.pt

# Run a benchmark
./build/bin/alphazero_cli --game gomoku --benchmark 10
```

### CLI Options

- `--game`: Game type (gomoku, chess, go)
- `--board-size`: Board size (default depends on game)
- `--simulations`: Number of MCTS simulations per move
- `--threads`: Number of parallel threads for MCTS
- `--model`: Path to neural network model
- `--use-gpu`: Use GPU for neural network inference
- `--use-renju`: Use Renju rules for Gomoku
- `--benchmark`: Run benchmark with specified number of games
- `--selfplay`: Generate self-play games for training
- `--output`: Output file pattern for self-play games

## Running Examples

The project includes several example programs demonstrating specific components:

```bash
# Run Gomoku example
./build/bin/examples/gomoku_example

# Run MCTS with Neural Network example
./build/bin/examples/test_mcts_nn --simulations 400 --threads 2
```

## Self-Play and Training

The AlphaZero engine can generate self-play games for training and then train the neural network:

### Generating Self-Play Games

```bash
# Using the Python script (recommended)
python3 python/alphazero_self_play.py \
  --executable ./build/bin/alphazero_cli \
  --output-dir data \
  --games 100 \
  --workers 4 \
  --game gomoku \
  --board-size 15 \
  --simulations 800 \
  --threads 2

# Directly using the CLI
./build/bin/alphazero_cli \
  --game gomoku \
  --board-size 15 \
  --simulations 800 \
  --threads 4 \
  --selfplay \
  --output "data/game_{}.json"
```

### Training the Neural Network

```bash
python3 python/alphazero_train.py \
  --data-dir data/games_gomoku_20250428_123456 \
  --output-dir models \
  --game gomoku \
  --board-size 15 \
  --epochs 20
```

### Training Pipeline

For a complete training pipeline that iterates between self-play and training:

```bash
./scripts/run_alphazero_pipeline.sh \
  --iterations 10 \
  --games 100 \
  --board-size 15 \
  --simulations 800
```

## Docker Support

The project includes a Dockerfile for containerized building and running:

```bash
# Build Docker image
docker build -t alphazero .

# Run the container
docker run -it --gpus all alphazero

# Run a specific command
docker run -it --gpus all alphazero ./build/bin/alphazero_cli --game gomoku
```

### Docker Compose

For more complex setups, a Docker Compose file is also provided:

```bash
docker-compose up
```

## GPU Acceleration

The AlphaZero engine supports GPU acceleration for neural network inference and training:

### Requirements

- CUDA 11.0+ with compatible GPU
- cuDNN 8+
- GPU-enabled PyTorch/LibTorch

### Enabling GPU Support

GPU support is enabled by default. To verify it's working:

```bash
# Check if GPU is being used
./build/bin/alphazero_cli --game gomoku --use-gpu --simulations 100 --verbose
```

The output should indicate that the neural network is using a CUDA device.

### Performance Tuning

For optimal GPU performance:

1. Increase the batch size for neural network inference
2. Use more MCTS threads to keep the GPU busy
3. Ensure your GPU has enough memory for the network

## Troubleshooting

### Common Issues

#### LibTorch Not Found

```
CMake Error: Could not find LibTorch at /path/to/libtorch
```

Solution:
- Set the `TORCH_DIR` environment variable or CMake parameter
- Check that LibTorch is properly installed and compatible with your CUDA version

#### CUDA Errors

```
CUDA error: no kernel image is available for execution on the device
```

Solution:
- Ensure your PyTorch/LibTorch version supports your CUDA version
- Check GPU compatibility with CUDA version
- Try rebuilding with `--disable-gpu` to use CPU-only mode

#### Python Binding Issues

```
ImportError: No module named pyalphazero
```

Solution:
- Ensure Python bindings were built with `ALPHAZERO_ENABLE_PYTHON=ON`
- Add the build directory to your Python path: `export PYTHONPATH=$PYTHONPATH:/path/to/alphazero/build/lib`

### Debugging

For debugging issues:

```bash
# Build in debug mode
./scripts/build.sh --debug

# Run with debug logging
./build/bin/alphazero_cli --game gomoku --verbose
```

## Contributing

We welcome contributions to the AlphaZero Multi-Game AI Engine!

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

### Style Guide

The project follows a consistent coding style:

- C++ code follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Python code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation (no tabs)

### Running Style Checks

```bash
# Format C++ code
./scripts/format.sh

# Check Python code style
pip3 install flake8
flake8 python/
```

Thank you for your interest in the AlphaZero Multi-Game AI Engine!