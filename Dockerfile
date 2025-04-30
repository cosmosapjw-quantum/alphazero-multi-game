# syntax=docker/dockerfile:1
###############################################################################
# 1. Base image with CUDA and development tools                              #
###############################################################################
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        unzip \
        ca-certificates \
        libopenblas-dev \
        libspdlog-dev \
        ninja-build \
        pkg-config \
        gdb \
        valgrind \
        python3-dev \
        python3-pip \
        patch \
        nlohmann-json3-dev \
        libfmt-dev \
        pybind11-dev \
        libgtest-dev \
        python3-numpy \
        python3-matplotlib && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CMake and upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel cmake==3.29.*

###############################################################################
# 2. PyTorch dependencies image                                              #
###############################################################################
FROM base AS pytorch

# Install PyTorch and related packages
ARG TORCH_VER=2.1.0
ARG CUDA_TAG=cu121
RUN pip install --no-cache-dir \
        torch==${TORCH_VER}+${CUDA_TAG} \
        torchvision==0.16.0+${CUDA_TAG} \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}

# Download and extract LibTorch
RUN mkdir -p /opt && \
    wget -q https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2B${CUDA_TAG}.zip \
        -O /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

# Set environment variables for LibTorch
ENV TORCH_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH:-}"
ENV CMAKE_PREFIX_PATH="/opt/libtorch:${CMAKE_PREFIX_PATH:-}"

# Install additional Python packages
RUN pip install --no-cache-dir \
        pybind11==2.10.4 \
        tqdm \
        tensorboard \
        pytest

###############################################################################
# 3. Build stage                                                             #
###############################################################################
FROM pytorch AS build
WORKDIR /src

# Copy the source code
COPY . .

# Create a patch for library compatibility
RUN mkdir -p cmake && \
    cat <<'EOF' > cmake/Compatibility.cmake
# Make sure Torch targets exist
if(NOT TARGET Torch::Torch AND TARGET torch)
    add_library(Torch::Torch ALIAS torch)
endif()

# Make sure fmt target exists
find_package(fmt REQUIRED)

# Make sure spdlog target exists
# ---- fmt ------------------------------------------------------
    find_package(fmt QUIET)
    if(NOT TARGET fmt::fmt)
        # Fallback: header-only interface target
        find_path(FMT_INCLUDE_DIR fmt/core.h REQUIRED)
        add_library(fmt INTERFACE IMPORTED)
        target_include_directories(fmt INTERFACE ${FMT_INCLUDE_DIR})
        add_library(fmt::fmt ALIAS fmt)
    endif()
    

# Make sure nlohmann_json target exists
if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json CONFIG QUIET)
    if(NOT TARGET nlohmann_json::nlohmann_json)
        find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
        add_library(nlohmann_json INTERFACE IMPORTED)
        target_include_directories(nlohmann_json INTERFACE ${NLOHMANN_JSON_INCLUDE_DIR})
        add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    endif()
endif()
EOF

# Add the compatibility file to CMakeLists.txt
RUN sed -i '1s/^/include(cmake\/Compatibility.cmake)\n/' CMakeLists.txt

# Configure the project with CMake
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_DIR=${TORCH_DIR} \
      -DCMAKE_PREFIX_PATH="${TORCH_DIR};/usr/share/cmake;/usr/lib/x86_64-linux-gnu/cmake" \
      -DTorch_DIR=${TORCH_DIR}/share/cmake/Torch \
      -Dpybind11_DIR=/usr/share/cmake/pybind11 \
      -DALPHAZERO_ENABLE_GPU=ON \
      -DALPHAZERO_ENABLE_PYTHON=ON \
      -DALPHAZERO_BUILD_TESTS=ON \
      -DALPHAZERO_BUILD_EXAMPLES=ON \
      -DALPHAZERO_BUILD_GUI=OFF \
      -DALPHAZERO_BUILD_API=ON \
      -DALPHAZERO_BUILD_CLI=ON \
      -DALPHAZERO_USE_CUDNN=ON \
      -DALPHAZERO_DOWNLOAD_DEPENDENCIES=OFF \
      -DALPHAZERO_STANDALONE_MODE=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/alphazero \
      -DCMAKE_CUDA_ARCHITECTURES="80;86" \
      -DCMAKE_VERBOSE_MAKEFILE=ON

# Build the project
RUN cmake --build build -j$(nproc)

# Install the project
RUN cmake --install build

# Build Python bindings
RUN if [ -f python/setup.py ]; then \
      cd python && \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/src/build/lib" \
      PYTHONPATH="${PYTHONPATH}:/src/build/lib" \
      python3 setup.py build && \
      python3 setup.py install; \
    fi

###############################################################################
# 4. Final runtime image                                                     #
###############################################################################
FROM pytorch AS final
WORKDIR /app

# Copy the installed files from the build stage
COPY --from=build /opt/alphazero /app
COPY --from=build /src/build/lib/ /app/lib/
COPY --from=build /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/config

# Set environment variables
ENV LD_LIBRARY_PATH="/app/lib:/opt/libtorch/lib:${LD_LIBRARY_PATH:-}"
ENV PYTHONPATH="/app/lib:/usr/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"

# Create a simplified test runner script
RUN cat <<'EOF' > /app/run-tests.sh && chmod +x /app/run-tests.sh
#!/bin/bash
echo "Running AlphaZero tests..."
if [ -d "/app/build" ]; then
  cd /app/build && ctest --output-on-failure
else
  cd /app/lib
  for test in *_test *_tests; do
    if [ -x "$test" ]; then
      echo "Running $test..."
      ./$test
    fi
  done
fi
EOF

# Create a simplified entrypoint script
RUN cat <<'EOF' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh
#!/bin/bash
set -e

# Handle test command
if [[ "$1" == "test" || "$1" == "tests" ]]; then
  exec /app/run-tests.sh
# Handle Python test command
elif [[ "$1" == "pytest" ]]; then
  exec python3 -m pytest /app/python/tests
# Default to CLI if command starts with dash or is not executable
elif [[ "${1:0:1}" == "-" || ! -x "$1" ]]; then
  if [ -x /app/bin/alphazero_cli ]; then
    exec /app/bin/alphazero_cli "$@"
  else
    echo "ERROR: alphazero_cli not found"
    exit 1
  fi
# Otherwise execute the command directly
else
  exec "$@"
fi
EOF

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]