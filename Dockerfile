# syntax=docker/dockerfile:1
###############################################################################
# ── 1.  Heavy dependency image  ──────────────────────────────────────────────
###############################################################################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git wget unzip ca-certificates \
        libopenblas-dev ninja-build pkg-config gdb valgrind \
        python3-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Modern CMake
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel cmake==3.29.*

# PyTorch + torchvision wheels (CUDA 12.1)
ARG TORCH_VER=2.1.0
ARG CUDA_TAG=cu121
RUN pip install --no-cache-dir \
        torch==${TORCH_VER}+${CUDA_TAG} \
        torchvision==0.16.0+${CUDA_TAG} \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}

# LibTorch C++ distribution
RUN mkdir -p /opt && \
    wget -q https://download.pytorch.org/libtorch/${CUDA_TAG}/\
libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2B${CUDA_TAG}.zip \
        -O /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV TORCH_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"

# Extra Python deps + a **wheel of pybind11** to avoid any git clone later
RUN pip install --no-cache-dir \
        numpy matplotlib tqdm tensorboard \
        pybind11==2.10.4

###############################################################################
# ── 2.  Build image (compiles the project)  ─────────────────────────────────
###############################################################################
FROM deps AS build
WORKDIR /src
COPY . .

RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_DIR=${TORCH_DIR} \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DALPHAZERO_ENABLE_GPU=ON \
      -DALPHAZERO_ENABLE_PYTHON=ON \
      -DALPHAZERO_BUILD_TESTS=OFF \
      -DALPHAZERO_BUILD_EXAMPLES=OFF && \
    cmake --build build -j$(nproc) && \
    cmake --install build --prefix /opt/alphazero

###############################################################################
# ── 3.  Final runtime image  ────────────────────────────────────────────────
###############################################################################
FROM deps AS final
WORKDIR /opt/alphazero
COPY --from=build /opt/alphazero .

ENTRYPOINT ["bin/alphazero_cli"]
