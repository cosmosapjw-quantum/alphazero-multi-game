# syntax=docker/dockerfile:1

###############################################################################
# 1. Base image  –  CUDA 12.4 + cuDNN on Ubuntu 22.04
###############################################################################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

###############################################################################
# 2. System packages  (no cmake here – we install a newer one via pip later)
###############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python3-dev \
        python3-pip \
        wget \
        unzip \
        ca-certificates \
        libopenblas-dev \
        ninja-build \
        pkg-config \
        gdb \
        valgrind && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

###############################################################################
# 3. Python tooling  –  pip upgrades + latest CMake (3.29.x)
###############################################################################
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel \
        cmake==3.29.*  # provides /usr/local/bin/cmake

###############################################################################
# 4. PyTorch wheels (CUDA 12.1) – compatible with runtime 12.4
###############################################################################
ARG TORCH_VER=2.1.0
ARG CUDA_TAG=cu121
RUN pip install --no-cache-dir \
        torch==${TORCH_VER}+${CUDA_TAG} \
        torchvision==0.16.0+${CUDA_TAG} \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}

###############################################################################
# 5. LibTorch C++ distro  (same version & tag)
###############################################################################
RUN mkdir -p /opt && \
    wget -q https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2B${CUDA_TAG}.zip -O /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

ENV TORCH_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"

###############################################################################
# 6. Project-specific Python deps
###############################################################################
RUN pip install --no-cache-dir numpy matplotlib tqdm tensorboard

###############################################################################
# 7. Copy project sources
###############################################################################
WORKDIR /app
COPY . .

###############################################################################
# 8. Configure & build (clean out-of-source)
###############################################################################
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTORCH_DIR=${TORCH_DIR} \
        -DALPHAZERO_ENABLE_GPU=ON \
        -DALPHAZERO_BUILD_TESTS=ON \
        -DALPHAZERO_ENABLE_PYTHON=ON \
        -DALPHAZERO_BUILD_EXAMPLES=ON && \
    cmake --build build -j$(nproc)

###############################################################################
# 9. Unit tests
###############################################################################
RUN ctest --test-dir build --output-on-failure

###############################################################################
# 10. Health-check & entrypoint
###############################################################################
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD "/app/build/bin/alphazero_cli" --help || exit 1

COPY scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["/app/build/bin/alphazero_cli"]
