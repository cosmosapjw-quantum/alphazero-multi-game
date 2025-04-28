# syntax=docker/dockerfile:1
###############################################################################
# ── 1.  Heavy dependency image ───────────────────────────────────────────────
###############################################################################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1-A.  Core build tools + Python libs that are tiny but torch depends on.
#       Installing them via APT avoids extra wheel downloads during pip install.
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git wget unzip ca-certificates \
        libopenblas-dev ninja-build pkg-config gdb valgrind \
        python3-dev python3-pip patch \
        nlohmann-json3-dev \
        pybind11-dev \
        python3-sympy python3-networkx python3-jinja2 \
        python3-filelock python3-fsspec python3-requests && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Git over system certificates (CI friends)
RUN git config --global http.sslVerify true && \
    git config --global http.sslCAinfo /etc/ssl/certs/ca-certificates.crt

# ---------------------------------------------------------------------------
# 1-B.  Modern CMake and helper env for pip robustness
# ---------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel cmake==3.29.*

# Retry parameters – pip respects these environment vars
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# 1-C.  PyTorch + torchvision wheels (CUDA 12.1 build)
#       Use pip’s built-in retry flag in case of transient drops.
# ---------------------------------------------------------------------------
ARG TORCH_VER=2.1.0
ARG CUDA_TAG=cu121
RUN pip install --no-cache-dir --retries 5 \
        torch==${TORCH_VER}+${CUDA_TAG} \
        torchvision==0.16.0+${CUDA_TAG} \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}

# ---------------------------------------------------------------------------
# 1-D.  LibTorch C++ distribution
# ---------------------------------------------------------------------------
RUN mkdir -p /opt && \
    wget -q https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2B${CUDA_TAG}.zip \
        -O /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV TORCH_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"

# Extra Python deps + pybind11 wheel (gives CMake config files)
RUN pip install --no-cache-dir \
        numpy matplotlib tqdm tensorboard \
        pybind11==2.10.4

###############################################################################
# ── 2.  Build stage ──────────────────────────────────────────────────────────
###############################################################################
FROM deps AS build
WORKDIR /src

# Project sources
COPY . .

# Patch CMakeLists.txt (Torch alias, json via find_package, Python module)
RUN <<'EOF' > /tmp/cmake.patch
diff --git a/CMakeLists.txt b/CMakeLists.txt
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@
 find_package(Torch REQUIRED)
+# ─── Torch target compatibility ────────────────────────────────────────
+if(NOT TARGET Torch::Torch AND TARGET torch)
+    add_library(Torch::Torch ALIAS torch)
+endif()
+
-include(FetchContent)
-FetchContent_Declare(
-    nlohmann_json
-    GIT_REPOSITORY https://github.com/nlohmann/json.git
-    GIT_TAG        v3.11.2
-)
-FetchContent_MakeAvailable(nlohmann_json)
+find_package(nlohmann_json CONFIG REQUIRED)
@@
 endif()
 
+# ─── Python bindings ───────────────────────────────────────────────────
 if(ALPHAZERO_ENABLE_PYTHON)
     pybind11_add_module(pyalphazero src/pybind/python_bindings.cpp)
     target_link_libraries(pyalphazero PRIVATE alphazero_lib)
@@
 endif()
EOF
RUN patch -p1 < /tmp/cmake.patch

# Configure, build, install
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_DIR=${TORCH_DIR} \
      -DCMAKE_PREFIX_PATH="${TORCH_DIR};/usr/lib/x86_64-linux-gnu/cmake" \
      -DTorch_DIR=${TORCH_DIR}/share/cmake/Torch \
      -DALPHAZERO_ENABLE_GPU=ON \
      -DALPHAZERO_ENABLE_PYTHON=ON \
      -DALPHAZERO_BUILD_TESTS=OFF \
      -DALPHAZERO_BUILD_EXAMPLES=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_VERBOSE_MAKEFILE=ON && \
    cmake --build build -j$(nproc) && \
    cmake --install build --prefix /opt/alphazero


###############################################################################
# ── 3.  Minimal runtime image ────────────────────────────────────────────────
###############################################################################
FROM deps AS final
WORKDIR /app
COPY --from=build /opt/alphazero .
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--help"]
