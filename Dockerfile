# syntax=docker/dockerfile:1
###############################################################################
# 1. Heavy dependency image (toolchains + Python + CUDA)                      #
###############################################################################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive

# ── 1-A. Core build tools and Python libs that PyTorch needs
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

# Git over system certificates (useful in CI)
RUN git config --global http.sslVerify true && \
    git config --global http.sslCAinfo /etc/ssl/certs/ca-certificates.crt

# ── 1-B. Modern CMake and pip robustness flags
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel cmake==3.29.*

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── 1-C. PyTorch & torchvision wheels (CUDA 12.1 build)
ARG TORCH_VER=2.1.0
ARG CUDA_TAG=cu121
RUN pip install --no-cache-dir --retries 5 \
        torch==${TORCH_VER}+${CUDA_TAG} \
        torchvision==0.16.0+${CUDA_TAG} \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}

# ── 1-D. LibTorch C++ distribution (same version)
RUN mkdir -p /opt && \
    wget -q https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2B${CUDA_TAG}.zip \
        -O /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV TORCH_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"

# ── 1-E. Extra Python utilities + pybind11 wheel (for headers only)
RUN pip install --no-cache-dir \
        numpy matplotlib tqdm tensorboard \
        pybind11==2.10.4


###############################################################################
# 2. Build stage (compiles the project)                                       #
###############################################################################
FROM deps AS build
WORKDIR /src

# Copy project sources
COPY . .

# ── Patch CMakeLists.txt once: Torch alias, drop all FetchContent, add module
RUN <<'EOF' > /tmp/az_patch.diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@
 find_package(Torch REQUIRED)
+# ─── Torch-target compatibility (torch ≥ 2.0 dropped Torch::Torch) ──────────
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
+# Find nlohmann_json system package or create an imported target
+if(NOT TARGET nlohmann_json::nlohmann_json)
+    find_package(nlohmann_json CONFIG QUIET)
+endif()
+
+# If not found via find_package, create an imported target manually
+if(NOT TARGET nlohmann_json::nlohmann_json)
+    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
+    add_library(nlohmann_json INTERFACE IMPORTED)
+    target_include_directories(nlohmann_json INTERFACE ${NLOHMANN_JSON_INCLUDE_DIR})
+endif()
+
+# Create alias for compatibility
+if(TARGET nlohmann_json::nlohmann_json AND NOT TARGET nlohmann_json)
+    add_library(nlohmann_json ALIAS nlohmann_json::nlohmann_json)
+endif()
@@
-    find_package(Python COMPONENTS Interpreter Development REQUIRED)
-    FetchContent_Declare(
-        pybind11
-        GIT_REPOSITORY https://github.com/pybind11/pybind11.git
-        GIT_TAG        v2.10.4
-    )
-    FetchContent_MakeAvailable(pybind11)    # target pybind11::pybind11
+    find_package(Python   COMPONENTS Interpreter Development REQUIRED)
+    find_package(pybind11 CONFIG REQUIRED)   # provided by pybind11-dev
+endif()
+
+# Remove any other potential FetchContent calls for pybind11
+if(DEFINED FETCHCONTENT_BASE_DIR AND EXISTS "${FETCHCONTENT_BASE_DIR}/pybind11-subbuild")
+    message(STATUS "Removing existing pybind11 FetchContent directory")
+    file(REMOVE_RECURSE "${FETCHCONTENT_BASE_DIR}/pybind11-subbuild")
 endif()
@@
 endif()
EOF
RUN set -e; \
    patch -p1 < /tmp/az_patch.diff; \
    echo "Patch command exited with status $?"; \
    echo "--- Verifying CMakeLists.txt patch ---"; \
    if grep pybind11 CMakeLists.txt; then \
        echo "grep found pybind11, exited with status $?"; \
    else \
        echo "grep found no pybind11, exited with status $?"; \
        # Decide if no pybind11 means success or failure. Here it means success. \
        # If it meant failure, we would add `exit 1` here. \
    fi; \
    echo "------------------------------------"

# ── Configure, build, install
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_DIR=${TORCH_DIR} \
      -DCMAKE_PREFIX_PATH="${TORCH_DIR};/usr/share/cmake" \
      -DTorch_DIR=${TORCH_DIR}/share/cmake/Torch \
      -Dpybind11_DIR=/usr/share/cmake/pybind11 \
      -DALPHAZERO_ENABLE_GPU=ON \
      -DALPHAZERO_ENABLE_PYTHON=OFF \
      -DALPHAZERO_BUILD_TESTS=OFF \
      -DALPHAZERO_BUILD_EXAMPLES=OFF \
      -DCMAKE_SKIP_INSTALL_EXPORT=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_VERBOSE_MAKEFILE=ON && \
    cmake --build build -j$(nproc) && \
    mkdir -p /opt/alphazero/bin /opt/alphazero/lib && \
    find ./build -name "*.so" -type f -exec cp {} /opt/alphazero/lib/ \; && \
    find ./build -type f -executable -not -name "*.so" -exec cp {} /opt/alphazero/bin/ \; || true && \
    cp -r include /opt/alphazero/ || true


###############################################################################
# 3. Minimal runtime image                                                    #
###############################################################################
FROM deps AS final
WORKDIR /app
COPY --from=build /opt/alphazero .
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--help"]
