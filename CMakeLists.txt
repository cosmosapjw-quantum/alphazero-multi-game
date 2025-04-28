cmake_minimum_required(VERSION 3.18)
project(alphazero VERSION 1.0.0 LANGUAGES CXX CUDA)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Global build flags
# ──────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All object files must be position-independent so they can be linked into
# shared libraries (e.g. the Python extension).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Build-time options
# ──────────────────────────────────────────────────────────────────────────────
option(ALPHAZERO_BUILD_TESTS    "Build GoogleTest targets"          ON)
option(ALPHAZERO_ENABLE_GPU     "Enable CUDA kernels"               ON)
option(ALPHAZERO_ENABLE_PYTHON  "Build pybind11 module"             ON)
option(ALPHAZERO_BUILD_EXAMPLES "Build small CLI examples"          ON)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Dependencies
# ──────────────────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# --- CUDA --------------------------------------------------------------------
if(ALPHAZERO_ENABLE_GPU)
    # We rely on the built-in CUDA language support; FindCUDA is deprecated.
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(WITH_CUDA)
else()
    add_compile_definitions(LIBTORCH_OFF)
endif()

# Optional cuDNN: warn if missing but keep going.
if(ALPHAZERO_ENABLE_GPU)
    find_library(CUDNN_LIBRARY cudnn
                 HINTS ${CUDAToolkit_LIBRARY_DIR}
                 PATH_SUFFIXES lib lib64)
    if(CUDNN_LIBRARY)
        message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
        add_compile_definitions(WITH_CUDNN)
    else()
        message(WARNING "cuDNN library not found – GPU inference will be slower.")
    endif()
endif()

# --- LibTorch ----------------------------------------------------------------
set(TORCH_DIR "" CACHE PATH "Path to a pre-downloaded libtorch distribution")
if(TORCH_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
else()
    message(STATUS "TORCH_DIR not set – trying system-wide Torch install.")
endif()
find_package(Torch REQUIRED)        # provides imported target Torch::Torch
message(STATUS "Torch include root: ${Torch_INCLUDE_DIRS}")

# ─── Torch target compatibility ────────────────────────────────────────────────
# Libtorch ≥ 2.0 no longer exports Torch::Torch; use torch instead.
# Keep projects that still refer to Torch::Torch working by creating an alias.
if(NOT TARGET Torch::Torch AND TARGET torch)
    add_library(Torch::Torch ALIAS torch)
endif()

# --- Header-only nlohmann_json ----------------------------------------------
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)   # target nlohmann_json::nlohmann_json

# --- GoogleTest (optional) ---------------------------------------------------
if(ALPHAZERO_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # On MSVC avoid /MDd vs /MD clashes
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include(GoogleTest)
endif()

# --- pybind11 (optional) -----------------------------------------------------
if(ALPHAZERO_ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind11/pybind11.git
        GIT_TAG        v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)    # target pybind11::pybind11
endif()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Source files
# ──────────────────────────────────────────────────────────────────────────────
file(GLOB_RECURSE ALPHAZERO_CPP_SOURCES
     src/core/*.cpp
     src/mcts/*.cpp
     src/nn/*.cpp
     src/games/*.cpp
     src/selfplay/*.cpp
     src/elo/*.cpp
     src/api/*.cpp)

if(ALPHAZERO_ENABLE_GPU)
    file(GLOB_RECURSE ALPHAZERO_CUDA_SOURCES src/nn/*.cu)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Core library
# ──────────────────────────────────────────────────────────────────────────────
add_library(alphazero_lib
            ${ALPHAZERO_CPP_SOURCES}
            ${ALPHAZERO_CUDA_SOURCES})

target_include_directories(alphazero_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(alphazero_lib PUBLIC
    Torch::Torch
    nlohmann_json::nlohmann_json
    Threads::Threads)

if(ALPHAZERO_ENABLE_GPU AND CUDNN_LIBRARY)
    target_link_libraries(alphazero_lib PUBLIC ${CUDNN_LIBRARY})
endif()

target_compile_features(alphazero_lib PUBLIC cxx_std_20)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Executables
# ──────────────────────────────────────────────────────────────────────────────
add_executable(alphazero_cli     src/cli/cli_main.cpp)
add_executable(alphazero_server  src/api/server_main.cpp)
target_link_libraries(alphazero_cli    PRIVATE alphazero_lib)
target_link_libraries(alphazero_server PRIVATE alphazero_lib)

# Optional GUI (if the file exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/gui_main.cpp")
    add_executable(alphazero_gui src/gui/gui_main.cpp)
    target_link_libraries(alphazero_gui PRIVATE alphazero_lib)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# 7. Python bindings (pybind11)
# ──────────────────────────────────────────────────────────────────────────────
# ── pybind11 (optional) ─────────────────────────────────────────────────────
if(ALPHAZERO_ENABLE_PYTHON)
    # Try to use the wheel we pre-installed in the deps layer
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Fallback: clone with FetchContent (only when the wheel is absent)
        message(STATUS "pybind11 wheel not found – fetching from GitHub")
        find_package(Python COMPONENTS Interpreter Development REQUIRED)

        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind11/pybind11.git
            GIT_TAG        v2.10.4
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()


# ──────────────────────────────────────────────────────────────────────────────
# 8. Tests & examples
# ──────────────────────────────────────────────────────────────────────────────
if(ALPHAZERO_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(ALPHAZERO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# 9. Installation
# ──────────────────────────────────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS alphazero_lib
        EXPORT  alphazeroTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS alphazero_cli alphazero_server
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(TARGET alphazero_gui)
    install(TARGETS alphazero_gui
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY include/alphazero
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ALPHAZERO_ENABLE_PYTHON)
    install(TARGETS pyalphazero
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# 10. Package-config files for find_package(alphazero)
# ──────────────────────────────────────────────────────────────────────────────
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/alphazero-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/alphazero-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alphazero
)

install(EXPORT alphazeroTargets
        FILE    alphazeroTargets.cmake
        NAMESPACE alphazero::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alphazero)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/alphazero-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alphazero)
