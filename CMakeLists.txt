cmake_minimum_required(VERSION 3.18)
project(alphazero VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ALPHAZERO_BUILD_TESTS "Build tests" ON)
option(ALPHAZERO_ENABLE_GPU "Enable GPU support" ON)
option(ALPHAZERO_ENABLE_PYTHON "Enable Python bindings" ON)
option(ALPHAZERO_BUILD_EXAMPLES "Build examples" ON)

# Find dependencies
find_package(Threads REQUIRED)

# Find CUDA
find_package(CUDA 11.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Check for cuDNN
if(ALPHAZERO_ENABLE_GPU)
    find_library(CUDNN_LIBRARY cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
    if(NOT CUDNN_LIBRARY)
        message(WARNING "cuDNN library not found. NN performance may be reduced.")
    else()
        message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
        add_definitions(-DWITH_CUDNN)
    endif()
endif()

# Find Torch
set(TORCH_DIR "" CACHE PATH "Path to libtorch")
if(TORCH_DIR)
    set(CMAKE_PREFIX_PATH ${TORCH_DIR})
else()
    message(STATUS "TORCH_DIR not set, using system-wide installation")
endif()

find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${Torch_DIR}")

# Setup CUDA flags
if(ALPHAZERO_ENABLE_GPU)
    message(STATUS "GPU support enabled")
    set(CUDA_FLAGS "--use_fast_math")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_FLAGS})
    add_definitions(-DWITH_CUDA)
else()
    message(STATUS "GPU support disabled")
    add_definitions(-DLIBTORCH_OFF)
endif()

# Find or fetch nlohmann_json
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

# Find or fetch GoogleTest if tests are enabled
if(ALPHAZERO_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include(GoogleTest)
endif()

# Find or fetch pybind11 if Python bindings are enabled
if(ALPHAZERO_ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Collect source files
file(GLOB_RECURSE ALPHAZERO_SOURCES 
    "src/core/*.cpp"
    "src/mcts/*.cpp"
    "src/nn/*.cpp"
    "src/games/*.cpp"
    "src/selfplay/*.cpp"
    "src/elo/*.cpp"
    "src/api/*.cpp"
)

# Add CUDA sources if GPU is enabled
if(ALPHAZERO_ENABLE_GPU)
    file(GLOB_RECURSE ALPHAZERO_CUDA_SOURCES 
        "src/nn/*.cu"
    )
    list(APPEND ALPHAZERO_SOURCES ${ALPHAZERO_CUDA_SOURCES})
endif()

# Create main library
add_library(alphazero_lib ${ALPHAZERO_SOURCES})
target_include_directories(alphazero_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(alphazero_lib PUBLIC 
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
)
if(CUDNN_LIBRARY)
    target_link_libraries(alphazero_lib PUBLIC ${CUDNN_LIBRARY})
endif()

# Set C++ standard for the library
target_compile_features(alphazero_lib PUBLIC cxx_std_20)

# Create CLI executable
add_executable(alphazero_cli src/cli/cli_main.cpp)
target_link_libraries(alphazero_cli PRIVATE alphazero_lib)

# Create server executable
add_executable(alphazero_server src/api/server_main.cpp)
target_link_libraries(alphazero_server PRIVATE alphazero_lib)

# Create GUI executable if available
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/gui_main.cpp")
    add_executable(alphazero_gui src/gui/gui_main.cpp)
    target_link_libraries(alphazero_gui PRIVATE alphazero_lib)
endif()

# Python bindings
if(ALPHAZERO_ENABLE_PYTHON)
    pybind11_add_module(pyalphazero src/pybind/python_bindings.cpp)
    target_link_libraries(pyalphazero PRIVATE alphazero_lib)
endif()

# Build tests
if(ALPHAZERO_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Build examples
if(ALPHAZERO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation rules
install(TARGETS alphazero_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS alphazero_cli alphazero_server
    RUNTIME DESTINATION bin
)

if(TARGET alphazero_gui)
    install(TARGETS alphazero_gui
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY include/alphazero
    DESTINATION include
)

# Install Python module if built
if(ALPHAZERO_ENABLE_PYTHON)
    install(TARGETS pyalphazero
        LIBRARY DESTINATION lib/python
    )
endif()

# Generate package config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/alphazero-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/alphazero-config.cmake
    INSTALL_DESTINATION lib/cmake/alphazero
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/alphazero-config.cmake
    DESTINATION lib/cmake/alphazero
)