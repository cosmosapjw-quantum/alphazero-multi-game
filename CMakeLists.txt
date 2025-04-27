# Update CMakeLists.txt to include new components
cmake_minimum_required(VERSION 3.10)
project(AlphaZero LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find required packages
find_package(Threads REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(json)
endif()

# Option to build with PyBind11 (for Python bindings)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Find or fetch pybind11
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.10.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    # Add Python include directories
    include_directories(${Python3_INCLUDE_DIRS})
endif()

# Option to build with LibTorch (for neural network)
option(LIBTORCH_OFF "Disable LibTorch" ON)
if(NOT LIBTORCH_OFF)
    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
        message(FATAL_ERROR "LibTorch not found. Please install LibTorch or set Torch_DIR.")
    endif()
endif()

# Collect source files
file(GLOB_RECURSE CORE_SOURCES 
    "src/core/*.cpp"
)

file(GLOB_RECURSE GAMES_SOURCES 
    "src/games/*.cpp"
)

file(GLOB_RECURSE MCTS_SOURCES 
    "src/mcts/*.cpp"
)

file(GLOB_RECURSE NN_SOURCES 
    "src/nn/*.cpp"
)

file(GLOB_RECURSE CLI_SOURCES 
    "src/cli/*.cpp"
)

file(GLOB_RECURSE API_SOURCES 
    "src/api/*.cpp"
)

file(GLOB_RECURSE SELFPLAY_SOURCES 
    "src/selfplay/*.cpp"
)

file(GLOB_RECURSE UI_SOURCES 
    "src/ui/*.cpp"
)

file(GLOB_RECURSE ELO_SOURCES 
    "src/elo/*.cpp"
)

# Create main library
add_library(alphazero_lib STATIC
    ${CORE_SOURCES}
    ${GAMES_SOURCES}
    ${MCTS_SOURCES}
    ${NN_SOURCES}
    ${SELFPLAY_SOURCES}
    ${ELO_SOURCES}
)

target_link_libraries(alphazero_lib
    PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
)

if(NOT LIBTORCH_OFF)
    target_link_libraries(alphazero_lib PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(alphazero_lib PRIVATE USE_TORCH)
endif()

# Create CLI executable
add_executable(alphazero_cli
    ${CLI_SOURCES}
    src/cli/cli_main.cpp
)

target_link_libraries(alphazero_cli
    PRIVATE
    alphazero_lib
)

# Create server executable if API sources exist
if(API_SOURCES)
    add_executable(alphazero_server
        ${API_SOURCES}
        src/api/server_main.cpp
    )
    
    target_link_libraries(alphazero_server
        PRIVATE
        alphazero_lib
    )
endif()

# Create GUI executable if UI sources exist
if(UI_SOURCES)
    add_executable(alphazero_gui
        ${UI_SOURCES}
        src/gui/gui_main.cpp
    )
    
    target_link_libraries(alphazero_gui
        PRIVATE
        alphazero_lib
    )
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(pyalphazero
        src/pybind/python_bindings.cpp
    )
    
    target_link_libraries(pyalphazero
        PRIVATE
        alphazero_lib
    )
    
    # Add compile definition for Python bindings
    target_compile_definitions(alphazero_lib PRIVATE BUILD_PYTHON_BINDINGS)
endif()

# Examples
add_subdirectory(examples)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Find or fetch GTest
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found - fetching from GitHub")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.11.0
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        
        # Set up modern CMake targets if using downloaded version
        if(NOT DEFINED GTest::GTest)
            add_library(GTest::GTest ALIAS gtest)
            add_library(GTest::Main ALIAS gtest_main)
        endif()
    endif()
    
    message(STATUS "GTest configuration:")
    message(STATUS "  GTest_FOUND: ${GTest_FOUND}")
    message(STATUS "  GTEST_INCLUDE_DIRS: ${GTEST_INCLUDE_DIRS}")
    message(STATUS "  GTEST_LIBRARIES: ${GTEST_LIBRARIES}")
    
    include_directories(${GTEST_INCLUDE_DIRS})
    
    add_subdirectory(tests)
endif()

# Install targets
install(TARGETS alphazero_lib alphazero_cli
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(API_SOURCES)
    install(TARGETS alphazero_server
        RUNTIME DESTINATION bin
    )
endif()

if(UI_SOURCES)
    install(TARGETS alphazero_gui
        RUNTIME DESTINATION bin
    )
endif()

# Install headers
install(DIRECTORY include/
    DESTINATION include
)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python Bindings: ${BUILD_PYTHON_BINDINGS}")
if(LIBTORCH_OFF)
    message(STATUS "  Using LibTorch: OFF")
else()
    message(STATUS "  Using LibTorch: ON")
endif()
message(STATUS "  Build Tests: ${BUILD_TESTS}")