cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(AlphaZero VERSION 1.0.0 LANGUAGES CXX)

# Options
option(ALPHAZERO_ENABLE_GPU "Enable GPU support" ON)
option(ALPHAZERO_BUILD_TESTS "Build tests" ON)
option(ALPHAZERO_ENABLE_PYTHON "Enable Python bindings" ON)
option(ALPHAZERO_BUILD_EXAMPLES "Build examples" ON)
option(ALPHAZERO_BUILD_GUI "Build GUI" ON)
option(ALPHAZERO_BUILD_API "Build REST API" ON)
option(ALPHAZERO_BUILD_CLI "Build command-line interface" ON)
option(ALPHAZERO_USE_CUDNN "Use cuDNN for acceleration" ON)
option(ALPHAZERO_DOWNLOAD_DEPENDENCIES "Download dependencies automatically" OFF)
option(ALPHAZERO_STANDALONE_MODE "Build in standalone mode without full dependencies" OFF)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add -fPIC flag for all targets to enable building shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create custom CMake modules directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required packages
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# Setup include directories globally
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

# Check if we're in standalone mode for basic build without PyTorch
if(NOT ALPHAZERO_STANDALONE_MODE)
  # Find Torch package
  if(DEFINED TORCH_DIR)
    set(Torch_DIR "${TORCH_DIR}/share/cmake/Torch")
    message(STATUS "Using custom Torch directory: ${TORCH_DIR}")
  endif()
  
  # Find pybind11 for Python bindings
  if(ALPHAZERO_ENABLE_PYTHON)
    if(DEFINED pybind11_DIR)
      message(STATUS "Using pybind11 from: ${pybind11_DIR}")
    else()
      # Try to locate pybind11 using Python
      execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(PYBIND11_CMAKE_DIR)
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
        message(STATUS "Found pybind11 at: ${pybind11_DIR}")
      endif()
    endif()
    
    # Try to find pybind11
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
      message(STATUS "pybind11 found: ${pybind11_VERSION}")
    else()
      # If not found, we need to set it up differently
      if(ALPHAZERO_DOWNLOAD_DEPENDENCIES)
        message(STATUS "pybind11 not found, will download it")
        # Download pybind11 during configuration
        include(FetchContent)
        FetchContent_Declare(
          pybind11
          GIT_REPOSITORY https://github.com/pybind/pybind11.git
          GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
      else()
        message(WARNING "pybind11 not found and ALPHAZERO_DOWNLOAD_DEPENDENCIES is OFF. Python bindings will be disabled.")
        set(ALPHAZERO_ENABLE_PYTHON OFF)
      endif()
    endif()
  endif()

  # Try to find Torch, but don't require it yet
  find_package(Torch QUIET)
  
  if(Torch_FOUND)
    message(STATUS "PyTorch found: ${TORCH_VERSION}")
    message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
    message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
    
    # Add specific torch include dirs to make <torch/torch.h> work
    list(APPEND TORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}/torch/csrc/api/include")
    
    # Include torch directory explicitly to improve header discovery
    include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
    
    # Add a global definition to indicate LibTorch is available
    add_definitions(-DHAS_LIBTORCH=1)
    
    # Check for CUDA support but don't enable CUDA language yet
    if(ALPHAZERO_ENABLE_GPU)
      # Simple check for CUDA in PyTorch - looking for CUDA libraries in the Torch libraries list
      string(FIND "${TORCH_LIBRARIES}" "cuda" TORCH_CUDA_FOUND)
      if(TORCH_CUDA_FOUND GREATER -1)
        message(STATUS "CUDA support detected in PyTorch")
        add_definitions(-DUSE_CUDA)
        
        # Find cuDNN if requested
        if(ALPHAZERO_USE_CUDNN)
          # Find cuDNN using our custom module
          find_package(CUDNN)
          if(CUDNN_FOUND)
            message(STATUS "Found cuDNN: ${CUDNN_VERSION} (${CUDNN_LIBRARY})")
            message(STATUS "cuDNN include dirs: ${CUDNN_INCLUDE_DIRS}")
            
            # Add cuDNN definitions
            add_definitions(-DUSE_CUDNN)
            
            # Override PyTorch cuDNN setting
            set(ENV{USE_CUDNN} 1)
            set(USE_CUDNN 1)
            set(CUDNN_LIBRARY_PATH ${CUDNN_LIBRARY})
            set(CUDNN_INCLUDE_PATH ${CUDNN_INCLUDE_DIR})
            list(APPEND TORCH_COMPILE_DEFINITIONS USE_CUDNN=1)
            
            # Include cuDNN directories
            include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
          else()
            message(WARNING "cuDNN not found, disabling cuDNN support")
            set(ALPHAZERO_USE_CUDNN OFF)
          endif()
        endif()
      else()
        message(WARNING "CUDA not found in PyTorch, reverting to CPU-only mode")
        set(ALPHAZERO_ENABLE_GPU OFF)
      endif()
    endif()
  else()
    message(WARNING "PyTorch not found. Building in minimal mode.")
    set(ALPHAZERO_STANDALONE_MODE ON)
  endif()
endif()

# Only include src directory if we're not in standalone mode
if(NOT ALPHAZERO_STANDALONE_MODE AND Torch_FOUND)
  # Core libraries
  add_subdirectory(src)

  # CUDA test executable - check if file exists
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_test.cpp")
    add_executable(cuda_test_app cuda_test.cpp)
    target_link_libraries(cuda_test_app PRIVATE ${TORCH_LIBRARIES})
    set_target_properties(cuda_test_app PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
    )
  else()
    message(STATUS "cuda_test.cpp not found, skipping cuda_test_app target")
  endif()

  # Standalone CUDA test - check if file exists
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/standalone_cuda_test.cpp")
    add_executable(standalone_cuda_test standalone_cuda_test.cpp)
    target_link_libraries(standalone_cuda_test PRIVATE ${TORCH_LIBRARIES})
    set_target_properties(standalone_cuda_test PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
    )
  else()
    message(STATUS "standalone_cuda_test.cpp not found, skipping standalone_cuda_test target")
  endif()

  # Tests if enabled
  if(ALPHAZERO_BUILD_TESTS)
    # Try to find GTest
    find_package(GTest QUIET)
    if(GTest_FOUND)
      message(STATUS "GoogleTest found")
      include(GoogleTest)
      enable_testing()
      # We'll include tests only if GoogleTest is found
      add_subdirectory(tests)
    else()
      message(WARNING "GoogleTest not found. Tests will not be built.")
      set(ALPHAZERO_BUILD_TESTS OFF)
    endif()
  endif()

  # Installation
  set(INSTALL_TARGETS "")
  if(TARGET cuda_test_app)
    list(APPEND INSTALL_TARGETS cuda_test_app)
  endif()
  if(TARGET standalone_cuda_test)
    list(APPEND INSTALL_TARGETS standalone_cuda_test)
  endif()
  
  if(INSTALL_TARGETS)
    install(TARGETS ${INSTALL_TARGETS}
      RUNTIME DESTINATION bin
    )
  endif()
else()
  # In standalone mode, just create a dummy target
  add_executable(alphazero_minimal minimal_torch.cpp)
  set_target_properties(alphazero_minimal PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
  
  # Installation
  install(TARGETS alphazero_minimal
    RUNTIME DESTINATION bin
  )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "AlphaZero Configuration Summary:")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Standalone mode: ${ALPHAZERO_STANDALONE_MODE}")
message(STATUS "  GPU support: ${ALPHAZERO_ENABLE_GPU}")
message(STATUS "  Python bindings: ${ALPHAZERO_ENABLE_PYTHON}")
message(STATUS "  Build examples: ${ALPHAZERO_BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${ALPHAZERO_BUILD_TESTS}")
message(STATUS "") 