# FindCUDNN.cmake
#
# Finds the cuDNN library and includes
# This module defines:
#  CUDNN_INCLUDE_DIRS - where to find cudnn.h
#  CUDNN_LIBRARIES    - the libraries to link against (including dependencies)
#  CUDNN_FOUND        - if false, cuDNN was not found
#  CUDNN_VERSION      - cuDNN version

# First check for the CUDNN header
set(CUDNN_INCLUDE_DIRS
  ${CUDNN_INCLUDE_DIR}
  $ENV{CUDNN_INCLUDE_DIR}
  /usr/include
  /usr/local/include
  /usr/local/cuda/include
  /usr/local/cuda-*/include
  /opt/cuda/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# Find the include file
find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDNN_INCLUDE_DIRS})

# Find the library
set(CUDNN_LIBRARY_PATHS
  ${CUDNN_LIBRARY}
  $ENV{CUDNN_LIBRARY}
  /usr/lib
  /usr/lib64
  /usr/local/lib
  /usr/local/lib64
  /usr/local/cuda/lib
  /usr/local/cuda/lib64
  /usr/local/cuda-*/lib
  /usr/local/cuda-*/lib64
  /opt/cuda/lib
  /opt/cuda/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# First look for exact file
find_library(CUDNN_LIBRARY cudnn PATHS ${CUDNN_LIBRARY_PATHS})

# If not found, look for any version of libcudnn
if(NOT CUDNN_LIBRARY)
  file(GLOB_RECURSE CUDNN_LIBRARIES /usr/lib*/libcudnn.so*)
  if(CUDNN_LIBRARIES)
    list(GET CUDNN_LIBRARIES 0 CUDNN_LIBRARY)
  endif()
endif()

# If still not found, try with -lcudnn
if(NOT CUDNN_LIBRARY)
  set(CUDNN_LIBRARY "cudnn")
endif()

# Determine version
if(CUDNN_INCLUDE_DIR)
  file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_H_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_MAJOR_VERSION ${CMAKE_MATCH_1})
  string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_MINOR_VERSION ${CMAKE_MATCH_1})
  string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_PATCH_VERSION ${CMAKE_MATCH_1})
  set(CUDNN_VERSION "${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCH_VERSION}")
endif()

# Handle the QUIETLY and REQUIRED arguments and set CUDNN_FOUND to TRUE if
# all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
  REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
  VERSION_VAR CUDNN_VERSION
)

if(CUDNN_FOUND)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)