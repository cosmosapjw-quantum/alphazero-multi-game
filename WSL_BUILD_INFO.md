# Building on WSL with Windows Anaconda Conflicts

When building this project on WSL (Windows Subsystem for Linux) with Anaconda installed on Windows, you might encounter pthread header conflicts. This document explains the issue and provides solutions.

## The Problem

When developing in WSL with Anaconda installed on Windows, paths from Windows might leak into the Linux environment. This causes conflicts between the Linux system headers and Anaconda's Windows pthread headers, resulting in compilation errors like:

```
/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h:27:27: error: conflicting declaration 'typedef long unsigned int pthread_t'
/mnt/c/Users/user/anaconda3/Library/include/pthread.h:590:24: note: previous declaration as 'typedef struct ptw32_handle_t pthread_t'
```

## Solutions

### 1. Use the Provided Build Script

We provide a special build script for WSL environments that filters out Windows paths and sets appropriate compiler flags:

```bash
./build_linux.sh
```

This script:
- Removes Windows/Anaconda paths from the environment
- Unsets Conda-related variables
- Sets compiler flags to work around pthread conflicts
- Runs the build with a clean environment

### 2. Manual Environment Cleanup

If you prefer to manage your environment manually:

```bash
# Filter Windows/Anaconda paths
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E "anaconda|Windows|Program Files" | tr '\n' ':' | sed 's/:$//')

# Unset Conda variables
unset CONDA_PREFIX CONDA_PYTHON_EXE CONDA_DEFAULT_ENV CONDA_EXE CONDA_SHLVL
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH CPATH CMAKE_INCLUDE_PATH CMAKE_LIBRARY_PATH

# Build with clean environment
mkdir -p build && cd build
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin cmake .. -DLIBTORCH_OFF=ON
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin cmake --build . --parallel
```

### 3. Using the Types Header Protection

Our `alphazero/types.h` header includes protections against pthread conflicts. If you're adding new files to the project, make sure to:

1. Include our types header first in each file:
   ```cpp
   #include "alphazero/types.h"
   // Other includes follow...
   ```

2. Use the `-D__MINGW64__` compiler flag to trigger our header protection:
   ```bash
   g++ -D__MINGW64__ -I./include yourfile.cpp
   ```

## Detailed Troubleshooting

### Windows Anaconda Headers Path Issue

The root of the issue is that WSL includes Windows paths in its environment, and when Anaconda is installed on Windows, the system finds Anaconda's Windows version of pthread.h before the Linux version. 

To see if this is the problem you're facing, run:

```bash
echo $PATH | tr ':' '\n' | grep anaconda
```

If this shows Windows Anaconda paths (like `/mnt/c/Users/username/anaconda3/`), they need to be removed.

### Common Error Messages

1. **Windows/Linux pthread.h conflict**:
   ```
   /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h:27:27: error: conflicting declaration 'typedef long unsigned int pthread_t'
   /mnt/c/Users/user/anaconda3/Library/include/pthread.h:590:24: note: previous declaration as 'typedef struct ptw32_handle_t pthread_t'
   ```

2. **Windows includes not found in Linux**:
   ```
   fatal error: windows.h: No such file or directory
   ```

3. **Include path issues with system headers**:
   ```
   fatal error: stdlib.h: No such file or directory
   ```

### Solutions for Header Errors

If you encounter errors related to standard C/C++ headers not being found:

1. Make sure your include paths are set correctly:
   ```bash
   export C_INCLUDE_PATH=/usr/include:/usr/include/x86_64-linux-gnu
   export CPLUS_INCLUDE_PATH=/usr/include:/usr/include/x86_64-linux-gnu:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11
   ```

2. If you're using CMake, you can specify include directories:
   ```
   cmake .. -DCMAKE_CXX_FLAGS="-I/usr/include -I/usr/include/x86_64-linux-gnu"
   ```

## Additional Tips

- If using VS Code Remote WSL, you might need to modify your settings to filter Windows paths
- Consider creating a dedicated WSL environment without Windows path integration
- Use WSL2 when possible as it has better isolation from Windows

If you continue to experience issues, please report them with details about your environment. 