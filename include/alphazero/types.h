#ifndef ALPHAZERO_TYPES_H
#define ALPHAZERO_TYPES_H

// This header helps resolve conflicts between Linux and Windows pthread definitions
// when using WSL with Anaconda paths in the environment

// If we're in a WSL environment with potential Windows header conflicts,
// we need to prevent the inclusion of conflicting pthread.h
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__MINGW64__)
  // We're on Windows or using Windows-related tools
  #define _PTHREAD_H
  #define __PTHREAD_H
  
  // Define stub types to satisfy any references if needed
  #ifndef _BITS_PTHREADTYPES_H
  typedef unsigned long int pthread_t;
  typedef void* pthread_attr_t;
  typedef void* pthread_mutex_t;
  typedef void* pthread_mutexattr_t;
  typedef void* pthread_cond_t;
  typedef void* pthread_condattr_t;
  typedef void* pthread_key_t;
  typedef void* pthread_once_t;
  typedef void* pthread_rwlock_t;
  typedef void* pthread_rwlockattr_t;
  typedef void* pthread_spinlock_t;
  typedef void* pthread_barrier_t;
  typedef void* pthread_barrierattr_t;
  #endif
#else
  // Normal Linux environment - do nothing
#endif

namespace alphazero {
namespace types {
    // Add any AlphaZero specific types here if needed
}
}

#endif // ALPHAZERO_TYPES_H 