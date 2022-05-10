#ifndef GEBLTTRF_COMMON_HPP
#define  GEBLTTRF_COMMON_HPP 1

#include <stdlib.h>
#include <assert.h>
#ifdef USE_CPU


#define DEVICE_FUNCTION 
#define GLOBAL_FUNCTION 
#define HOST_FUNCTION 
#define SYNCTHREADS 
typedef int rocblas_int;
typedef int rocblas_status;
typedef void * rocblas_handle;
typedef long rocblas_stride;

#define rocblas_status_success 0

#else
#include "rocblas.hpp"
#define DEVICE_FUNCTION __device__
#define GLOBAL_FUNCTION __global__
#define HOST_FUNCTION  __host__
#define SYNCTHREADS { __syncthreads(); }
#endif

#endif
