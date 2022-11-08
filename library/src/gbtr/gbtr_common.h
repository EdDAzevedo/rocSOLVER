#ifndef GBTR_COMMON_H
#define GBTR_COMMON_H

#include <complex>
#include <cmath>

#define USE_GPU
#ifdef USE_GPU

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "rocblas/rocblas.h"

#include "rocsolver_status.h"
typedef rocblas_handle rocsolverHandle_t;



#define GLOBAL_FUNCTION __global__
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__

#else


typedef void * hipStream_t;
typedef int rocblas_int;
typedef long rocblas_long;
typedef void * rocblas_handle;

typedef std::complex<double> rocblas_double_complex;
typedef std::complex<float>  rocblas_float_complex;

#define GLOBAL_FUNCTION 
#define SYNCTHREADS 
#define SHARED_MEMORY 
#define DEVICE_FUNCTION 
#define HOST_FUNCTION 

#endif


#define indx4f( i1,i2,i3,i4,  n1,n2,n3) \
	(indx3f(i1,i2,i3, n1,n2) + ((i4)-1)*((n1)*(n2))*(n3) )
#define indx3f( i1,i2,i3, n1,n2) \
	(indx2f(i1,i2,n1) + ((i3)-1)*((n1)*(n2)) )
#define indx2f( i1,i2, n1) \
	(((i1)-1) + ((i2)-1)*(n1))


#endif
