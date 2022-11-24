/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas.hpp"
#include "assert.h"

#pragma once
template< typename T>
rocsolver_status
rocsolver_getrf_npvt_strided_batched(
	rocblas_handle handle,
        rocblas_int const m,
        rocblas_int const n,
        T* const A_,
        rocblas_int const lda,
        rocblas_stride const strideA,
        rocblas_int info_array[],
        rocblas_int batch_count )
{
 assert( false );
 return( rocblas_status_internal_error );
};
        

template<>
rocsolver_status
rocsolver_getrf_npvt_strided_batched(
	rocblas_handle handle,
        rocblas_int const m,
        rocblas_int const n,
        double* const A_,
        rocblas_int const lda,
        rocblas_stride const strideA,
        rocblas_int info_array[],
        rocblas_int batch_count )
{
  return( rocsolver_dgetrf_npvt_strided_batched(
              handle,
              m, n,     A,lda,strideA,   
              info_array, batch_count )
        );
}


