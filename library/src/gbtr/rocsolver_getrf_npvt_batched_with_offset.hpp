
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "assert.h"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

#pragma once
#ifndef ROCSOLVER_GETRF_NPVT_BATCHED_WITH_OFFSET_HPP
#define ROCSOLVER_GETRF_NPVT_BATCHED_WITH_OFFSET_HPP

#include "rocsolver_adjust_batch.hpp"
#include "rocsolver_getrf_npvt_batched.hpp"

template <typename T>
rocblas_status rocsolver_getrf_npvt_batched_with_offset(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            T* A_array[],
                                            const rocblas_int offsetA,
                                            const rocblas_int lda,
                                            rocblas_int* info,
                                            const rocblas_int batch_count)
{

 

 {
 bool const is_add = true;
 rocblas_status const istat = rocsolver_adjust_batch( handle,
                                is_add, A_array, offsetA, batch_count );
 
 bool const isok = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
 if (!isok) {
   return( istat );
   };
 };


 {
 rocblas_status const istat = rocsolver_getrf_npvt_batched( handle,
               m, n, A_array, lda, info, batch_count );
 
 bool const isok = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
 if (!isok) { return( istat ); };
 };
 

 {
 bool const is_add = false;
 rocblas_status const istat = rocsolver_adjust_batch( handle, 
			is_add, A_array, offsetA, batch_count );
    
 if (!isok) { return( istat ); };
 };

 return( rocblas_status_success );
};
