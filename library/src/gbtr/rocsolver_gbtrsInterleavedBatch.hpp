#pragma once
#ifndef ROCSOLVER_GBTRS_INTERLEAVED_BATCH
#define ROCSOLVER_GBTRS_INTERLEAVED_BATCH

#include "gbtr_common.h"
#include "gbtrs_npvt_bf.hpp"


template<typename T>
rocsolverStatus_t  rocsolver_gbtrsInterleavedBatch_template(
                     rocsolverHandle_t handle,
                     int nb,
                     int nblocks,
                     const T* A_,
                     int lda,
                     const T* B_,
                     int ldb,
                     const T* C_,
                     int ldc,
                     T* brhs_,
                     int ldbrhs,
                     int batchCount
                     )
{
 hipStream_t stream;
 rocblas_handle blas_handle( handle );
 rocblas_get_stream( blas_handle , &stream );
 
 int info = 0;
 int nrhs = 1;

 gbtrs_npvt_bf_template<T>(
      stream,
      nb,
      nblocks,
      batchCount,
      nrhs,
      A_,
      lda,
      B_, 
      ldb,
      C_,
      ldc,
      brhs_,
      ldbrhs,
      &info );


 return(  (info == 0) ? ROCSOLVER_STATUS_SUCCESS :
                        ROCSOLVER_STATUS_INTERNAL_ERROR );


}




#endif
