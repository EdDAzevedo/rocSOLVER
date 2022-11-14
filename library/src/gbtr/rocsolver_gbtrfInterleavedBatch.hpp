#pragma once
#ifndef ROCSOLVER_GBTRF_INTERLEAVED_BATCH
#define ROCSOLVER_GBTRF_INTERLEAVED_BATCH

#include "gbtr_common.h"
#include "gbtrf_npvt_bf.hpp"

template <typename T>
rocblas_status rocsolver_gbtrfInterleavedBatch_template(rocblas_handle handle,
                                                        int nb,
                                                        int nblocks,
                                                        T* A_,
                                                        int lda,
                                                        T* B_,
                                                        int ldb,
                                                        T* C_,
                                                        int ldc,
                                                        int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);
    rocblas_get_stream(blas_handle, &stream);

    int info = 0;
    gbtrf_npvt_bf_template<T>(stream, nb, nblocks, batchCount, A_, lda, B_, ldb, C_, ldc, &info);

    return ((info == 0) ? rocblas_status_success : rocblas_status_internal_error);
}

#endif
