#include "rocsolver_gbtrfStridedBatched.hpp"

extern "C" {

rocblas_status rocsolverDgbtrfStridedBatched(rocblas_handle handle,
                                           int nb,
                                           int nblocks,
                                           double* A_,
                                           int lda,
                                           rocblas_stride strideA,
                                           double* B_,
                                           int ldb,
                                           rocblas_stride strideB,
                                           double* C_,
                                           int ldc,
                                           rocblas_stride strideC,
                                           int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrf_npvt_strided_batch<double>(stream, nb, nblocks, batchCount, A_, lda, strideA, B_, ldb,
                                     strideB, C_, ldc, strideC, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};
}
