#include "rocsolver_gbtrsStridedBatched.hpp"

extern "C" {

rocblas_status rocsolverDgbtrsStridedBatched(rocblas_handle handle,
                                             int nb,
                                             int nblocks,
                                             int nrhs,
                                             double* A_,
                                             int lda,
                                             rocblas_stride strideA,
                                             double* B_,
                                             int ldb,
                                             rocblas_stride strideB,
                                             double* C_,
                                             int ldc,
                                             rocblas_stride strideC,

                                             double* brhs_,
                                             int ldbrhs,
                                             int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_strided_batched_template<double>(stream, nb, nblocks, nrhs, batchCount, A_, lda,
                                                strideA, B_, ldb, strideB, C_, ldc, strideC, brhs_,
                                                ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverSgbtrsStridedBatched(rocblas_handle handle,
                                             int nb,
                                             int nblocks,
                                             int nrhs,
                                             float* A_,
                                             int lda,
                                             rocblas_stride strideA,
                                             float* B_,
                                             int ldb,
                                             rocblas_stride strideB,
                                             float* C_,
                                             int ldc,
                                             rocblas_stride strideC,

                                             float* brhs_,
                                             int ldbrhs,
                                             int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_strided_batched_template<float>(stream, nb, nblocks, nrhs, batchCount, A_, lda,
                                               strideA, B_, ldb, strideB, C_, ldc, strideC, brhs_,
                                               ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverCgbtrsStridedBatched(rocblas_handle handle,
                                             int nb,
                                             int nblocks,
                                             int nrhs,
                                             rocblas_float_complex* A_,
                                             int lda,
                                             rocblas_stride strideA,
                                             rocblas_float_complex* B_,
                                             int ldb,
                                             rocblas_stride strideB,
                                             rocblas_float_complex* C_,
                                             int ldc,
                                             rocblas_stride strideC,

                                             rocblas_float_complex* brhs_,
                                             int ldbrhs,
                                             int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_strided_batched_template<rocblas_float_complex>(
        stream, nb, nblocks, nrhs, batchCount, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,
        brhs_, ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverZgbtrsStridedBatched(rocblas_handle handle,
                                             int nb,
                                             int nblocks,
                                             int nrhs,
                                             rocblas_double_complex* A_,
                                             int lda,
                                             rocblas_stride strideA,
                                             rocblas_double_complex* B_,
                                             int ldb,
                                             rocblas_stride strideB,
                                             rocblas_double_complex* C_,
                                             int ldc,
                                             rocblas_stride strideC,

                                             rocblas_double_complex* brhs_,
                                             int ldbrhs,
                                             int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_strided_batched_template<rocblas_double_complex>(
        stream, nb, nblocks, nrhs, batchCount, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,
        brhs_, ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};
}
