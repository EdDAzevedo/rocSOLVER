#include "rocsolver_gbtrsBatched.hpp"

extern "C" {

rocblas_status rocsolverDgbtrsBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      int nrhs,
                                      double* A_array[],
                                      int lda,
                                      double* B_array[],
                                      int ldb,
                                      double* C_array[],
                                      int ldc,
                                      double* brhs_,
                                      int ldbrhs,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_batched_template<double>(stream, nb, nblocks, nrhs, batchCount, A_array, lda,
                                        B_array, ldb, C_array, ldc, brhs_, ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverSgbtrsBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      int nrhs,
                                      float* A_array[],
                                      int lda,
                                      float* B_array[],
                                      int ldb,
                                      float* C_array[],
                                      int ldc,
                                      float* brhs_,
                                      int ldbrhs,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_batched_template<float>(stream, nb, nblocks, nrhs, batchCount, A_array, lda, B_array,
                                       ldb, C_array, ldc, brhs_, ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverCgbtrsBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      int nrhs,
                                      rocblas_float_complex* A_array[],
                                      int lda,
                                      rocblas_float_complex* B_array[],
                                      int ldb,
                                      rocblas_float_complex* C_array[],
                                      int ldc,
                                      rocblas_float_complex* brhs_,
                                      int ldbrhs,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_batched_template<rocblas_float_complex>(stream, nb, nblocks, nrhs, batchCount,
                                                       A_array, lda, B_array, ldb, C_array, ldc,
                                                       brhs_, ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverZgbtrsBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      int nrhs,
                                      rocblas_double_complex* A_array[],
                                      int lda,
                                      rocblas_double_complex* B_array[],
                                      int ldb,
                                      rocblas_double_complex* C_array[],
                                      int ldc,
                                      rocblas_double_complex* brhs_,
                                      int ldbrhs,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_handle blas_handle(handle);

    int host_info = 0;
    gbtrs_npvt_batched_template<rocblas_double_complex>(stream, nb, nblocks, nrhs, batchCount,
                                                        A_array, lda, B_array, ldb, C_array, ldc,
                                                        brhs_, ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};
}
