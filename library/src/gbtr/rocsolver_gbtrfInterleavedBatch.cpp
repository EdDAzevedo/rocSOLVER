#include "rocsolver_gbtrfInterleavedBatch.hpp"

extern "C" {

rocblas_status rocsolverDgbtrfInterleavedBatch(rocblas_handle handle,
                                               int nb,
                                               int nblocks,
                                               double* A_,
                                               int lda,
                                               double* B_,
                                               int ldb,
                                               double* C_,
                                               int ldc,
                                               int batchCount)
{
    return (rocsolver_gbtrfInterleavedBatch_template<double>(handle, nb, nblocks, A_, lda, B_, ldb,
                                                             C_, ldc, batchCount));
};

rocblas_status rocsolverSgbtrfInterleavedBatch(rocblas_handle handle,
                                               int nb,
                                               int nblocks,
                                               float* A_,
                                               int lda,
                                               float* B_,
                                               int ldb,
                                               float* C_,
                                               int ldc,
                                               int batchCount)
{
    return (rocsolver_gbtrfInterleavedBatch_template<float>(handle, nb, nblocks, A_, lda, B_, ldb,
                                                            C_, ldc, batchCount));
};
}
