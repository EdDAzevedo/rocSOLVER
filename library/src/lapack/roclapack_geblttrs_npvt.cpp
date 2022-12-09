/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_geblttrs_npvt.hpp"

template <typename T, typename U>
rocblas_status rocsolver_geblttrs_npvt_impl(rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            const rocblas_int nrhs,
                                            U A,
                                            const rocblas_int lda,
                                            U B,
                                            const rocblas_int ldb,
                                            U C,
                                            const rocblas_int ldc,
                                            U X,
                                            const rocblas_int ldx)
{
    ROCSOLVER_ENTER_TOP("geblttrs_npvt", "--nb", nb, "--nblocks", nblocks, "--nrhs", nrhs, "--lda",
                        lda, "--ldb", ldb, "--ldc", ldc, "--ldx", ldx);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_geblttrs_npvt_argCheck(handle, nb, nblocks, nrhs, lda, ldb, ldc,
                                                         ldx, A, B, C, X);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftC = 0;
    rocblas_int shiftX = 0;

    // normal execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideC = 0;
    rocblas_stride strideX = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace
    size_t size_work;

    rocsolver_geblttrs_npvt_getMemorySize<false, false, T>(nb, nblocks, nrhs, batch_count,
                                                           &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;
    work = mem[0];

    // Execution
    return rocsolver_geblttrs_npvt_template<false, false, T>(
        handle, nb, nblocks, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, C, shiftC, ldc,
        strideC, X, shiftX, ldx, strideX, batch_count, work);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        float* A,
                                        const rocblas_int lda,
                                        float* B,
                                        const rocblas_int ldb,
                                        float* C,
                                        const rocblas_int ldc,
                                        float* X,
                                        const rocblas_int ldx)
{
    return rocsolver_geblttrs_npvt_impl<float>(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X,
                                               ldx);
}

rocblas_status rocsolver_dgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        double* A,
                                        const rocblas_int lda,
                                        double* B,
                                        const rocblas_int ldb,
                                        double* C,
                                        const rocblas_int ldc,
                                        double* X,
                                        const rocblas_int ldx)
{
    return rocsolver_geblttrs_npvt_impl<double>(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc,
                                                X, ldx);
}

rocblas_status rocsolver_cgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        rocblas_float_complex* A,
                                        const rocblas_int lda,
                                        rocblas_float_complex* B,
                                        const rocblas_int ldb,
                                        rocblas_float_complex* C,
                                        const rocblas_int ldc,
                                        rocblas_float_complex* X,
                                        const rocblas_int ldx)
{
    return rocsolver_geblttrs_npvt_impl<rocblas_float_complex>(handle, nb, nblocks, nrhs, A, lda, B,
                                                               ldb, C, ldc, X, ldx);
}

rocblas_status rocsolver_zgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        rocblas_double_complex* A,
                                        const rocblas_int lda,
                                        rocblas_double_complex* B,
                                        const rocblas_int ldb,
                                        rocblas_double_complex* C,
                                        const rocblas_int ldc,
                                        rocblas_double_complex* X,
                                        const rocblas_int ldx)
{
    return rocsolver_geblttrs_npvt_impl<rocblas_double_complex>(handle, nb, nblocks, nrhs, A, lda,
                                                                B, ldb, C, ldc, X, ldx);
}

} // extern C
