
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
// #include "roclapack_getrs.hpp"
#include "rocsolver_geblttrs_npvt_strided_batched_large.hpp"
#include "rocsolver_geblttrs_npvt_strided_batched_small.hpp"

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_geblttrs_strided_batched_impl(rocblas_handle handle,
                                                       I nb,
                                                       I nblocks,
                                                       I nrhs,
                                                       const T* A_,
                                                       I lda,
                                                       Istride strideA,
                                                       const T* B_,
                                                       I ldb,
                                                       Istride strideB,
                                                       const T* C_,
                                                       I ldc,
                                                       Istride strideC,

                                                       T* X_,
                                                       I ldx,
                                                       Istride strideX,
                                                       I batch_count)
{
    ROCSOLVER_ENTER_TOP("getrs_strided_batched", "--nb", nb, "--nblocks", nblocks, "--nrhs", nrhs,
                        "--lda", lda, "--strideA", strideA, "--ldb", ldb, "--strideB", strideB,
                        "--batch_count", batch_count);
    /* 
    ---------------
    check arguments
    ---------------
    */
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    // no work
    if((nb == 0) || (nblocks == 0) || (batch_count == 0) || (nrhs == 0))
    {
        return (rocblas_status_success);
    };

    {
        bool const isok = (nb >= 1) && (nblocks >= 1) && (batch_count >= 1) && (strideA >= 1)
            && (strideB >= 1) && (strideC >= 1) && (strideX >= 1) && (lda >= nb) && (ldb >= nb)
            && (ldc >= nb) && (ldx >= nb);
        if(!isok)
        {
            return (rocblas_status_invalid_size);
        };

        // check no overlap
        bool const isok_stride = (batch_count >= 2) && (strideA >= (lda * nb) * nblocks)
            && (strideB >= (ldb * nb) * nblocks) && (strideC >= (ldc * nb) * nblocks)
            && (strideX >= (ldx * nblocks) * nrhs);
        if(!isok_stride)
        {
            return (rocblas_status_invalid_size);
        };
    };

    if((A_ == nullptr) || (B_ == nullptr) || (C_ == nullptr) || (X_ == nullptr))
    {
        return (rocblas_status_invalid_pointer);
    };

    if(nb < NB_SMALL)
    {
        return (rocsolver_geblttrs_npvt_strided_batched_small_template<T, I, Istride>(
            handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC, X_,
            ldx, strideX, batch_count));
    }
    else
    {
        return (rocsolver_geblttrs_npvt_strided_batched_large_template<T, I, Istride>(
            handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC, X_,
            ldx, strideX, batch_count));
    };
};

extern "C" {

rocblas_status rocsolver_dgeblttrs_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   rocblas_int nrhs,
                                                   double* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   double* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   double* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,

                                                   double* X_,
                                                   rocblas_int ldx,
                                                   rocblas_stride strideX,
                                                   rocblas_int batch_count)
{
    return (rocsolver_geblttrs_strided_batched_impl<double, rocblas_int, rocblas_stride>(
        handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,

        X_, ldx, strideX, batch_count));
};

rocblas_status rocsolver_sgeblttrs_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   rocblas_int nrhs,
                                                   float* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   float* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   float* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,

                                                   float* X_,
                                                   rocblas_int ldx,
                                                   rocblas_stride strideX,
                                                   rocblas_int batch_count)
{
    return (rocsolver_geblttrs_strided_batched_impl<float, rocblas_int, rocblas_stride>(
        handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,

        X_, ldx, strideX, batch_count));
};

rocblas_status rocsolver_zgeblttrs_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   rocblas_int nrhs,
                                                   rocblas_double_complex* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   rocblas_double_complex* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   rocblas_double_complex* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,

                                                   rocblas_double_complex* X_,
                                                   rocblas_int ldx,
                                                   rocblas_stride strideX,
                                                   rocblas_int batch_count)
{
    return (
        rocsolver_geblttrs_strided_batched_impl<rocblas_double_complex, rocblas_int, rocblas_stride>(
            handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,

            X_, ldx, strideX, batch_count));
};

rocblas_status rocsolver_cgeblttrs_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   rocblas_int nrhs,
                                                   rocblas_float_complex* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   rocblas_float_complex* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   rocblas_float_complex* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,

                                                   rocblas_float_complex* X_,
                                                   rocblas_int ldx,
                                                   rocblas_stride strideX,
                                                   rocblas_int batch_count)
{
    return (
        rocsolver_geblttrs_strided_batched_impl<rocblas_float_complex, rocblas_int, rocblas_stride>(
            handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,

            X_, ldx, strideX, batch_count));
};
}
