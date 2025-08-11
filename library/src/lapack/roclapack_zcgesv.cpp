/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "roclapack_zcgesv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename Tfull, typename Tlu, typename I, typename Istride>
rocblas_status rocsolver_zcgesv_impl(rocblas_handle handle,
                                     I const n,
                                     I const nrhs,

                                     Tfull* const A,
                                     Istride const strideA,
                                     I const lda,
                                     Istride strideA,

                                     I* const ipiv,

                                     Tfull* const B,
                                     Istride const strideB,
                                     I const ldb,
                                     Istride strideB,

                                     Tfull* const X,
                                     Istride const strideX,
                                     I const ldx,
                                     Istride strideX,

                                     I const max_iter_arg,
                                     double const tol_arg,
                                     I* niter,

                                     I* info,
                                     const I batch_count,
                                     void* work,
                                     size_t size_work)

{
    using Treduced = Tlu;

    ROCSOLVER_ENTER_TOP("zcgesv", "-n", n, "-nrhs", nrhs, "--lda", lda, "--ldb", ldb, "--ldx", ldx,
                        "--max_iter", max_iter_arg, "--tol", tol_arg);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_zcgesv_mxp_argCheck<Tfull, I>(handle, n, nrhs, lda, ldb, A, B,
                                                                ipiv, info, batch_count);

    if(st != rocblas_status_continue)
        return st;

    size_t size_work = 0;
    rocsolver_gesv_mxp_getMemorySize<Tfull, Tlu, Treduced, I>(n, nrhs, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation

    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    void* work = (void*)mem[0];

    // execution
    //

    return rocsolver_zcgesv_mxp_template(handle, n, nrhs,

                                         A, shiftA, lda, strideA,

                                         ipiv,

                                         B, shiftB, ldb, strideB,

                                         X, shiftX, ldx, strideX,

                                         max_iter_arg, tol_arg, niter, info, batch_count,
                                         work size_work);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_zcgesv_strided_batched(rocblas_handle handle,
                                                rocblas_int const n,
                                                rocblas_int const nrhs,

                                                rocblas_double_complex* const A,
                                                rocblas_stride const strideA,
                                                rocblas_int lda,
                                                rocblas_stride strideA,

                                                rocblas_int* ipiv,

                                                rocblas_double_complex* const B,
                                                rocblas_stride const strideB,
                                                rocblas_int ldb,
                                                rocblas_stride strideB,

                                                rocblas_double_complex* const X,
                                                rocblas_stride const strideX,
                                                rocblas_int ldx,
                                                rocblas_stride strideX,

                                                rocblas_int const max_iter,
                                                double const tol,
                                                rocblas_int* niter,
                                                rocblas_int* info,
                                                rocblas_int const batch_count)
{
    return rocsolver_gesv_impl<rocblas_double_complex, rocblas_float_complex, rocblas_int, rocblas_stride>(
        handle, n, nrhs,

        A, strideA, lda, strideA,

        ipiv,

        B, strideB, ldb, strideB,

        X, strideX, ldx, strideX,

        max_iter, tol, niter, info, batch_count);
}

rocblas_status rocsolver_zcgesv(rocblas_handle handle,
                                rocblas_int const n,
                                rocblas_int const nrhs,

                                rocblas_double_complex* const A,
                                rocblas_int lda,

                                rocblas_int* ipiv,

                                rocblas_double_complex* const B,
                                rocblas_int ldb,

                                rocblas_double_complex* const X,
                                rocblas_int ldx,

                                rocblas_int const max_iter,
                                double const tol,
                                rocblas_int* niter,
                                rocblas_int* info,
                                rocblas_int const batch_count)
{
    rocblas_int const batch_count = 1;

    rocblas_stride const shiftA = 0;
    rocblas_stride const shiftB = 0;
    rocblas_stride const shiftX = 0;

    rocblas_stride const strideA = lda * n;
    rocblas_stride const strideB = ldb * nrhs;
    rocblas_stride const strideX = ldx * nrhs;

    return rocsolver_zcgesv_strided_batched(handle, n, nrhs,

                                            A, strideA, lda, strideA,

                                            ipiv,

                                            B, strideB, ldb, strideB,

                                            X, strideX, ldx, strideX,

                                            max_ter, tol, niter, info, batch_count);
}

rocblas_status rocsolver_dsgesv_strided_batched(rocblas_handle handle,
                                                rocblas_int const n,
                                                rocblas_int const nrhs,

                                                double* const A,
                                                rocblas_stride const strideA,
                                                rocblas_int lda,
                                                rocblas_stride strideA,

                                                rocblas_int* ipiv,

                                                double* const B,
                                                rocblas_stride const strideB,
                                                rocblas_int ldb,
                                                rocblas_stride strideB,

                                                double* const X,
                                                rocblas_stride const strideX,
                                                rocblas_int ldx,
                                                rocblas_stride strideX,

                                                rocblas_int const max_iter,
                                                double const tol,
                                                rocblas_int* niter,
                                                rocblas_int* info,
                                                rocblas_int const batch_count)
{
    return rocsolver_gesv_impl<double, float, rocblas_int, rocblas_stride>(handle, n, nrhs,

                                                                           A, strideA, lda, strideA,

                                                                           ipiv,

                                                                           B, strideB, ldb, strideB,

                                                                           X, strideX, ldx, strideX,

                                                                           max_iter, tol, niter,
                                                                           info, batch_count);
}

rocblas_status rocsolver_dsgesv(rocblas_handle handle,
                                rocblas_int const n,
                                rocblas_int const nrhs,

                                double* const A,
                                rocblas_int lda,

                                rocblas_int* ipiv,

                                double* const B,
                                rocblas_int ldb,

                                double* const X,
                                rocblas_int ldx,

                                rocblas_int const max_iter,
                                double const tol,
                                rocblas_int* niter,
                                rocblas_int* info,
                                rocblas_int const batch_count)
{
    rocblas_int const batch_count = 1;

    rocblas_stride const shiftA = 0;
    rocblas_stride const shiftB = 0;
    rocblas_stride const shiftX = 0;

    rocblas_stride const strideA = lda * n;
    rocblas_stride const strideB = ldb * nrhs;
    rocblas_stride const strideX = ldx * nrhs;

    return rocsolver_dsgesv_strided_batched(handle, n, nrhs,

                                            A, strideA, lda, strideA,

                                            ipiv,

                                            B, strideB, ldb, strideB,

                                            X, strideX, ldx, strideX,

                                            max_ter, tol, niter, info, batch_count);
}

} // extern C
