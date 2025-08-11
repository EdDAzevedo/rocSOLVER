/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_gesv_ex.hpp"
#include "rocblas_utility.hpp"
#include <rocblas/internal/rocblas-types.h>

#include "roclapack_gesv.hpp"
#include "roclapack_getrf.hpp"
#include "roclapack_getrf_mxp.hpp"
#include "roclapack_getrs.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// The following traits enforce that homogenous computation is viable if
//  - A, B, X, compute_type are all the same type
//  - A, B, X, compute type are lapack SDCZ storage types

template <typename T>
constexpr bool is_gesv_ex_homogenous_storage
    = std::is_same_v<
          T,
          float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>;

template <typename T, typename... Ts>
constexpr bool gesv_ex_homogenous_accepts = (std::is_same_v<T, Ts> && ...)
    && (is_gesv_ex_homogenous_storage<T> && ... && is_gesv_ex_homogenous_storage<Ts>);

template <typename T>
rocblas_status rocsolver_gesv_ex_homogenous(rocblas_handle handle,
                                            const rocblas_int n,
                                            const rocblas_int nrhs,
                                            T* A,
                                            const rocblas_int lda,
                                            rocblas_int* ipiv,
                                            T* B,
                                            const rocblas_int ldb,
                                            T* X,
                                            const rocblas_int ldx,
                                            const rocblas_int max_iter,
                                            const double tol,
                                            rocblas_int* niter,
                                            rocblas_int* info)
{
    // XXX: this is just the body of gesv_impl from roclapack_gesv.cpp, it is not adapted for explicit X, etc.
    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesv_argCheck(handle, n, nrhs, lda, ldb, A, B, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling GETRF and GETRS)
    bool optim_mem;
    size_t size_work, size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETRF
    size_t size_pivotval, size_pivotidx, size_iinfo, size_iipiv;
    rocsolver_gesv_getMemorySize<false, false, T>(
        n, nrhs, batch_count, &size_scalars, &size_work, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work, size_work1, size_work2, size_work3, size_work4,
            size_pivotval, size_pivotidx, size_iipiv, size_iinfo);
    // memory workspace allocation
    void *scalars, *work, *work1, *work2, *work3, *work4, *pivotval, *pivotidx, *iinfo, *iipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_work1, size_work2, size_work3,
                              size_work4, size_pivotval, size_pivotidx, size_iipiv, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    work1 = mem[2];
    work2 = mem[3];
    work3 = mem[4];
    work4 = mem[5];
    pivotval = mem[6];
    pivotidx = mem[7];
    iipiv = mem[8];
    iinfo = mem[9];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesv_template<false, false, T>(
        handle, n, nrhs, A, shiftA, lda, strideA, ipiv, strideP, B, shiftB, ldb, strideB, info,
        batch_count, (T*)scalars, (T*)work, work1, work2, work3, work4, (T*)pivotval,
        (rocblas_int*)pivotidx, (rocblas_int*)iipiv, (rocblas_int*)iinfo, optim_mem);
}

template <typename TA, typename TB, typename TX, typename Tc, typename...>
struct gesv_homogenous_call
{
    rocblas_status operator()(rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int nrhs,
                              void* A,
                              const rocblas_int lda,
                              rocblas_int* ipiv,
                              void* B,
                              const rocblas_int ldb,
                              void* X,
                              const rocblas_int ldx,
                              const rocblas_int max_iter,
                              const double tol,
                              rocblas_int* niter,
                              rocblas_int* info)
    {
        if constexpr(gesv_ex_homogenous_accepts<TA, TB, TX, Tc>)
        {
            return rocsolver_gesv_ex_homogenous(handle, n, nrhs, (TA*)A, lda, ipiv, (TB*)B, ldb,
                                                (TX*)X, ldx, max_iter, tol, niter, info);
        }

        return rocblas_status_not_implemented;
    }
};

// The following traits allow selecting a reduced precision companion type for a given compute type

template <typename T>
struct gesv_ex_mxp_lu_reduced_precision
{
    using type = void;
};

template <>
struct gesv_ex_mxp_lu_reduced_precision<float>
{
    using type = rocblas_half;
};

template <typename T>
using gesv_ex_mxp_lu_reduced_precision_t = typename gesv_ex_mxp_lu_reduced_precision<T>::type;

// The following traits enforce that MXP LU computation is viable if:
//  - A, B, X are all the same type
//  - A, B, X are lapack SDCZ storage types
//  - compute_type is SDCZ, half, or bfloat16

template <typename T>
constexpr bool is_gesv_ex_mxp_lu_storage
    = std::is_same_v<
          T,
          float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>;

template <typename T>
constexpr bool is_gesv_ex_mxp_lu_compute
    = std::is_same_v<T, rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || is_gesv_ex_mxp_lu_storage<T>;

template <typename TA, typename TB, typename TX, typename Tc>
constexpr bool gesv_ex_mxp_lu_accepts
    = (std::is_same_v<TA, TB> && std::is_same_v<TA, TX>)&&is_gesv_ex_mxp_lu_storage<
        TA>&& is_gesv_ex_mxp_lu_compute<Tc>;

template <typename T, typename LU, typename R = LU>
rocblas_status rocsolver_gesv_ex_mxp_lu(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        T* A,
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        T* B,
                                        const rocblas_int ldb,
                                        T* X,
                                        const rocblas_int ldx,
                                        const rocblas_int max_iter,
                                        const double tol,
                                        rocblas_int* niter,
                                        rocblas_int* info)
{
    // MXP LU implementation goes here
    return rocblas_status_not_implemented;
}

template <typename TA, typename TB, typename TX, typename Tc, typename...>
struct gesv_mxp_lu_call
{
    rocblas_status operator()(rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int nrhs,
                              void* A,
                              const rocblas_int lda,
                              rocblas_int* ipiv,
                              void* B,
                              const rocblas_int ldb,
                              void* X,
                              const rocblas_int ldx,
                              const rocblas_int max_iter,
                              const double tol,
                              rocblas_int* niter,
                              rocblas_int* info)
    {
        if constexpr(gesv_ex_mxp_lu_accepts<TA, TB, TX, Tc>)
        {
            // here we can choose a reduced precision type based on Tc if we need to, otherwise we default to Tc for both
            // return rocsolver_gesv_ex_mxp_lu<TA, Tc, gesv_ex_mxp_lu_reduced_precision_t<Tc>>(...)
            return rocsolver_gesv_ex_mxp_lu<TA, Tc, Tc>(handle, n, nrhs, (TA*)A, lda, ipiv, (TB*)B,
                                                        ldb, (TX*)X, ldx, max_iter, tol, niter, info);
        }
        return rocblas_status_not_implemented;
    }
};

rocblas_status rocsolver_gesv_ex_impl(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nrhs,
                                      void* A,
                                      const rocblas_datatype A_type,
                                      const rocblas_int lda,
                                      rocblas_int* ipiv,
                                      void* B,
                                      const rocblas_datatype B_type,
                                      const rocblas_int ldb,
                                      void* X,
                                      const rocblas_datatype X_type,
                                      const rocblas_int ldx,
                                      const rocblas_int max_iter,
                                      const double tol,
                                      rocblas_int* niter,
                                      rocblas_datatype compute_type,
                                      rocblas_int* info)
{
    using T = void*;
    ROCSOLVER_ENTER_TOP("gesv_ex", "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb);

    rocblas_status ret;

    // the most specific cases are tried first

    ret = rocsolver_ex_datatype_dispatch<gesv_homogenous_call>(A_type, B_type, X_type, compute_type,
                                                               handle, n, nrhs, A, lda, ipiv, B, ldb,
                                                               X, ldx, max_iter, tol, niter, info);

    if(ret != rocblas_status_not_implemented)
    {
        return ret;
    }

    ret = rocsolver_ex_datatype_dispatch<gesv_mxp_lu_call>(A_type, B_type, X_type, compute_type,
                                                           handle, n, nrhs, A, lda, ipiv, B, ldb, X,
                                                           ldx, max_iter, tol, niter, info);

    return ret;
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_gesv_ex(rocblas_handle handle,
                                            const rocblas_int n,
                                            const rocblas_int nrhs,
                                            void* A,
                                            const rocblas_datatype A_type,
                                            const rocblas_int lda,
                                            rocblas_int* ipiv,
                                            void* B,
                                            const rocblas_datatype B_type,
                                            const rocblas_int ldb,
                                            void* X,
                                            const rocblas_datatype X_type,
                                            const rocblas_int ldx,
                                            const rocblas_int max_iter,
                                            const double tol,
                                            rocblas_int* niter,
                                            rocblas_datatype compute_type,
                                            rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_ex_impl(handle, n, nrhs, A, A_type, lda, ipiv, B, B_type, ldb, X,
                                             X_type, ldx, max_iter, tol, niter, compute_type, info);
}
