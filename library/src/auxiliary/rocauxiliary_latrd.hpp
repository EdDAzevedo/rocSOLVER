/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
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

#pragma once

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

static bool constexpr latrd_use_gemv = true;

template <bool BATCHED, typename T>
void rocsolver_latrd_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_norms,
                                   size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_workArr = 0;
        return;
    }

    size_t n1 = 0, n2 = 0;
    size_t w1 = 0, w2 = 0, w3 = 0;

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements for calling larfg
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w1, &n1);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w2);

    // size of re-usable workspace
    // TODO: replace with rocBLAS call
    constexpr int ROCBLAS_DOT_NB = 512;
    w3 = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    w3 *= sizeof(T) * batch_count;
    n2 = sizeof(T) * batch_count;

    *size_norms = std::max(n1, n2);
    *size_work = std::max({w1, w2, w3});

    if(latrd_use_gemv)
    {
        // -------------------------------------------------------
        // scratch space to store the strictly upper triangular or
        // strictly lower triangular part
        // -------------------------------------------------------
        //       auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };
        //        size_t const len_triangle_matrix = is_even(n) ? static_cast<int64_t>(n / 2) * (n + 1)
        //                                                      : static_cast<int64_t>(n) * ((n + 1) / 2);

        auto const len_triangle_matrix = n * n;
        *size_work
            += sizeof(T) * len_triangle_matrix * std::max(static_cast<rocblas_int>(1), batch_count);
    }
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_latrd_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int lda,
                                        const rocblas_int ldw,
                                        T A,
                                        S E,
                                        U tau,
                                        U W,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || k < 0 || k > n || lda < n || ldw < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !E) || (n && !tau) || (n && k && !W))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// ------------------------------------------------------------
// copy the strictly upper or strictly lower triangular matrix
// of n by n matrix
//
// A is symmetric full matrix
// C is conceptualy triangular matrix
//
// C <- A   if is_update_C is true
// A <- C   if is_update_C is false
//
//
// update the lower triangular part if "is_update_lower" is true
// update the upper triangular part if "is_update_lower" is false
// ------------------------------------------------------------
template <typename T, typename I>
static __device__ void copy_triang_body(bool const is_update_lower,
                                        bool const is_update_C,
                                        I const n,
                                        T* const A_,
                                        I const lda,
                                        T* const C_)
{
    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    auto idx_lower = [=](auto i, auto j, auto n) {
        assert((0 <= i) && (i < n));
        assert((0 <= j) && (j < n));
        assert((i >= j));

        auto const tmp = is_even(j) ? static_cast<int64_t>(j / 2) * (2 * n + 1 - j)
                                    : static_cast<int64_t>(j) * ((2 * n + 1 - j) / 2);

        return (((i - j) + tmp));
    };

    auto idx_upper = [=](auto i, auto j, auto n) {
        assert((0 <= i) && (i < n));
        assert((0 <= j) && (j < n));
        assert((i <= j));

        auto const tmp = is_even(j) ? static_cast<int64_t>(j / 2) * (j + 1)
                                    : static_cast<int64_t>(j) * ((j + 1) / 2);

        return ((i + tmp));
    };

    auto idx_compact = [=](auto i, auto j, auto n) {
        return ((is_update_lower) ? idx_lower(i, j, n) : idx_upper(i, j, n));
    };

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    // auto C = [=](auto i, auto j) -> T& { return (C_[idx_compact(i, j, n)]); };
    I const ldc = n;
    auto C = [=](auto i, auto j) -> T& { return (C_[idx2D(i, j, ldc)]); };

    auto A = [=](auto i, auto j) -> T& { return (A_[idx2D(i, j, lda)]); };

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;

    I const ibx = hipBlockIdx_x;
    I const iby = hipBlockIdx_y;

    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;
    {
        assert(hipBlockDim_z == 1);
    }

    I const nbx = hipGridDim_x;
    I const nby = hipGridDim_y;

    I const i_start = tix + ibx * nx;
    I const j_start = tiy + iby * ny;

    I const i_inc = nx * nbx;
    I const j_inc = ny * nby;

    bool const is_update_upper = (!is_update_lower);

    for(I j = (0 + j_start); j < n; j += j_inc)
    {
        for(I i = (0 + i_start); i < n; i += i_inc)
        {
            bool const is_strictly_lower = (i > j);
            bool const is_strictly_upper = (i < j);

            bool const do_work
                = (is_update_lower && is_strictly_lower) || (is_update_upper && is_strictly_upper);

            if(do_work)
            {
                if(is_update_C)
                {
                    C(i, j) = A(i, j);
                }
                else
                {
                    A(i, j) = C(i, j);
                }
            }
        } // end for i
    } // end for j
}

template <typename T, typename I, typename Istride, typename UA, typename UC>
static __global__ void copy_triang_kernel(bool const is_update_lower,
                                          bool const is_update_C,
                                          I const n,
                                          UA A_,
                                          Istride const shiftA,
                                          I const lda,
                                          Istride const strideA,
                                          UC C_,
                                          Istride const shiftC,
                                          Istride const strideC,
                                          I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    for(I bid = (0 + bid_start); bid < batch_count; bid += bid_inc)
    {
        auto const Ap = load_ptr_batch(A_, bid, shiftA, strideA);
        auto const Cp = load_ptr_batch(C_, bid, shiftC, strideC);

        copy_triang_body(is_update_lower, is_update_C, n, Ap, lda, Cp);
    }
}

template <typename T, typename I, typename Istride, typename UA, typename UC>
static void copy_triang(rocblas_handle handle,
                        bool const is_update_lower,
                        bool const is_update_C,
                        I const n,
                        UA A_,
                        Istride const shiftA,
                        I const lda,
                        Istride const strideA,
                        UC C_,
                        Istride const shiftC,
                        Istride const strideC,
                        I const batch_count)
{
    auto ceil = [](auto m, auto n) { return (1 + (m - 1) / n); };

    I const nb_max = 64 * 1000;
    I const nx = 32;
    I const ny = 32;

    I const nbx = std::max(I(1), std::min(nb_max, ceil(n, nx)));
    I const nby = std::max(I(1), std::min(nb_max, ceil(n, ny)));
    I const nbz = std::max(I(1), std::min(nb_max, batch_count));

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    copy_triang_kernel<T, I, Istride, UA, UC>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(is_update_lower, is_update_C,

                                                              n,

                                                              A_, shiftA, lda, strideA,

                                                              C_, shiftC, strideC,

                                                              batch_count);
}

// --------------------------------------------
// update strictly lower triangular part or
// strictly upper triangular part so matrix is
// symmetric or hermitian
//
// update the strictly lower triangular part if "is_update_lower" is true
// update the strictly upper triangular part if "is_update_lower" is false
// --------------------------------------------
template <typename T, typename I>
static __device__ void
    symmetrize_matrix_body(bool const is_update_lower, I const n, T* const A_, I const lda)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto A = [=](auto i, auto j) -> T& { return (A_[idx2D(i, j, lda)]); };

    I i_start = 0;
    I j_start = 0;
    I i_inc = 0;
    I j_inc = 0;
    {
        auto const tix = hipThreadIdx_x;
        auto const tiy = hipThreadIdx_y;

        auto const nx = hipBlockDim_x;
        auto const ny = hipBlockDim_y;
        {
            assert(hipBlockDim_z == 1);
        }

        auto const ibx = hipBlockIdx_x;
        auto const iby = hipBlockIdx_y;

        auto const nbx = hipGridDim_x;
        auto const nby = hipGridDim_y;

        i_start = tix + ibx * nx;
        j_start = tiy + iby * ny;

        i_inc = nx * nbx;
        j_inc = ny * nby;
    }

    for(I j = (0 + j_start); j < n; j += j_inc)
    {
        // --------------------------------------------------------
        // process strictly upper or strictly lower triangular part
        // --------------------------------------------------------
        I const row_start = (is_update_lower) ? j + 1 : 0;
        I const row_end = (is_update_lower) ? n : j;

        for(I i = (row_start + i_start); i < row_end; i += i_inc)
        {
            if constexpr(is_complex)
            {
                A(i, j) = conj(A(j, i));
            }
            else
            {
                A(i, j) = A(j, i);
            }

        } // end for i
    } // end for j
}

template <typename T, typename I>
static __device__ void
    symmetrize_matrix_body_org(bool const is_update_lower, I const n, T* const A_, I const lda)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto A = [=](auto i, auto j) -> T& { return (A_[idx2D(i, j, lda)]); };

    I const tix = hipThreadIdx_x;
    I const tiy = hipThreadIdx_y;

    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;
    {
        assert(hipBlockDim_z == 1);
    }

    I const ibx = hipBlockIdx_x;
    I const iby = hipBlockIdx_y;

    I const nbx = hipGridDim_x;
    I const nby = hipGridDim_y;

    I const i_start = tix + ibx * nx;
    I const j_start = tiy + iby * ny;

    I const i_inc = nx * nbx;
    I const j_inc = ny * nby;

    bool const is_update_upper = (!is_update_lower);

    for(I j = (0 + j_start); j < n; j += j_inc)
    {
        for(I i = (0 + i_start); i < n; i += i_inc)
        {
            bool const is_strictly_lower = (i > j);
            bool const is_strictly_upper = (i < j);

            bool const do_work
                = (is_update_lower && is_strictly_lower) || (is_update_upper && is_strictly_upper);

            if(do_work)
            {
                if constexpr(is_complex)
                {
                    A(i, j) = conj(A(j, i));
                }
                else
                {
                    A(i, j) = A(j, i);
                }
            }
        } // end for i
    } // end for j
}

template <typename T, typename I, typename Istride, typename UA>
static __global__ void symmetrize_matrix_kernel(bool is_update_lower,
                                                I const n,

                                                UA A_,
                                                Istride const shiftA,
                                                I const lda,
                                                Istride const strideA,

                                                I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    for(I bid = (0 + bid_start); bid < batch_count; bid += bid_inc)
    {
        T* const Ap = load_ptr_batch(A_, bid, shiftA, strideA);
        symmetrize_matrix_body<T, I>(is_update_lower, n, Ap, lda);
    }
}

template <typename T, typename I, typename Istride, typename UA>
static void symmetrize_matrix(rocblas_handle handle,
                              bool const is_update_lower,
                              I const n,
                              UA A_,
                              Istride const shiftA,
                              I const lda,
                              Istride const strideA,
                              I const batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const nx = 32;
    I const ny = 32;

    I const nb_max = 64 * 1000;

    I const nbx = std::max(I(1), std::min(nb_max, ceil(n, nx)));
    I const nby = std::max(I(1), std::min(nb_max, ceil(n, ny)));
    I const nbz = std::max(I(1), std::min(batch_count, nb_max));

    symmetrize_matrix_kernel<T, I, Istride, UA><<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(

        is_update_lower, n,

        A_, shiftA, lda, strideA,

        batch_count);
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        T* W,
                                        const rocblas_int shiftW,
                                        const rocblas_int ldw,
                                        const rocblas_stride strideW,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T* norms,
                                        T** workArr)
{
    ROCSOLVER_ENTER("latrd", "uplo:", uplo, "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "shiftW:", shiftW, "ldw:", ldw, "bc:", batch_count);
    // quick return
    if(n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // configure kernels
    rocblas_int blocks = (batch_count - 1) / BS1 + 1;
    dim3 grid_b(blocks, 1);
    dim3 threads(BS1, 1, 1);
    blocks = (n - 1) / BS1 + 1;
    dim3 grid_n(blocks, batch_count);

    if(latrd_use_gemv)
    {
        // --------------------------------------------
        // enforce symmetry to use gemv instead of symv
        // --------------------------------------------

        rocblas_stride const lshiftA = shiftA;

        bool const is_update_lower = (uplo == rocblas_fill_upper);
        symmetrize_matrix<T>(handle, is_update_lower, n,

                             A, lshiftA, lda, strideA,

                             batch_count);
    }

    if(uplo == rocblas_fill_lower)
    {
        // reduce the first k columns of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update column j of A with reflector computed in step j-1
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda), lda,
                                strideA, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j, 0, ldw), ldw,
                                strideW, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), A,
                                     shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1, strideA,
                                     (tau + j), strideP, batch_count, work, norms);

            // copy to E(j) the corresponding off-diagonal element of A, which is set to 1
            ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, (E + j), strideE);

            // compute/update column j of W
            if(latrd_use_gemv)
            {
                auto const nn = n - 1 - j;
                rocblasCall_gemv<T>(handle, rocblas_operation_none, nn, nn, (scalars + 2), 0,

                                    A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,

                                    A, shiftA + idx2D(j + 1, j, lda), 1, strideA,

                                    (scalars + 1), 0,

                                    W, shiftW + idx2D(j + 1, j, ldw), 1, strideW,

                                    batch_count, workArr);
            }
            else
            {
                rocblasCall_symv_hemv<T>(
                    handle, uplo, n - 1 - j, (scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda),
                    lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA, (scalars + 1), 0, W,
                    shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, work, workArr);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            rocblasCall_scal<T>(handle, n - j - 1, (tau + j), strideP, W,
                                shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count);

            rocblasCall_dot<COMPLEX, T>(handle, n - 1 - j, W, shiftW + idx2D(j + 1, j, ldw), 1,
                                        strideW, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                        batch_count, norms, work, workArr);

            // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
            //  available, we can use it instead of the scale_axpy kernel, if it provides
            //  better performance.)
            ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, n - 1 - j, norms,
                                    tau + j, strideP, A, shiftA + idx2D(j + 1, j, lda), strideA, W,
                                    shiftW + idx2D(j + 1, j, ldw), strideW);
        }
    }

    else
    {
        // reduce the last k columns of A
        // main loop running forwards (for each column)
        rocblas_int jw;
        for(rocblas_int j = n - 1; j >= n - k; --j)
        {
            jw = j - n + k;
            // update column j of A with reflector computed in step j-1
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(j, j + 1, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, j, A, shiftA + idx2D(j - 1, j, lda), A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, (tau + j - 1), strideP,
                                     batch_count, work, norms);

            // copy to E(j) the corresponding off-diagonal element of A, which is set to 1
            ROCSOLVER_LAUNCH_KERNEL(set_offdiag<T>, grid_b, threads, 0, stream, batch_count, A,
                                    shiftA + idx2D(j - 1, j, lda), strideA, (E + j - 1), strideE);

            // compute/update column j of W
            if(latrd_use_gemv)
            {
                auto const nn = j;
                rocblasCall_gemv<T>(handle, rocblas_operation_none, nn, nn, (scalars + 2), 0,

                                    A, shiftA, lda, strideA,

                                    A, shiftA + idx2D(0, j, lda), 1, strideA,

                                    (scalars + 1), 0,

                                    W, shiftW + idx2D(0, jw, ldw), 1, strideW,

                                    batch_count, workArr);
            }
            else
            {
                rocblasCall_symv_hemv<T>(handle, uplo, j, (scalars + 2), 0, A, shiftA, lda, strideA,
                                         A, shiftA + idx2D(0, j, lda), 1, strideA, (scalars + 1), 0,
                                         W, shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count,
                                         work, workArr);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_scal<T>(handle, j, (tau + j - 1), strideP, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count);

            rocblasCall_dot<COMPLEX, T>(handle, j, W, shiftW + idx2D(0, jw, ldw), 1, strideW, A,
                                        shiftA + idx2D(0, j, lda), 1, strideA, batch_count, norms,
                                        work, workArr);

            // (TODO: rocblas_axpy is not yet ready to be used in rocsolver. When it becomes
            //  available, we can use it instead of the scale_axpy kernel, if it provides
            //  better performance.)
            ROCSOLVER_LAUNCH_KERNEL(scale_axpy<T>, grid_n, threads, 0, stream, j, norms,
                                    tau + j - 1, strideP, A, shiftA + idx2D(0, j, lda), strideA, W,
                                    shiftW + idx2D(0, jw, ldw), strideW);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
