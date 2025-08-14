/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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

#include "rocblas.hpp"
#include "roclapack_getrf.hpp"
#include "roclapack_getrs.hpp"
#include "rocsolver/rocsolver.h"

#include "roclapack_gesv.hpp"
#include "roclapack_getrf_mxp.hpp"

#include "auxiliary/rocauxiliary_lacpy.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// -----------------------------------------------------
// gather the value from iamax into xnrm
//
// assume launch as dim3( nbx, 1, nbz ), dim3(nx,1,1)
// nbx = ceil( nrhs, nx )
// nby = batch_count
// -----------------------------------------------------
template <typename T, typename S, typename I, typename Istride>
__device__ static void gather_norm_kernel(I const n,
                                          I const nrhs,

                                          T* const X,
                                          Istride const shiftX,
                                          I const ldx,
                                          Istride const strideX,

                                          I* const ixnrm,

                                          S* const xnrm,

                                          I const batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const irhs_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const irhs_inc = blockDim.x * gridDim.x;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    for(I irhs = irhs_start; irhs < nrhs; irhs += irhs_inc)
    {
        for(I bid = bid_start; bid < batch_count; bid += bid_inc)
        {
            auto const Xp = load_ptr_batch(X, bid, shiftX, strideX);

            auto const irow = ixnrm[bid + irhs * batch_count];
            auto const jcol = irhs;
            auto const xi = Xp[idx2D(irow, jcol, ldx)];
            xnrm[bid + irhs * batch_count] = std::abs(xi);
        }
    }
}

template <typename T, typename S, typename I, typename Istride>
static void gather_norm(rocblas_handle handle,
                        I const n,
                        I const nrhs,

                        T* const X,
                        Istride const shiftX,
                        I const ldx,
                        Istride const strideX,

                        I* const ixnrm,

                        S* const xnrm,

                        I const batch_count)
{
    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const max_blocks = 1024;
    I const nx = 64;
    I const nbx = std::min(max_blocks, ceil(n, nx));
    I const nby = 1;
    I const nbz = std::min(max_blocks, batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    gather_norm_kernel<T, S, I, Istride><<<dim3(nbx, nby, nbz), dim3(nx, 1, 1), 0, stream>>>(

        n, nrhs,

        X, shiftX, ldx, strideX,

        ixnrm, xnrm, batch_count);
}

// -----------------------------------------
// assume one thread block handle one vector
// launch as
// dim3(1,nrhs,batch_count), dim3(nx,1,1,), ldsize, stream
// ldsize = nx * sizeof(double)
// -----------------------------------------
template <typename T, typename I, typename Istride>
__global__ static void check_convergence_kernel(I const n,
                                                I const nrhs,

                                                T* X,
                                                Istride const shiftX,
                                                I const ldx,
                                                Istride const strideX,

                                                T* R,
                                                Istride const shiftR,
                                                I const ldr,
                                                Istride const strideR,

                                                I const batch_count,
                                                double tol,

                                                bool* p_is_converged)
{
    bool is_converged = false;
    *p_is_converged = is_converged;

    extern __shared__ double ldmem[];

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const irhs_start = blockIdx.y;
    I const irhs_inc = gridDim.y;

    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;

    I const i_start = tid;
    I const i_inc = nthreads;

    assert(gridDim.x == 1);

    I nconverged = 0;
    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const Xp = load_ptr_batch(X, bid, shiftX, strideX);
        T const* const Rp = load_ptr_batch(R, bid, shiftR, strideR);

        for(I irhs = irhs_start; irhs < nrhs; irhs += irhs_inc)
        {
            double xmax = 0;
            double rmax = 0;
            for(I i = i_start; i < n; i += i_inc)
            {
                auto const xi = Xp[idx2D(i, irhs, ldx)];
                auto const ri = Rp[idx2D(i, irhs, ldr)];
                double const abs_xi = std::abs(xi);
                double const abs_ri = std::abs(ri);

                xmax = std::max(xmax, abs_xi);
                rmax = std::max(rmax, abs_ri);
            }
            __syncthreads();

            // ---------------------------------
            // perform max reduction
            // the answer is in the [0] position
            // ---------------------------------
            auto max_reduce = [=](auto& xmax) {
                ldmem[tid] = xmax;
                __syncthreads();

                for(I gap = nthreads / 2; gap > 0; gap = gap / 2)
                {
                    if(tid < gap)
                    {
                        ldmem[tid] = std::max(ldmem[tid], ldmem[tid + gap]);
                    }
                    __syncthreads();
                }
                __syncthreads();
                xmax = ldmem[0];
                __syncthreads();
            };

            max_reduce(xmax);
            max_reduce(rmax);

            if(rmax <= xmax * tol)
            {
                nconverged++;
            }

        } // end for irhs
        __syncthreads();
    } // end for bid

    __syncthreads();

    is_converged = (nconverged >= (nrhs * batch_count));

    *p_is_converged = is_converged;
}

// check for convergence
template <typename T, typename I, typename Istride>
static void check_convergence(rocblas_handle handle,
                              I const n,
                              I const nrhs,

                              T* X,
                              Istride const shiftX,
                              I const ldx,
                              Istride const strideX,

                              T* R,
                              Istride const shiftR,
                              I const ldr,
                              Istride const strideR,

                              I const batch_count,
                              double tol,

                              bool* d_is_converged)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const nx = 1024;
    size_t const ldsize = sizeof(double) * nx;
    check_convergence_kernel<T, I, Istride>
        <<<dim3(1, nrhs, batch_count), dim3(nx, 1, 1), ldsize, stream>>>(n, nrhs,

                                                                         X, shiftX, ldx, strideX,

                                                                         R, shiftR, ldr, strideR,

                                                                         batch_count, tol,
                                                                         d_is_converged);
}

template <typename T, typename I>
rocblas_status rocsolver_zcgesv_mxp_argCheck(rocblas_handle handle,
                                             const I n,
                                             const I nrhs,
                                             const I lda,
                                             const I ldb,
                                             T* A,
                                             T* B,
                                             const I* ipiv,
                                             const I* info,
                                             const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (nrhs && n && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename Tlu, typename Treduced, typename I>
void rocsolver_gesv_mxp_getMemorySize(const I n,
                                      const I nrhs,
                                      const I batch_count,

                                      size_t* p_size_work

)
{
    using S = decltype(std::real(T{}));

    bool constexpr BATCHED = true;
    bool constexpr STRIDED = true;

    size_t size_work = 0;
    *p_size_work = size_work;

    // if quick return, no workspace is needed
    bool const has_work = (n >= 1) && (nrhs >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    // ---------------------------------------------------
    // storage for copies of matrices for iterative refinement
    // ---------------------------------------------------
    {
        size_t size_A_lu = sizeof(Tlu) * n * n * batch_count;
        size_t size_R = sizeof(T) * n * nrhs * batch_count;
        size_t size_B_lu = sizeof(Tlu) * n * nrhs * batch_count;

        size_work += size_A_lu;
        size_work += size_R;
        size_work += size_B_lu;
    }

    // --------------------------------------
    // storage for LU factorization (in FP32)
    // --------------------------------------
    bool const use_pivot = true;
    size_t size_getrf = 0;
    {
        size_t size_scalars = 0;
        size_t size_work0 = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        size_t size_pivotval = 0;
        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;
        size_t optim_mem = true;

        bool opt1 = true;
        bool opt2 = true;

        // ------------------------------------
        // workspace required for calling GETRF
        // ------------------------------------
        rocsolver_getrf_getMemorySize<BATCHED, STRIDED, Tlu>(
            n, n, use_pivot, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
            &size_work4, &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &opt1);

        size_getrf = size_scalars + size_work1 + size_work2 + size_work3 + size_work4
            + size_pivotval + size_pivotidx + size_iipiv + size_iinfo;
    }

    // ----------------
    // workspace  for GETRS
    // ----------------
    {
        bool opt1 = true;
        bool opt2 = true;

        size_t w1 = 0;
        size_t w2 = 0;
        size_t w3 = 0;
        size_t w4 = 0;

        rocsolver_getrs_getMemorySize<BATCHED, STRIDED, Tlu>(rocblas_operation_none, n, nrhs,
                                                             batch_count, &w1, &w2, &w3, &w4, &opt2);

        size_t const size_getrs = w1 + w2 + w3 + w4;

        size_work += size_getrs;
    }

    // ------------------------------------
    // storage for mixed precision LU solver
    // ------------------------------------
    size_t size_getrf_mxp = 0;
    {
        auto const m = n;
        rocsolver_getrf_mxp_getMemorySize<Tlu, Treduced, I>(m, n, use_pivot, batch_count,
                                                            &size_getrf_mxp);
    }

    size_work += std::max(size_getrf, size_getrf_mxp);

    // ----------------------
    // storage for xnrm, rnrm
    // ----------------------
    {
        size_t const size_xnrm = sizeof(S) * batch_count * nrhs;
        size_t const size_rnrm = sizeof(S) * batch_count * nrhs;

        size_t const size_ixnrm = sizeof(I) * batch_count * nrhs;
        size_t const size_irnrm = sizeof(I) * batch_count * nrhs;

        // ---------------------------------------------------------------
        // TODO: not clear how much workspace is needed in rocblas_iamax()
        // ---------------------------------------------------------------
        size_t const size_iamax = 2 * sizeof(S*) * n * batch_count;

        size_work += size_xnrm;
        size_work += size_rnrm;

        size_work += size_ixnrm;
        size_work += size_irnrm;

        size_work += size_iamax;
    }

    {
        // ----------------
        // storage for GESV
        // ----------------

        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;

        size_t size_scalars = 0;
        size_t size_work0 = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        size_t size_pivotval = 0;
        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;
        bool optim_mem = true;

        rocsolver_gesv_getMemorySize<BATCHED, STRIDED, T>(
            n, nrhs, batch_count,

            &size_scalars, &size_work0, &size_work1, &size_work2, &size_work3, &size_work4,
            &size_pivotval,

            &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        size_t const size_gesv = size_scalars + size_work0 + size_work1 + size_work2 + size_work3
            + size_work4 + size_pivotval + size_pivotidx + size_iipiv + size_iinfo;

        size_work = std::max(size_work, size_gesv);
    }

    {
        // -----------------
        // check convergence
        // -----------------

        size_work += sizeof(bool);
    }

    *p_size_work = size_work;
}

template <typename T, typename Treduced, typename I, typename Istride>
rocblas_status rocsolver_zcgesv_mxp_template(rocblas_handle handle,
                                             I const n,
                                             I const nrhs,

                                             T* const A,
                                             Istride const shiftA,
                                             I const lda,
                                             Istride const strideA,

                                             I* const ipiv,
                                             Istride const strideP,

                                             T* const B,
                                             Istride const shiftB,
                                             I const ldb,
                                             Istride const strideB,

                                             T* const X,
                                             Istride const shiftX,
                                             I const ldx,
                                             Istride const strideX,

                                             I const max_iter_arg,
                                             double const tol_arg,
                                             I* const niter,

                                             I* info,
                                             I const batch_count,
                                             bool const use_pivot,

                                             void* const work,
                                             size_t const size_work)
{
    ROCSOLVER_ENTER("zcgesv_mxp", "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    *niter = 0;
    *info = 0;

    bool const has_work = (n >= 1) && (nrhs >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    bool constexpr is_complex = rocblas_is_complex<T>;

    using Sfull = typename std::conditional<is_complex, decltype(std::real(T{})), T>::type;
    using Tlu = typename std::conditional<is_complex, rocblas_complex_num<float>, float>::type;
    using Slu = decltype(std::real(Tlu{}));
    using Sreduced = decltype(std::real(Treduced{}));

    double const tol_default = std::numeric_limits<Sfull>::epsilon() * n;

    double const tol = (tol_arg <= 0) ? tol_default : tol_arg;

    I const max_iter_default = 30;
    I const max_iter = (max_iter_arg <= 0) ? max_iter_default : max_iter_arg;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    // ----------
    // reset info
    // ----------

    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(ceil(batch_count, BS1), 1, 1), dim3(BS1, 1, 1), 0,
                            stream, info, batch_count, 0);

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

#ifndef CHECK_MEM
#define CHECK_MEM(pfree)                                       \
    {                                                          \
        bool const is_mem_ok = (pfree <= (pwork + size_work)); \
        if(!is_mem_ok)                                         \
        {                                                      \
            return (rocblas_status_memory_error);              \
        }                                                      \
    }
#endif

    // ----------------------
    // allocate B_lu and A_lu
    // ----------------------

    auto const nrows_A = n;
    auto const ncols_A = n;

    auto const nrows_B = n;
    auto const ncols_B = nrhs;

    auto const nrows_X = nrows_B;
    auto const ncols_X = ncols_B;

    I const ldB_lu = nrows_B;
    Istride const strideB_lu = ldB_lu * ncols_B;
    size_t const size_B_lu = sizeof(Tlu) * strideB_lu * batch_count;
    Tlu* const B_lu = (Tlu*)pfree;
    pfree += size_B_lu;

    I const ldA_lu = nrows_A;
    Istride const strideA_lu = ldA_lu * ncols_A;
    size_t const size_A_lu = sizeof(Tlu) * strideA_lu * batch_count;
    Tlu* const A_lu = (Tlu*)pfree;
    pfree += size_A_lu;

    CHECK_MEM(pfree);

    Istride const shiftA_lu = 0;
    Istride const shiftB_lu = 0;

    // ----------------
    // copy B into B_lu
    // copy A into A_lu
    // ----------------
    {
        char const uplo = 'A';
        lacpy<I, Istride>(handle, uplo, nrows_A, ncols_A,

                          A, shiftA, lda, strideA,

                          A_lu, shiftA_lu, ldA_lu, strideA_lu,

                          batch_count);

        lacpy<I, Istride>(handle, uplo, nrows_B, ncols_B,

                          B, shiftB, ldb, strideB,

                          B_lu, shiftB_lu, ldB_lu, strideB_lu,

                          batch_count);
    }

    // ------------------------
    // perform LU factorization
    // ------------------------

    bool const use_mixed_precision = !std::is_same<Tlu, Treduced>::value;
    if(use_mixed_precision)
    {
        auto const pfree_saved = pfree;

        size_t const size_remain = (pwork + size_work) - pfree;

        I const inca = 1;

        Istride shiftP = 0;
        bool const use_pivot = true;

        auto const istat
            = rocsolver_getrf_mxp_template<Tlu, Treduced>(handle, nrows_A, ncols_A,

                                                          A_lu, shiftA_lu, inca, ldA_lu, strideA_lu,

                                                          ipiv, shiftP, strideP,

                                                          info, batch_count, use_pivot,

                                                          pfree, size_remain);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        pfree = pfree_saved;
    }
    else
    {
        auto const pfree_saved = pfree;

        // ------------------------------------------------------
        // use regular LU factorization (without mixed precision)
        // ------------------------------------------------------

        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;

        I const inca = 1;
        Istride const shiftP = 0;

        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        size_t size_pivotval = 0;
        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;
        bool optim_mem = true;

        rocsolver_getrf_getMemorySize<BATCHED, STRIDED, Tlu>(
            n, n, use_pivot, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
            &size_work4, &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        Tlu* const scalars = (Tlu*)pfree;
        pfree += size_scalars;
        Tlu* const work1 = (Tlu*)pfree;
        pfree += size_work1;
        Tlu* const work2 = (Tlu*)pfree;
        pfree += size_work2;
        Tlu* const work3 = (Tlu*)pfree;
        pfree += size_work3;
        Tlu* const work4 = (Tlu*)pfree;
        pfree += size_work4;
        Tlu* const pivotval = (Tlu*)pfree;
        pfree += size_pivotval;
        I* const pivotidx = (I*)pfree;
        pfree += size_pivotidx;
        I* const iipiv = (I*)pfree;
        pfree += size_iipiv;
        I* const iinfo = (I*)pfree;
        pfree += size_iinfo;

        CHECK_MEM(pfree);

        rocsolver_getrf_template<BATCHED, STRIDED, Tlu>(
            handle, n, n,

            A_lu, shiftA_lu, inca, ldA_lu, strideA_lu,

            ipiv, shiftP, strideP, info, batch_count, scalars, work1, work2, work3, work4, pivotval,
            pivotidx, iipiv, iinfo, optim_mem, use_pivot);

        pfree = pfree_saved;
    }

    // ------------------------------------
    // solve the system  A_lu * X_lu = B_lu,
    // where X_lu over-write B_lu
    // ------------------------------------

    auto solve_rhs = [=]() {
        auto pfree_local = pfree;

        I const inca = 1;
        I const incb = 1;

        rocblas_operation const trans = rocblas_operation_none;

        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        bool optim_mem = true;

        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;
        rocsolver_getrs_getMemorySize<BATCHED, STRIDED, Tlu, I>(trans, n, nrhs, batch_count,

                                                                &size_work1, &size_work2,
                                                                &size_work3, &size_work4, &optim_mem);

        Tlu* const work1 = (Tlu*)pfree_local;
        pfree_local += size_work1;
        Tlu* const work2 = (Tlu*)pfree_local;
        pfree_local += size_work2;
        Tlu* const work3 = (Tlu*)pfree_local;
        pfree_local += size_work3;
        Tlu* const work4 = (Tlu*)pfree_local;
        pfree_local += size_work4;

        CHECK_MEM(pfree_local);

        return (rocsolver_getrs_template<BATCHED, STRIDED, Tlu>(
            handle, trans, n, nrhs,

            A_lu, shiftA_lu, inca, ldA_lu, strideA_lu,

            ipiv, strideP,

            B_lu, shiftB_lu, incb, ldB_lu, strideB_lu,

            batch_count, work1, work2, work3, work4, optim_mem, use_pivot));
    }; // end solve_rhs()

    {
        auto const istat = solve_rhs();
        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }

    // --------------------
    // convert solution back to FP64
    // --------------------
    {
        char const uplo = 'A';
        lacpy(handle, uplo, nrows_B, ncols_B,

              B_lu, shiftB_lu, ldB_lu, strideB_lu,

              X, shiftX, ldx, strideX,

              batch_count);
    }
    // ---------------------
    // compute R = B - A * X
    // (1) R <- B
    // (2) R <- R - A * X
    // ---------------------
    I const nrows_R = nrows_B;
    I const ncols_R = ncols_B;

    I const ldr = n;
    Istride const strideR = ldr * ncols_R;
    Istride const shiftR = 0;

    size_t const size_R = sizeof(T) * strideR * batch_count;
    T* const R = (T*)pfree;
    pfree += size_R;

    CHECK_MEM(pfree);

    // ----------------------------------------------
    // compute residual using the latest version of X
    // the residual matrix R will be updated
    // ----------------------------------------------
    auto compute_residual = [=] {
        // ----------
        // (1) R <- B
        // ----------
        {
            char const uplo = 'A';
            lacpy(handle, uplo, n, nrhs,

                  B, shiftB, ldb, strideB,

                  R, shiftR, ldr, strideR,

                  batch_count);
        }

        // ------------------
        // (2) R <- R - A * X
        // ------------------
        {
            T alpha = -1;
            T beta = 1;

            I const mm = nrows_R;
            I const nn = ncols_R;
            I const kk = ncols_A;

            rocblas_operation const trans1 = rocblas_operation_none;
            rocblas_operation const trans2 = rocblas_operation_none;

            auto const istat = rocblasCall_gemm<T, I>(handle, trans1, trans2, mm, nn, kk,

                                                      &alpha,

                                                      A, shiftA, lda, strideA,

                                                      X, shiftX, ldx, strideX,

                                                      &beta,

                                                      R, shiftR, ldr, strideR,

                                                      batch_count, (T**)pfree);

            assert(istat == rocblas_status_success);
        }
    }; // end compute_residual()

    I iter = 0;
    bool is_all_converged = false;

    {
        bool* const d_is_all_converged = (bool*)pfree;
        pfree += sizeof(bool);
        check_convergence(handle, n, nrhs,

                          X, shiftX, ldx, strideX,

                          R, shiftR, ldr, strideR,

                          batch_count, tol, d_is_all_converged);

        HIP_CHECK(hipMemcpyAsync(&is_all_converged, d_is_all_converged, sizeof(bool),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        pfree = pfree - sizeof(bool);
    }

    if(is_all_converged)
    {
        *niter = iter;
        *info = 0;
        return (rocblas_status_success);
    }

    for(I iter = 0; iter < max_iter; iter++)
    {
        // ---------------------------
        // convert R from FP64 to FP32
        // ---------------------------
        {
            char const uplo = 'A';
            lacpy(handle, uplo, nrows_R, ncols_R,

                  R, shiftR, ldr, strideR,

                  B_lu, shiftB_lu, ldB_lu, strideB_lu,

                  batch_count);
        }

        // -------------------------
        // solve for "dx" correction
        // answer over-writes B_lu
        // -------------------------
        {
            auto const istat = solve_rhs();
            if(istat != rocblas_status_success)
            {
                return (istat);
            }
        }

        // ------------
        // update X <-  X + dx
        // dx is stored in B_lu
        // ------------
        auto update_X = [=] {
            // ------------------
            // update X <- X + dx
            // (1) R <- dx
            // (2) X <- X + R
            // ------------------
            {
                // -----------
                // (1) R <- dx
                // -----------
                char const uplo = 'A';
                lacpy(handle, uplo, nrows_R, ncols_R,

                      B_lu, shiftB_lu, ldB_lu, strideB_lu,

                      R, shiftR, ldr, strideR,

                      batch_count);
            }

            {
                // --------------
                // (2) X <- X + R
                // --------------
                Istride stride_alpha = 0;
                T alpha = 1;
                I const inc1 = 1;
                I const inc2 = 1;

                for(I irhs = 0; irhs < nrhs; irhs++)
                {
                    auto const istat
                        = rocblasCall_axpy(handle,

                                           n, &alpha, stride_alpha,

                                           R, shiftR, inc1, strideR,

                                           X + idx2D(0, irhs, ldx), shiftX, inc2, strideX,

                                           batch_count);
                    assert(istat == rocblas_status_success);
                } // end for irhs
            }
        }; // end update_X()

        update_X();

        // ---------------
        // compute residual R
        // using latest version of X
        // ---------------
        compute_residual();

        // ---------
        // B_lu <- R
        // ---------
        {
            char const uplo = 'A';
            lacpy(handle, uplo, nrows_B, ncols_B,

                  R, shiftR, ldr, strideR,

                  B_lu, shiftB_lu, ldB_lu, strideB_lu,

                  batch_count);
        }

        // -----------------------
        // compute correction "dx"
        // dx over-writes B_lu
        // -----------------------
        solve_rhs();

        // -----------------
        // check convergence
        // -----------------

        bool is_all_converged = false;

        {
            bool* const d_is_all_converged = (bool*)pfree;
            pfree += sizeof(bool);
            check_convergence(handle, n, nrhs,

                              X, shiftX, ldx, strideX,

                              R, shiftR, ldr, strideR,

                              batch_count, tol, d_is_all_converged);

            HIP_CHECK(hipMemcpyAsync(&is_all_converged, d_is_all_converged, sizeof(bool),
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            pfree = pfree - sizeof(bool);
        }

        if(is_all_converged)
        {
            *info = 0;
            *niter = iter;
            return (rocblas_status_success);
        }

    } // for iter

    //  ----------------------------------------------------------
    //  iterative refinement using LU in FP32 with mixed precision
    //  was not able to converge
    //  ----------------------------------------------------------

    *niter = -(max_iter + 1);

    // --------------
    // reset workspace
    // --------------
    pfree = pwork;

    {
        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;

        size_t size_scalars = 0;
        size_t size_work0 = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        size_t size_pivotval = 0;

        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;
        bool optim_mem = true;

        rocsolver_gesv_getMemorySize<BATCHED, STRIDED, T>(
            n, nrhs, batch_count,

            &size_scalars, &size_work0, &size_work1, &size_work2, &size_work3, &size_work4,

            &size_pivotval,

            &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        T* const scalars = (T*)pfree;
        pfree += size_scalars;

        T* const work0 = (T*)pfree;
        pfree += size_work0;

        T* const work1 = (T*)pfree;
        pfree += size_work1;

        T* const work2 = (T*)pfree;
        pfree += size_work2;

        T* const work3 = (T*)pfree;
        pfree += size_work3;

        T* const work4 = (T*)pfree;
        pfree += size_work4;

        T* const pivotval = (T*)pfree;
        pfree += size_pivotval;

        I* const pivotidx = (I*)pfree;
        pfree += size_pivotidx;

        I* const iipiv = (I*)pfree;
        pfree += size_iipiv;

        I* const iinfo = (I*)pfree;
        pfree += size_iinfo;

        CHECK_MEM(pfree);

        // ------
        // X <- B
        // ------
        {
            auto const uplo = 'A';
            lacpy(handle, uplo, nrows_B, ncols_B,

                  B, shiftB, ldb, strideB,

                  X, shiftX, ldx, strideX,

                  batch_count);
        }

        {
            auto const istat = rocsolver_gesv_template<BATCHED, STRIDED, T>(
                handle, n, nrhs,

                A, shiftA, lda, strideA,

                ipiv, strideP,

                X, shiftX, ldx, strideX,

                info, batch_count,

                scalars, work0, work1, work2, work3, work4,

                pivotval, pivotidx, iipiv, iinfo, optim_mem);

            if(istat != rocblas_status_success)
            {
                return (istat);
            }
        }
    }

    return (rocblas_status_success);
}

ROCSOLVER_END_NAMESPACE
