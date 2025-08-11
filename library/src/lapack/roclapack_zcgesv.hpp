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

#include "rocauxiliary_lacpy.hpp"

ROCSOLVER_BEGIN_NAMESPACE

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
    using S = decltype{std::real(T{})};

    bool constexpr BATCHED = true;
    bool constexpr STRIDD = true;

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

        size_t const size_getrs += w1 + w2 + w3 + w4;

        size_work += size_getrs;
    }

    // ------------------------------------
    // storage for mixed precision LU solver
    // ------------------------------------
    size_t size_getrf_mxp = 0;
    {
        auto const m = n;
        rocsolver_getrf_mxp_getMemorySize( Tlu, Treduced, I>(
				    m, n,  use_pivot, batch_count,
				    &size_getrf_mxp );
    }

    size_work += std::max(size_getrf, size_getrf_mxp);

    // ----------------------
    // storage for xnrm, rnrm
    // ----------------------
    {
        size_t const size_xnrm = sizeof(S) * batch_count;
        size_t const size_rnrm = sizeof(S) * batch_count;

        // ---------------------------------------------------------------
        // TODO: not clear how much workspace is needed in rocblas_iamax()
        // ---------------------------------------------------------------
        size_t const size_iamax = sizeof(S) * n * batch_count;

        size_work += size_xnrm;
        size_work += size_rnrm;
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

    *p_size_work = size_work;
}

template <typename T, typename Treduced, typename I, typename Istride>
rocblas_status rocsolver_zcgesv_mxp_template(rocblas_handle handle,
                                             const I n,
                                             const I nrhs,
                                             T* const A,
                                             const Istride shiftA,
                                             const I lda,
                                             const Istride strideA,

                                             I* ipiv,
                                             const Istride strideP,

                                             T* const B,
                                             const Istride shiftB,
                                             const I ldb,
                                             const Istride strideB,

                                             T* const X,
                                             const Istride shiftX,
                                             const I ldx,
                                             const Istride strideX,

                                             I const max_iter_arg,
                                             double const tol_arg,
                                             I* niter,

                                             I* info,
                                             const I batch_count,
                                             void* work,
                                             size_t size_work)
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

    using Sfull = std::conditional<is_complex, decltype(std::real(T{})), T>::type;
    using Tlu = std::conditional<is_complex, rocblas_complex_num<float>, float>::type;
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
    size_t const strideB_lu = ldB_lu * ncols_B;
    size_t const size_B_lu = sizeof(Tlu) * strideB_lu * batch_count;
    Tlu* const B_lu = (Tlu*)pfree;
    pfree += size_B_lu;

    I const lda_lu = nrows_A;
    size_t const strideA_lu = ldA_lu * ncols_A;
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
        lacpy(handle, uplo, nrows_A, ncols_A,

              A, shiftA, lda, strideA,

              A_lu, shiftA_lu, lda_lu, strideA_lu,

              batch_count);

        lacpy(handle, uplo, nrows_B, ncols_B,

              B, shiftB, ldb, strideB,

              B_lu, shiftB_lu, ldb_lu, strideB_lu,

              batch_count);
    }

    // ------------------------
    // perform LU factorization
    // ------------------------
    bool const use_pivot = true;

    bool const use_mixed_precision = !std::is_same<Tlu, Treduced>::value;
    if(use_mixed_precision)
    {
        auto const pfree_saved = pfree;

        size_t const size_remain = (pwork + size_work) - pfree;

        I const inca = 1;

        auto const istat = rocsolver_getrf_mxp_template<Tlu, Treduced>(
            handle, nrows_A, ncols_A,

            A_lu, shiftA_lu, inca, ldA_lu, strideA_lu,

            ipiv, shiftP, strideP, info, pfree, size_remain);

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
            handle, n, n, A, shiftA, inca, lda, strideA, ipiv, shiftP, strideP, info, batch_count,
            scalars, work1, work2, work3, work4, pivotval, pivotidx, iipiv, iinfo, optim_mem,
            use_pivot);

        pfree = pfree_saved;
    }

    // ------------------------------------
    // solve the system  A_lu * X_lu = B_lu,
    // where X_lu over-write B_lu
    // ------------------------------------

    auto solve_rhs = [=]() {
        auto const pfree_saved = pfree;

        I const inca = 1;
        I const incb = 1;

        rocblas_operation const trans = rocblas_operation_none;

        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        bool optim_mem = true;

        rocsolver_getrs_getMemorySize<BATCHED, STRIDED, Tlu, I>(
            rocblas_operation trans, n, nrhs, batch_count,

            &size_work1, &size_work2, &size_work3, &size_work4, &optim_mem);

        Tlu* const work1 = (Tlu*)pfree;
        pfree += size_work1;
        Tlu* const work2 = (Tlu*)pfree;
        pfree += size_work2;
        Tlu* const work3 = (Tlu*)pfree;
        pfree += size_work3;
        Tlu* const work4 = (Tlu*)pfree;
        pfree += size_work4;

        CHECK_MEM(pfree);

        rocsolver_getrs_template<BATCHED, STRIDED, Tlu>(handle, trans, n, nrhs,

                                                        A_lu, shiftA_lu, inca, lda_lu, strideA_lu,

                                                        ipiv, strideP,

                                                        B_lu, shiftB_lu, incb, ldb_lu, strideB_lu,

                                                        batch_count, work1, work2, work3, work4,
                                                        optim_mem, use_pivot);

        pfree = pfree_saved;
    }; // end solve_rhs()

    solve_rhs();

    // --------------------
    // convert solution back to FP64
    // --------------------
    {
        char const uplo = 'A';
        lacpy(handle, uplo, nrows_B, ncols_B,

              B_lu, shiftB_lu, ldb_lu, strideB_lu,

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

            auto const trans1 = rocblas_operation_none;
            auto const trans2 = rocblas_operation_none;

            auto const istat = rocblasCall_gemm(handle, trans1, trans2, mm, nn, kk,

                                                &alpha,

                                                A, shiftA, lda, strideA,

                                                X, shiftX, ldx, strideX,

                                                &beta,

                                                R, shiftR, ldr, strideR,

                                                batch_count, pfree);

            assert(istat == rocblas_status_success);
        }
    }; // end compute_residual()

    compute_residual();

    auto num_converged = [=]() -> I {
        auto const pfree_saved = pfree;

        I nconverged = 0;
        I const incx = 1;
        I const incr = 1;

        size_t const size_rnrm = sizeof(S) * batch_count;
        size_t const size_xnrm = sizeof(S) * batch_count;

        S* xnrm = (S*)pfree;
        pfree += size_xnrm;
        S* rnrm = (S*)pfree;
        pfree += size_rnrm;

        CHECK_MEM(pfree);

        std::vector<S> h_xnrm(batch_count);
        std::vector<S> h_rnrm(batch_count);

        for(I irhs = 0; irhs < nrhs; irhs++)
        {
            {
                auto const istat = rocblasCall_iamax(handle, X, shiftX + idx2D(0, irhs, ldx), incx,
                                                     strideX, xnrm, batch_count, (void*)pfree);
                assert(istat == rocblas_status_success);
            }

            {
                auto const istat = rocblasCall_iamax(handle, R, shiftR + idx2D(0, irhs, ldr), incr,
                                                     strideR, rnrm, batch_count, pfree);
                assert(istat == rocblas_status_success);
            }

            HIP_CHECK(hipMemcpyAsync(&(h_rnrm[0]), rnrm, size_rnrm, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipMemcpyAsync(&(h_xnrm[0]), xnrm, size_xnrm, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            for(I bid = 0; bid < batch_count; bid++)
            {
                bool const is_converged = (h_rnrm[bid] <= h_xnrm[bid] * tol);
                if(is_converged)
                {
                    nconverged++;
                }
            }

        } // end for irhs

        pfree = pfree_saved;
        return (nconverged);
    }; // end num_converged()

    I iter = 0;
    bool const is_all_converged = (num_converged() >= nrhs * batch_count);
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

                  B_lu, shiftB_lu, ldb_lu, strideB_lu,

                  batch_count);
        }

        // -------------------------
        // solve for "dx" correction
        // answer over-writes B_lu
        // -------------------------
        solve_rhs();

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

                      B_lu, shiftB_lu, ldb_lu, strideB_lu,

                      R, shfitR, ldr, strideR,

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

                  B_lu, shiftB_lu, ldb_lu, strideB_lu,

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
        bool const is_all_converged = (num_converged() >= nrhs * batch_count);
        if(is_all_converge)
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

        T* const size_scalars = (T*)pfree;
        pfree += size_scalars;
        T* const size_work0 = (T*)pfree;
        pfree += size_work0;
        T* const size_work1 = (T*)pfree;
        pfree += size_work1;
        T* const size_work2 = (T*)pfree;
        pfree += size_work2;
        T* const size_work3 = (T*)pfree;
        pfree += size_work3;
        T* const size_work4 = (T*)pfree;
        pfree += size_work4;

        T* const size_pivotval = (T*)pfree;
        pfree += size_pivotval;

        I* const size_pivotidx = (I*)pfree;
        pfree += size_pivotidx;
        I* const size_iipiv = (I*)pfree;
        pfree += size_iipiv;
        I* const size_iinfo = (I*)pfree;
        pfree += size_iinfo;

        CHECK_MEM(pfree);

        // ------
        // X <- B
        // ------
        {
            auto const uplo = 'A';
            lacpy(handle, uplo, nrows_B, ncolsB,

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
