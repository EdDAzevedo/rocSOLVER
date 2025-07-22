/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
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

#include "hip/hip_bf16.h"
#include "hip/hip_bfloat16.h"
#include "hip/hip_fp16.h"
#include <type_traits>

#include "rocblas.hpp"
#include "roclapack_getf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

#include "rocsolver_getrf.hpp"

#include "auxiliary/rocauxiliary_amax_matrix.hpp"
#include "auxiliary/rocauxiliary_complex2reim.hpp"
#include "auxiliary/rocauxiliary_lacpy.hpp"
#include "auxiliary/rocauxiliary_reim2complex.hpp"
#include "auxiliary/rocauxiliary_scale_and_convert.hpp"

ROCSOLVER_BEGIN_NAMESPACE

static bool constexpr use_out_of_place = true;

// ---------------------------------------------------------
// the block size may be a tuning parameter for optimization
// ---------------------------------------------------------
template <bool ISBATCHED, typename T, typename I>
static I getrf_mxp_get_blksize(I const n, bool const use_pivot)
{
    return (std::min(n, 1024));
}

/** Return the sizes of the different workspace arrays **/
template <bool BATCHED, bool STRIDED, typename T, typename Treduced, typename I>
void rocsolver_getrf_mxp_getMemorySize(const I m,
                                       const I n,
                                       const bool pivot,
                                       const I batch_count,

                                       size_t* p_size_work,
                                       const I lda = 1,
                                       const I inca = 1)
{
    *p_size_work = 0;

    // if quick return, no need of workspace
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    size_t size_work = 0;

    bool constexpr is_fp16 = std::is_same<Treduced, rocblas_half>::value
        || std::is_same<Treduced, __half>::value || std::is_same<Treduced, _Float16>::value;

    bool constexpr is_complex = rocblas_is_complex<T>;
    bool constexpr is_batched = (BATCHED || STRIDED);

    if(is_batched)
    {
        // work space for rocblas GEMM
        size_t const size_ptr_array = sizeof(T**) * batch_count * 3;
        size_work += size_ptr_array;
    }

    {
        // ----------------
        // memory for getrf
        // ----------------

        bool optim_mem = true;
        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        size_t size_pivotval = 0;
        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;

        auto const min_mn = std::min(m, n);
        auto const dim = min_mn;
        I const blk = getrf_mxp_get_blksize<ISBATCHED, T>(dim, pivot);

        auto const nn = std::min(blk, min_mn);
        rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T, I>(
            m, nn, pivot, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
            &size_work4, &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        size_t const size_getrf = size_scalars + size_work1 + size_work2 + size_work3 + size_work4
            + size_pivotval + size_pivotidx + size_iipiv + size_iinfo;

        size_work += size_getrf;
    }

    {
        size_t size_L21_re_chop = (sizeof(Treduced) * m * blk) * batch_count;
        size_t size_L21_im_chop = (is_complex) ? size_L21_re_chop : 0;

        size_t size_U12_re_chop = (sizeof(Treduced) * blk * n) * batch_count;
        size_t size_U12_re_chop = (is_complex) ? size_U12_re_chop : 0;

        size_work += size_L21_re_chop + size_L21_im_chop;
        size_work += size_U12_re_chop + size_U12_im_chop;
    }

    if(is_fp16)
    {
        // -----------------------------------------
        // TODO: fill in amount of scratch space needed for
        // amax and scaling for limited dynamic range of FP16
        // -----------------------------------------
    }

    if(!use_out_of_place)
    {
        // -------------------------------
        // TODO: fill in amount of scratch space needed for
        // in-place conversion
        // -------------------------------
    }

    if(is_complex)
    {
        // ---------------------------------------
        // use out of place for complex conversion
        // ---------------------------------------
        {
            size_t const size_A22_re = (use_out_of_place) ? (sizeof(T) * m * n) * batch_count : 0;
            size_t const size_A22_im = size_A22_re;

            size_work += size_A22_re;
            size_work += size_A22_im;
        }
    }

    if(is_fp16)
    {
        // ------------------------------------------
        // storage for L21_re, L21_im, U12_re, U12_im
        // ------------------------------------------
        size_t const size_L21_re = sizeof(T) * m * blk * batch_count;
        size_t const size_L21_im = (is_complex) ? size_L21_re : 0;

        size_t const size_U12_re = sizeof(T) * blk * n * batch_count;
        size_t const size_U12_im = (is_complex) ? size_U12_re : 0;

        size_work += size_L21_re + size_L21_im;
        size_work += size_U12_re + size_U12_im;
    }

    *p_size_work = size_work;
}

#ifndef CHECK_MEM
#define CHECK_MEM(pfree)                      \
    {                                         \
        assert(pfree <= (pwork + size_work)); \
    }
#endif

template <typename T>
static rocblas_datatype type2datatype(T** A)
{
    return (type2datatype(T{}));
}

template <typename T>
static rocblas_datatype type2datatype(T* A)
{
    return (type2datatype<T>(T{}));
}

template <typename T>
static rocblas_datatype type2datatype(T A)
{
    return (std::is_same<T, float>::value                            ? rocblas_datatype_f32_r
                : std::is_same<T, rocblas_complex_num<float>>::value ? rocblas_datatype_f32_c
                : std::is_same<T, std::complex<float>>::value        ? rocblas_datatype_f32_c
                :

                std::is_same<T, double>::value                        ? rocblas_datatype_f64_r
                : std::is_same<T, rocblas_complex_num<double>>::value ? rocblas_datatype_f64_c
                : std::is_same<T, std::complex<double>>::value        ? rocblas_datatype_f64_c
                :

                std::is_same<T, rocblas_half>::value ? rocblas_datatype_f16_r
                : std::is_same<T, _Float16>::value   ? rocblas_datatype_f16_r
                : std::is_same<T, __half>::value     ? rocblas_datatype_f16_r
                :

                std::is_same<T, rocblas_complex_num<rocblas_half>>::value ? rocblas_datatype_f16_c
                : std::is_same<T, rocblas_complex_num<_Float16>>::value   ? rocblas_datatype_f16_c
                : std::is_same<T, rocblas_complex_num<__half>>::value     ? rocblas_datatype_f16_c
                :

                std::is_same<T, std::complex<rocblas_half>>::value ? rocblas_datatype_f16_c
                : std::is_same<T, std::complex<_Float16>>::value   ? rocblas_datatype_f16_c
                : std::is_same<T, std::complex<__half>>::value     ? rocblas_datatype_f16_c
                :

                std::is_same<T, rocblas_bfloat16>::value ? rocblas_datatype_bf16_r
                : std::is_same<T, __bf16>::value         ? rocblas_datatype_bf16_r
                :

                std::is_same<T, rocblas_complex_num<rocblas_bfloat16>>::value ? rocblas_datatype_bf16_c
                : std::is_same<T, rocblas_complex_num<__bf16_t>>::value  ? rocblas_datatype_bf16_c
                : std::is_same<T, std::complex<rocblas_bfloat16>>::value ? rocblas_datatype_bf16_c
                : std::is_same<T, std::complex<__bf16_t>>::value         ? rocblas_datatype_bf16_c
                                                                         :

                                                                 rocblas_datatype_invalid);
}

// ------------------------------------------------------------------------
// wrapper to prepare call to rocblas_gemm_ex() + batched + strided_batched
// ------------------------------------------------------------------------
template <typename T, typename Treduced, typename I, typename Istride>
static rocblas_status void rocblasCall_gemm_ex(

    rocblas_handle handle,
    rocblas_operation const trans_a,
    rocblas_operation const trans_b,

    I const m,
    I const n,
    I const k,

    T const* const alpha,
    Istride const stride_alpha,

    Treduced const* const A,
    Istride const offset_a,
    I const ld_a,
    Istride const stride_a,

    Treduced const* const B,
    Istride const offset_b,
    I const ld_b,
    Istride const stride_b,

    T const* const beta,
    Istride const stride_beta,

    T* const C,
    Istride const offset_c,
    I const ld_c,
    Istride const stride_c,

    I const batch_count,
    void* work)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("rocblasCall_gemm_ex", "transA:", trans_a, "transB:", trans_b, "m:", m, "n:", n,
                  "k:", k, "shiftA:", offset_a, "lda:", ld_a, "shiftB:", offset_b, "ldb:", ld_b,
                  "shiftC:", offset_c, "ldc:", ld_c, "bc:", batch_count);

    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    int32_t solution_index = 0;
    uint32_t flags = rocblas_gemm_flags_none;

    rocblas_datatype const a_type = type2datatype(Treduced{});
    rocblas_datatype const b_type = type2datatype(Treduced{});
    rocblas_datatype const c_type = type2datatype(T{});
    rocblas_datatype const d_type = c_type;

    auto const compute_type = c_type;
    rocblas_gemm_algo const algo = rocblas_gemm_algo_standard;

    // ---------------------------------------------------------------
    // compute   C(i) = beta(i) * C(i) + alpha(i) * A(i) * B(i), for i=0:(batch_count)
    // each batch item may have a different "alpha(i)" or "beta(i)"
    //
    // TODO: consider using hipBLASLt for this capability
    // ---------------------------------------------------------------
    bool const need_for_loop = (stride_alpha != 0) || (stride_beta != 0);
    if(need_for_loop)
    {
        for(I bid = 0; bid < batch_count; bid++)
        {
            auto const Ap = A + bid * stride_a;
            auto const Bp = B + bid * stride_b;
            auto const Cp = C + bid * stride_c;

            auto const Dp = Cp;
            auto const d_type = c_type;
            auto const ld_d = ld_c;

            auto const istat
                = rocblas_gemm_ex(handle, trans_a, trans_b, m, n, k, alpha + bid * stride_alpha,

                                  Ap, a_type, ld_a,

                                  Bp, b_type, ld_b,

                                  beta + bid * stride_beta,

                                  Cp, c_type, ld_c,

                                  Dp, d_type, ld_d,

                                  compute_type, algo, solution_index, flags);

            if(istat != rocblas_status_success)
            {
                return (istat);
            }

        } // end for bid
    }
    else
    {
        // ------------------------------------
        // all A and B and C are strided_batched
        // so use the strided batched version
        // of rocblas_gemm_strided_batched_ex()
        // ------------------------------------

        auto D = C;
        auto const d_type = c_type;
        auto const ld_c = ld_d;
        auto const stride_d = stride_c;

        auto const istat = rocblas_gemm_strided_batched_ex(handle, trans_a, trans_b, m, n, k, alpha,

                                                           A, a_type, ld_a, stride_a,

                                                           B, b_type, ld_b, stride_b,

                                                           beta,

                                                           C, c_type, ld_c, stride_c,

                                                           D, d_type, ld_d, stride_d,

                                                           batch_count, compute_type, algo,
                                                           solution_index, flags);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }

    return (rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename Treduced, typename I, typename INFO, typename UA, typename UA_S>
rocblas_status rocsolver_getrf_mxp_template(rocblas_handle handle,
                                            const I m,
                                            const I n,
                                            T* const A,
                                            const rocblas_stride shiftA,
                                            const I inca,
                                            const I lda,
                                            const rocblas_stride strideA,
                                            I* ipiv,
                                            const rocblas_stride shiftP,
                                            const rocblas_stride strideP,
                                            INFO* const info,
                                            const I batch_count,
                                            const bool pivot,
                                            void* work,
                                            size_t const size_work)
{
    ROCSOLVER_ENTER("getrf_mxp", "m:", m, "n:", n, "shiftA:", shiftA, "inca:", inca, "lda:", lda,
                    "shiftP:", shiftP, "bc:", batch_count);

    bool constexpr is_complex = rocblas_is_complex<T>;

    using Smax = decltype(1.0);
    using S = std::conditional< is_complex, decltype(std::real(T{}), T >::type;

    I const ldA = lda;
    I const fp16_max = 65504; // max valid representable value in FP16
    I const fp16_max_m1 = fp16_max - 1;

    bool constexpr is_fp16
        = std::is_same<Treduced,
                       __half> || std::is_same<Treduced, _Float16> || std::is_same<Treduced, rocblas_half>;


    // quick return
    if(batch_count == 0)
    {
        return rocblas_status_success;
    }

    auto ceil = [](auto n, auto b) {
        return ((n - 1) / b + 1); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;
    I dim = std::min(m, n);
    I blocks = 0, blocksy = 0;
    dim3 grid, threads;

    // quick return if no dimensions
    if(m == 0 || n == 0)
    {
        blocks = ceil(batch_count, BS1);
        grid = dim3(blocks, 1, 1);
        threads = dim3(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, info, batch_count, 0);
        return rocblas_status_success;
    }

    // size of outer blocks
    I const blk = getrf_mxp_get_blksize<ISBATCHED, T>(dim, pivot);

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    T* scalars = nullptr;
    void* work1 = nullptr;
    void* work2 = nullptr;
    void* work3 = nullptr;
    void* work4 = nullptr;
    T* pivotval = nullptr;
    I* pivotidx = nullptr;
    INFO* iinfo = nullptr;
    I* iipiv = nullptr;

    {
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

        I const min_mn = std::min(m, n);
        I const dim = min_mn;
        I const nn = std::min(blk, dim);
        rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T, I>(

            m, nn, pivot, batchcount,

            &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4, &size_pivotval,
            &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        scalars = (T*)pfree;
        pfree += size_scalars;
        work1 = (void*)pfree;
        pfree += size_work1;
        work2 = (void*)pfree;
        pfree += size_work2;
        work3 = (void*)pfree;
        pfree += size_work3;
        work4 = (void*)pfree;
        pfree += size_work4;

        pivotval = (T*)pfree;
        pfree += size_pivotval;
        pivotidx = (I*)pfree;
        pfree += size_pivotidx;
        iipiv = (I*)pfree;
        pfree += size_iipiv;
        iinfo = (INFO*)pfree;
        pfree += size_iinfo;

        CHECK_MEM(pfree);
    }

    if(blk == 0)
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, inca, lda, strideA,
                                                      ipiv, shiftP, strideP, info, batch_count,
                                                      scalars, pivotval, pivotidx, pivot);


    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T one = 1;
    T minone = -1;

    I jb, dimx, dimy;
    I nextpiv, mm, nn;
    size_t lmemsize;
    I j = 0;

    // in the npvt cases, panel determines whether the whole block-panel or only the
    // diagonal block is factorized
    bool panel = false;
    if(blk < 0)
    {
        panel = true;
        blk = -blk;
    }

    // MAIN LOOP
    for(I j = 0; j < dim; j += blk)
    {
        jb = std::min(dim - j, blk);

        if(pivot || panel)
        {
            // factorize outer block panel
            getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j, jb, n, A, shiftA + j * inca, inca,
                                               lda, strideA, ipiv, shiftP + j, strideP, info,
                                               batch_count, pivot, scalars, work1, work2, work3,
                                               work4, optim_mem, pivotval, pivotidx, j, iipiv, m);
        }
        else
        {
            // factorize only outer diagonal block
            getrf_panelLU<BATCHED, STRIDED, T>(handle, jb, jb, n, A, shiftA + j * inca, inca, lda,
                                               strideA, ipiv, shiftP + j, strideP, info,
                                               batch_count, pivot, scalars, work1, work2, work3,
                                               work4, optim_mem, pivotval, pivotidx, j, iipiv, m);

            // update remaining rows in outer panel
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
                m - j - jb, jb, A, shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
                shiftA + idx2D(jb + j, j, inca, lda), inca, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);
        }

        // update trailing matrix
        nextpiv = j + jb; //position for the matrix update
        mm = m - nextpiv; //size for the matrix update
        nn = n - nextpiv; //size for the matrix update
        if(nextpiv < n)
        {
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb, nn, A,
                shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
                shiftA + idx2D(j, nextpiv, inca, lda), inca, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);

            if(nextpiv < m)
            {
#if(0)
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mm, nn, jb,
                               &minone,

                               A, shiftA + idx2D(nextpiv, j, inca, lda), inca, lda, strideA,

                               A, shiftA + idx2D(j, nextpiv, inca, lda), inca, lda, strideA,

                               &one,

                               A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), inca, lda, strideA,

                               batch_count, (T**)nullptr);
#endif

                auto L21 = A;
                auto const shift_L21 = shiftA + idx2D(nextpiv, j, inca, lda);
                auto const ldL21 = lda;
                auto const stride_L21 = strideA;

                auto U12 = A;
                auto const shift_U12 = shiftA + idx2D(j, nextpiv, inca, lda);
                auto const ldU12 = lda;
                auto const stride_U12 = strideA;

                I const nrows_L21 = mm;
                I const ncols_L21 = jb;

                I const ldL21_chop = nrows_L21;
                Istride const stride_L21_chop = ldL21_chop * ncols_L21;
                size_t const size_L21_chop = sizeof(Treduced) * strideL_L21_chop * batch_count;

                auto const size_L21_re_chop = size_L21_chop;
                auto const size_L21_im_chop = size_L21_chop;

                I const nrows_U12 = jb;
                I const ncols_U12 = nn;

                I const ldU12_chop = nrows_U12;
                Istride const strie_U12_chop = ldU12_chop * ncols_U12;

                size_t const size_U12_chop = sizeof(Treduced) * stride_U12_chop * batch_count;

                auto const size_U12_re_chop = size_U12_chop;
                auto const size_U12_im_chop = size_U12_chop;

                // A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), inca, lda, strideA,
                auto const nrows_A22_re = mm;
                auto const ncols_A22_re = nn;
                auto const nrows_A22_im = nrows_A_re;
                auto const ncols_A22_im = ncols_A_im;

                auto const A22 = A;
                auto const shift_A22 = shiftA + idx2D(nextpiv, nextpiv, inc, lda);
                auto const ldA22 = lda;
                auto const stride_A22 = strideA;

                // ------------------------------
                // assume out of place conversion
                // ------------------------------

                auto const ldA22_re = (use_out_of_place) ? nrows_A22 : ldA22;
                auto const ldA22_im = ldA22_re;

                auto const shift_A22_re = (use_out_of_place) ? 0 : shift_A22;
                auto const shift_A22_im = shift_A22_re;

                auto const stride_A22_re = (use_of_out_place)
                    ? (ldA22_re * ncols_A22_re)
                    : strideA auto const stride_A22_im = stride_A22_re;

                if(!is_complex)
                {
                    // --------------------------------------------
                    // need to convert to L21_chop and U12_chop to
                    // reduced precision FP16 or BF16
                    // to use rocblas_gemm_ex()
                    // --------------------------------------------

                    auto const pfree_save = pfree;

                    // ------------------------------
                    // allocate L21_chop and U12_chop
                    // ------------------------------

                    Treduced* const L21_chop = (Treduced*)pfree;
                    pfree += size_L21_chop;

                    Treduced* const U12_chop = (Treduced*)pfree;
                    pfree += size_U12_chop;

                    if(is_fp16)
                    {
                        // ---------------------------------------
                        // need to scale and convert L21 and U12
                        // to fit in limited dynamic range of FP16
                        // ---------------------------------------

                        Istride const stride_amax_U12 = 1;
                        size_t const size_amax_U12 = sizeof(Smax) * stride_amax_U12 * batch_count;
                        Smax* const amax_U12 = pfree;
                        pfree += size_amax_U12;

                        Istride const stride_amax_L21 = 1;
                        size_t const size_amax_L21 = sizeof(Smax) * stride_amax_L21 * batch_count;
                        Smax* const amax_L21 = pfree;
                        pfree += size_amax_L21;

                        CHECK_MEM(pfree);

                        // --------------------------------------
                        // set amax_U12, amax_L21 to zero
                        // this is necessary for correctness when
                        // calling amax_matrix() for computing
                        // the max absolute value of L21 and U12
                        // --------------------------------------
                                HIP_CHECK( hipMemsetAsync( (void *) amax_U12, 0, size_amax_U12, stream );
			        HIP_CHECK( hipMemsetAsync( (void *) amax_L21, 0, size_amax_L21, stream );


			amax_matrix( stream, nrows_L21, ncols_L21,
					amax_L21, stride_amax_L21,

					L21, shift_L21, ldL21, stride_L21,

				        batch_count, (void *) pfree  );

			amax_matrix( stream, nrows_U12, ncols_U12,
					amax_U12,

					U12, shift_U12, ldU12, stride_U12,

					batch_count, (void *) pfree );

			// ------------------------------------------------
			// scale and convert to fit in limited dynamic range
			// of FP16
			// ------------------------------------------------


			scale_and_convert( stream,
					nrows_L21, ncols_L21,
					amax_L21, stride_amax_L21,
					fp16_max_m1,

					L21, shift_L21, ldL21, stride_L21,

					L21_chop, shift_L21_chop, ldL21_chop, stride_L21_chop,

					batch_count );

			scale_and_convert( stream,
					nrows_U12, ncols_U12,
					amax_U12, stride_amax_U12,
					fp16_max_m1,

					U12, shift_U12, ldU12, stride_U12,

					U12_chop, shift_U12_chop, ldU12_chop, stride_U12_chop,

					batch_count );
                    }
                    else
                    {
                        // ------------------------------------
                        // Bfloat16 is used,  so no need to rescale
                        // to fit in limited dynamic range
                        // ------------------------------------

                        char const uplo = 'A';
                        lacpy(stream, uplo, nrows_L21, ncols_L21,

                              L21, shift_L21, ldL21, shift_L21,

                              L21_chop, shift_L21_chop, ldL21_chop, shift_L21_chop,

                              batch_count);

                        lacpy(stream, uplo, nrows_U12, ncols_U12,

                              U12, shift_U12, ldU12, shift_U12,

                              U12_chop, shift_U12_chop, ldU12_chop, shift_U12_chop,

                              batch_count);
                    }

                    // ----------------------
                    // perform GEMM operation
                    // ----------------------

                    {
                        rocblas_operation const trans_a = rocblas_operation_none;
                        rocblas_operation const trans_b = rocblas_operation_none;

                        Istride const stride_alpha = 0;
                        Istride const stride_beta = 0;
                        I const kk = jb;
                        auto const istat = rocblasCall_gemm_ex(
                            handle, trans_a, trans_b, mm, nn, kk, &minone, stride_alpha,

                            L21_chop, shift_L21_chop, ldL21_chop, stride_L21_chop,

                            U12_chop, shift_U12_chop, ldU12_chop, stride_U12_chop,

                            &one, stride_beta,

                            // A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), lda, strideA,
                            A22, shift_A22, ldA22, stride_A22,

                            batch_count, (void*)pfree);

                        if(istat != rocblas_status_success)
                        {
                            return (istat);
                        }
                    }

                    // --------------------------------------
                    // deallocate temporary storage by
                    // resetting the value of variable "pfree"
                    // --------------------------------------
                    pfree = pfree_saved;
                }
                else
                {
                    // ------------------------------
                    // complex version, need to split
                    // into real and imaginary parts
                    // ------------------------------

                    auto const pfree_saved = pfree;
                    // -------------------------------
                    // create L21_re_chop, L21_im_chop
                    // and U12_re_chop, U12_im_chop
                    // -------------------------------

                    // ---------------------------------------------------------
                    // (1) create L21_re, L21_im, using complex2reim_outofplace()
                    //            U12_re, U12_im
                    // (2) compute amax_L21_re, amax_L21_im, using amax_matrix()
                    //             amax_U12_re, amax_U12_im
                    // (3) form L21_re_chop, L21_im_chop, using scale_and_convert()
                    //          U12_re_chop, U12_im_chop
                    // ---------------------------------------------------------

                    Treduced* const L21_re_chop = pfree;
                    pfree += size_L21_re_chop;
                    Treduced* const L21_im_chop = pfree;
                    pfree += size_L21_im_chop;

                    Treduced* const U12_re_chop = pfree;
                    pfree += size_U12_re_chop;
                    Treduced* const U12_im_chop = pfree;
                    pfree += size_U12_im_chop;

                    CHECK_MEM(pfree);

                    // -------------------------
                    // (1) create L21_re, L21_im
                    //            U12_re, U12_im
                    // -------------------------
                    I const ldL21_re = nrows_L21;
                    I const ldL21_im = nrows_L21;
                    Istride const stride_L21_re = ldL21_re * ncols_L21;
                    Istride const stride_L21_im = ldL21_im * ncols_L21;

                    size_t const size_L21_re = sizeof(T) * stride_L21_re * batch_count;
                    size_t const size_L21_im = sizeof(T) * stride_L21_im * batch_count;

                    T* const L21_re = pfree;
                    pfree += size_L21_re;
                    T* const L21_im = pfree;
                    pfree += size_L21_im;

                    CHECK_MEM(pfree);

                    complex2reim_outofplace(stream, nrows_L21, ncols_L21, L21, shift_L21, ldL21,
                                            stride_L21,

                                            L21_re, shift_L21_re, ldL21_re, stride_L21_re, L21_im,
                                            shift_L21_im, ldL21_im, stride_L21_re,

                                            batch_count);

                    I const ldU12_re = nrows_U12;
                    I const ldU12_im = nrows_U12;
                    Istride const stride_U12_re = ldU12_re * ncols_U12;
                    Istride const stride_U12_im = ldU12_im * ncols_U12;

                    size_t const size_U12_re = sizeof(T) * stride_U12_re * batch_count;
                    size_t const size_U12_im = sizeof(T) * stride_U12_im * batch_count;

                    T* const U12_re = pfree;
                    pfree += size_U12_re;
                    T* const U12_im = pfree;
                    pfree += size_U12_im;

                    CHECK_MEM(pfree);

                    complex2reim_outofplace(stream, nrows_U12, ncols_U12, U12, shift_U12, ldU12,
                                            stride_U12,

                                            U12_re, shift_U12_re, ldU12_re, stride_U12_re, U12_im,
                                            shift_U12_im, ldU12_im, stride_U12_re,

                                            batch_count);

                    Smax const dlimit = (is_fp16) ? fp16_max_m1 : 1;

                    Istride const stride_amax_L21_re = 1;
                    Istride const stride_amax_L21_im = 1;
                    Istride const stride_amax_U12_re = 1;
                    Istride const stride_amax_U12_im = 1;

                    size_t const size_amax_L21_re = sizeof(Smax) * stride_amax_L21_re * batch_count;
                    size_t const size_amax_L21_im = sizeof(Smax) * stride_amax_L21_im * batch_count;
                    size_t const size_amax_U12_re = sizeof(Smax) * stride_amax_U12_re * batch_count;
                    size_t const size_amax_U12_im = sizeof(Smax) * stride_amax_U12_im * batch_count;

                    Smax* const amax_L21_re = (is_fp16) ? pfree : nullptr;
                    pfree = (is_fp16) ? pfree + size_amax_L21_re : pfree;

                    Smax* const amax_L21_im = (is_fp16) ? pfree : nullptr;
                    pfree = (is_fp16) ? pfree + size_amax_L21_im : pfree;

                    Smax* const amax_U12_re = (is_fp16) ? pfree : nullptr;
                    pfree = (is_fp16) ? pfree + size_amax_U12_re : pfree;

                    Smax* const amax_U12_im = (is_fp16) ? pfree : nullptr;
                    pfree = (is_fp16) ? pfree + size_amax_U12_im : pfree;

                    CHECK_MEM(pfree);

                    if(is_fp16)
                    {
                        auto const pfree_saved_chop = pfree;

                        // --------------------------------------
                        // set amax_U12_re, amax_U12_im,
                        // amax_L21_re, amax_L21_im to zero
                        // this is necessary for correctness when
                        // calling amax_matrix() for computing
                        // the max absolute value of L21 and U12
                        // --------------------------------------
                        HIP_CHECK(hipMemsetAsync((void*)amax_L21_re, 0, size_amax_L21_re, stream));
                        HIP_CHECK(hipMemsetAsync((void*)amax_L21_im, 0, size_amax_L21_im, stream));
                        HIP_CHECK(hipMemsetAsync((void*)amax_U12_re, 0, size_amax_U12_re, stream));
                        HIP_CHECK(hipMemsetAsync((void*)amax_U12_im, 0, size_amax_U12_im, stream));

                        // ----------------------------------------------------------------
                        // Note: assume no scratch storage needed for calling amax_matrix()
                        // ----------------------------------------------------------------
                        amax_matrix(handle, nrows_L21, ncols_L21, amax_L21_re, stride_amax_L21_re,
                                    L21_re, shift_L21_re, ldL21_re, stride_L21_re, batch_count);

                        amax_matrix(handle, nrows_L21, ncols_L21, amax_L21_im, stride_amax_L21_im,
                                    L21_im, shift_L21_im, ldL21_im, stride_L21_im, batch_count);

                        amax_matrix(handle, nrows_U12, ncols_U12, amax_U12_re, stride_amax_U12_re,
                                    U12_re, shift_U12_re, ldU12_re, stride_U12_re, batch_count);

                        amax_matrix(handle, nrows_U12, ncols_U12, amax_U12_im, stride_amax_U12_im,
                                    U12_im, shift_U12_im, ldU12_im, stride_U12_im, batch_count);

                        scale_and_convert(handle, nrows_L21, ncols_L21, amax_L21_re,
                                          stride_amax_L21_re, dlimit, L21_re, shift_L21_re,
                                          ldL21_re, stride_L21_re, L21_re_chop, shift_L21_re_chop,
                                          ldL21_re_chop, stride_L21_re_chop, batch_count);

                        scale_and_convert(handle, nrows_L21, ncols_L21, amax_L21_im,
                                          stride_amax_L21_im, dlimit, L21_im, shift_L21_im,
                                          ldL21_im, stride_L21_im, L21_im_chop, shift_L21_im_chop,
                                          ldL21_im_chop, stride_L21_im_chop, batch_count);

                        scale_and_convert(handle, nrows_U12, ncols_U12, amax_U12_re,
                                          stride_amax_U12_re, dlimit, U12_re, shift_U12_re,
                                          ldU12_re, stride_U12_re, U12_re_chop, shift_U12_re_chop,
                                          ldU12_re_chop, stride_U12_re_chop, batch_count);

                        scale_and_convert(handle, nrows_U12, ncols_U12, amax_U12_im,
                                          stride_amax_U12_im, dlimit, U12_im, shift_U12_im,
                                          ldU12_im, stride_U12_im, U12_im_chop, shift_U12_im_chop,
                                          ldU12_im_chop, stride_U12_im_chop, batch_count);

                        pfree = pfree_saved_chop;
                    }
                    else
                    {
                        // ------------------------------
                        // use bfloat16, no need to scale
                        // ------------------------------

                        char const uplo = 'A';

                        lacpy(handle, uplo, nrows_L21, ncols_L21,

                              L21_re, shift_L21_re, ldL21_re, stride_L21_re,

                              L21_re_chop, shift_L21_re_chop, ldL21_re_chop, stride_L21_re_chop,

                              batch_count);

                        lacpy(handle, uplo, nrows_L21, ncols_L21,

                              L21_im, shift_L21_im, ldL21_im, stride_L21_im,

                              L21_im_chop, shift_L21_im_chop, ldL21_im_chop, stride_L21_im_chop,

                              batch_count);

                        lacpy(handle, uplo, nrows_U12, ncols_U12,

                              U12_re, shift_U12_re, ldU12_re, stride_U12_re,

                              U12_re_chop, shift_U12_re_chop, ldU12_re_chop, stride_U12_re_chop,

                              batch_count);

                        lacpy(handle, uplo, nrows_U12, ncols_U12,

                              U12_im, shift_U12_im, ldU12_im, stride_U12_im,

                              U12_im_chop, shift_U12_im_chop, ldU12_im_chop, stride_U12_im_chop,

                              batch_count);

                    } // end if (is_fp16)

                    // ---------------------------
                    // use out-of-place conversion
                    // to form A_re, A_im
                    // ---------------------------

                    size_t const size_A22_re
                        = (use_out_of_place) ? sizeof(T) * stride_A22_re * batch_count : 0;
                    size_t const size_A22_im
                        = (use_out_of_place) ? sizeof(T) * stride_A22_im * batch_count : 0;

                    T* const A22_re = (use_out_of_place) ? (T*)pfree : A;
                    pfree += size_A22_re;

                    T* const A22_im = (use_out_of_place) ? (T*)pfree : A;
                    pfree += size_A22_im;

                    CHECK_MEM(pfree);

                    complex2reim_outofplace(
                        handle, nrows_A22_re, ncols_A22_re,

                        // A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), lda, strideA,
                        A22, shift_A22, ldA22, stride_A22,

                        A22_re, shift_A22_re, ldA22_re, stride_A22_re,

                        A22_im, shift_A22_im, ldA22_im, stride_A22_im,

                        batch_count);

                    /*
	% -------------------------------------------------------------
	% A22( i3:m, j3:n ) = A22( i3:m, j3:n) - single(L21) * single(U12);
	% -------------------------------------------------------------



	% ----------------------------------
	% Use 4 GEMM to emulate complex GEMM
	% ----------------------------------
	% (A22_re + J * A22_im) = (A22_re + J * A22_im) -
	%                     (L21_re + J * L21_im) * (U12_re + J * U12_im )
	%
	% A22_re = A22_re - ( L21_re * U12_re - L21_im * U12_im )
	% A22_im = A22_im - ( L21_re * U12_im + L21_im * U12_re )
	%
	% A22_re = A22_re - L21_re * U12_re
	% A22_re = A22_re + L21_im * U12_im
	%
	% A22_im = A22_im - L21_re * U12_im
	% A22_im = A22_im - L21_im * U12_re
	%  ------------------------------
*/

                    double const dlimit_sq = dlimit * dlimit;

                    std::vector<double> h_amax_L21_re(batch_count);
                    std::vector<double> h_amax_L21_im(batch_count);
                    std::vector<double> h_amax_U12_re(batch_count);
                    std::vector<double> h_amax_U12_im(batch_count);
                    /*
	------------------------------------
	% A22_im = A22_im - L21_re * U12_im
	% A22_im = A22_im - L21_im * U12_re
	------------------------------------
*/

                    // --------------------------------------------
                    // copy the amax arrays to host
                    // to compute the alpha(i) for each batch case
                    // --------------------------------------------
                    if(is_fp16)
                    {
                        HIP_CHECK(hipMemcpyAsync(&(h_amax_L21_re[0]), amax_L21_re, size_amax_L21_re,
                                                 hipMemcpyDeviceToHost, stream));
                        HIP_CHECK(hipMemcpyAsync(&(h_amax_L21_im[0]), amax_L21_im, size_amax_L21_im,
                                                 hipMemcpyDeviceToHost, stream));
                        HIP_CHECK(hipMemcpyAsync(&(h_amax_U12_re[0]), amax_U12_re, size_amax_U12_re,
                                                 hipMemcpyDeviceToHost, stream));
                        HIP_CHECK(hipMemcpyAsync(&(h_amax_U12_im[0]), amax_U12_im, size_amax_U12_im,
                                                 hipMemcpyDeviceToHost, stream));

                        HIP_CHECK(hipStreamSynchronize(stream));
                    }

                    if(is_fp16)
                    {
                        /*
	alpha = single( (amax_L21_re / dlimit_sq) * amax_U12_re );
	A22_re( i3:m, j3:n) = A22_re( i3:m, j3:n ) - ...
		alpha * single(L21_re_chop) * single(U12_re_chop);
*/

                        std::vector<S> h_alpha(batch_count);
                        for(I bid = 0; bid < batch_count; bid++)
                        {
                            h_alpha[bid] = -((h_amax_L21_re[bid] / dlimit_sq) * h_amax_U12_re[bid]);
                        }
                        Istride const stride_alpha = 1;
                        Istride const stride_beta = 0;

                        rocblas_operation const trans_a = rocblas_operation_none;
                        rocblas_operation const trans_b = rocblas_operation_none;

                        {
                            I const mm = nrows_A22_re;
                            I const nn = ncols_A22_re;
                            I const kk = ncols_L21_re;

                            auto const istat = rocblasCall_gemm_ex(
                                handle, trans_a, trans_b, mm, nn, kk,

                                &(h_alpha[0]), stride_alpha,

                                L21_re_chop, shift_L21_re_chop, ldL21_chop_re, stride_L21_re,

                                U12_re_chop, shift_U12_re_chop, ldU12_chop_re, stride_U12_re,

                                &one, stride_beta,

                                A22_re, shift_A22_re, ldA22_re, stride_A22_re, batch_count, pfree);

                            if(istat != rocblas_status_success)
                            {
                                return (istat);
                            }
                        }

                        /*
	   alpha = single( (amax_L21_im/dlimit_sq) * amax_U12_im);
	   A22_re( i3:m, j3:n) = A22_re( i3:m, j3:n) + ...
		alpha * single(L21_im_chop) * single(U12_im_chop);
*/
                        for(I bid = 0; bid < batch_count; bid++)
                        {
                            h_alpha[bid] = ((h_amax_L21_im[bid] / dlimit_sq) * h_amax_U12_im[bid]);
                        }

                        {
                            I const mm = nrows_A22_re;
                            I const nn = ncols_A22_re;
                            I const kk = ncols_L21_im;

                            auto const istat = rocblasCall_gemm_ex(
                                handle, trans_a, trans_b, mm, nn, kk,

                                &(h_alpha[0]), stride_alpha,

                                L21_im_chop, shift_L21_im_chop, ldL21_im_chop, stride_L21_im_chop,

                                U12_im_chop, shift_U12_im_chop, ldU12_im_chop, stride_U12_im_chop,

                                &one, stride_beta,

                                A22_re, shift_A22_re, ldA22_re, stride_A22_re,

                                batch_count, pfree);

                            if(istat != rocblas_status_success)
                            {
                                return (istat);
                            }
                        }
                    }
                    else
                    {
                        //  -----------------------------
                        //  A22_re = A22_re - L21_re_chop * U12_re_chop
                        //  -----------------------------

                        {
                            auto const trans_a = rocblas_operation_none;
                            auto const trans_b = rocblas_operation_none;

                            auto const mm = nrows_A22_re;
                            auto const nn = ncols_A22_re;
                            auto const kk = nrows_U12_re;

                            Istride const stride_alpha = 0;
                            Istride const stride_beta = 0;
                            auto const istat = rocblasCall_gemm_ex(
                                handle, trans_a, trans_b, mm, nn, kk, &minone, stride_alpha,

                                L21_re_chop, shift_L21_re_chop, ldL21_re_chop, stride_L21_re_chop,

                                U12_re_chop, shift_U12_re_chop, ldU12_re_chop, stride_U12_re_chop,

                                &one, stride_beta,

                                A22_re, shift_A22_re, ldA22_re, stride_A22_re,

                                batch_count, (void*)pfree);

                            if(istat != rocblas_status_success)
                            {
                                return (istat);
                            }
                        }
                        //  -----------------------------
                        //  A_re = A_re + L21_im * U12_im
                        //  -----------------------------

                        {
                            auto const trans_a = rocblas_operation_none;
                            auto const trans_b = rocblas_operation_none;

                            auto const mm = nrows_A22_re;
                            auto const nn = ncols_A22_re;
                            auto const kk = nrows_U12_im;

                            Istride const stride_alpha = 0;
                            Istride const stride_beta = 0;
                            auto const istat = rocblasCall_gemm_ex(
                                handle, trans_a, trans_b, mm, nn, kk, &one, stride_alpha,

                                L21_im_chop, shift_L21_im_chop, ldL21_im_chop, stride_L21_im_chop,

                                U12_im_chop, shift_U12_im_chop, ldU12_im_chop, stride_U12_im_chop,

                                &one, stride_beta,

                                A22_re, shift_A22_re, ldA22_re, stride_A22_re,

                                batch_count, (void*)pfree);

                            if(istat != rocblas_status_success)
                            {
                                return (istat);
                            }
                        }
                    }

                    if(is_fp16)
                    {
                        std::vector<S> h_alpha(batch_count);
                        rocblas_operation const trans_a = rocblas_operation_none;
                        rocblas_operation const trans_b = rocblas_operation_none;

                        Istride const stride_alpha = 1;
                        Istride const stride_beta = 0;

                        /*
        % -----------------------------
	% A22_im = A22_im - L21_re * U12_im
	% A22_im = A22_im - L21_im * U12_re
        % -----------------------------
*/

                        /*
		alpha = single( (amax_L21_re/dlimit_sq ) * amax_U12_im );
		A22_im( i3:m, j3:n) = A22_im( i3:m, j3:n) - ...
			alpha * single(L21_re_chop) * single(U12_im_chop);
*/
                        {
                            for(I bid = 0; bid < batch_count; bid++)
                            {
                                h_alpha[bid]
                                    = -((h_amax_L21_re[bid] / dlimit_sq) * h_amax_U12_im[bid]);
                            }

                            I const mm = nrows_A22_im;
                            I const nn = ncols_A22_im;
                            I const kk = ncols_L21_re;

                            auto istat = rocblasCall_gemm_ex(
                                handle, trans_a, trans_b, mm, nn, kk,

                                &(h_alpha[0]), stride_alpha,

                                L21_re_chop, shift_L21_re_chop, ldL21_re_chop, stride_L21_re_chop,

                                U12_im_chop, shift_U12_im_chop, ldU12_im_chop, stride_U12_im_chop,

                                &one, stride_beta,

                                A22_im, shift_A22_im, ldA22_im, stride_A22_im,

                                batch_count, (void*)pfree);

                            if(istat != rocblas_status_success)
                            {
                                return (istat);
                            }
                        }

                        /*
		alpha = single( (amax_L21_im/dlimit_sq) * amax_U12_re );
		A22_im( i3:m, j3:n) = A22_im( i3:m, j3:n) - ...
			alpha * single(L21_im_chop) * single(U12_re_chop);
                */

                        {
                            for(I bid = 0; bid < batch_count; bid++)
                            {
                                h_alpha[bid]
                                    = -((h_amax_L21_im[bid] / dlimit_sq) * h_amax_U12_re[bid]);
                            }

                            I const mm = nrows_A22_im;
                            I const nn = ncols_A22_im;
                            I const kk = ncols_L21_im;

                            auto const istat = rocblasCall_gemm(
                                handle, trans_a, trans_b, mm, nn, kk,

                                &(h_alpha[0]), stride_alpha,

                                L21_im_chop, shift_L21_im_chop, ldL21_im_chop, stride_L21_im_chop,

                                U12_re_chop, shift_U12_re_chop, ldU12_re_chop, stride_U12_re_chop,

                                &one, stride_beta,

                                batch_count, (void*)pfree);

                            if(istat != rocblas_status_success)
                            {
                                return (istat);
                            }
                        }
                    }
                    else
                    {
                        //  -----------------------------
                        //  A22_im = A22_im - L21_re * U12_im
                        //  -----------------------------

                        I const mm = nrows_A22_im;
                        I const nn = ncols_A22_im;
                        I const kk = ncols_L21_re;

                        rocblas_operation const trans_a = rocblas_operation_none;
                        rocblas_operation const trans_b = rocblas_operation_none;

                        Istride const stride_alpha = 0;
                        Istride const stride_beta = 0;

                        auto const istat = rocblasCall_gemm_ex(
                            handle, trans_a, trans_b, mm, nn, kk,

                            &minone, stride_alpha,

                            L21_re_chop, shift_L21_re_chop, ldL21_re_chop, stride_L21_re_chop,

                            U12_im_chop, shift_U12_im_chop, ldU12_im_chop, stride_U12_im_chop,

                            &one, stride_beta,

                            A22_im, shift_A22_im, ldA22_im, stride_A22_im,

                            batch_count, (void*)pfree);

                        if(istat != rocblas_status_success)
                        {
                            return (istat);
                        }

                        // -------------------------------
                        //  A22_im = A22_im - L21_im * U12_re
                        // -------------------------------

                        auto const istat = rocblasCall_gemm_ex(
                            handle, trans_a, trans_b, mm, nn, kk, &minone, stride_alpha,

                            L21_im_chop, shift_L21_im_chop, ldL21_im_chop, stride_L21_im_chop,

                            U12_re_chop, shift_U12_re_chop, ldU12_re_chop, stride_U12_re_chop,

                            &one, stride_beta,

                            A22_im, shift_A22_im, ldA22_im, stride_A22_im,

                            batch_count, (void*)pfree);

                        if(istat != rocblas_status_success)
                        {
                            return (istat);
                        }
                    }

                    // -------------------------------
                    // restore back to complex storage
                    // -------------------------------
                    reim2complex_outofplace(
                        handle, nrows_A22_re, ncols_A22_re,

                        A22_re, shift_A22_re, ldA22_re, stride_A22_re,

                        A22_im, shift_A22_im, ldA22_im, stride_A22_im,

                        // A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), lda, strideA,
                        A22, shift_A22, ldA22, stride_A22, batch_count);

                } // end if (is_complex)
            }
        }
    } // end for j

            rocblas_set_pointer_mode(handle, old_mode);
            return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
