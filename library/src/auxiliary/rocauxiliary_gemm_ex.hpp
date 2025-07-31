/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocsolver/rocsolver.h"

#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "lib_host_helpers.hpp"

#include "auxiliary/rocauxiliary_complex2reim_inplace.hpp"

#include <limits>

ROCSOLVER_BEGIN_NAMESPACE

#ifndef CHECK_MEM
#define CHECK_MEM(pfree)                                          \
    {                                                             \
        bool const is_memory_ok = (pfree <= (pwork + size_work)); \
        assert(is_memory_ok);                                     \
        if(!is_memory_ok)                                         \
        {                                                         \
            return (rocblas_status_internal_error);               \
        }                                                         \
    }
#endif

#ifndef ROCBLAS_CHECK
#define ROCBLAS_CHECK(fcn)                  \
    {                                       \
        auto const istat = (fcn);           \
        if(istat != rocblas_status_success) \
        {                                   \
            return (istat);                 \
        }                                   \
    }
#endif

template <typename Tfull, typename Treduced, typename I, typename Istride>
static void rocblasCall_gemm_strided_batched_ex_getMemorySize(rocblas_operation const trans_A,
                                                              rocblas_operation const trans_B,
                                                              I const m,
                                                              I const n,
                                                              I const k,
                                                              I const batch_count,
                                                              size_t* p_size_work)
{
    using Sf = decltype(std::real(Tfull{}));
    using Sr = decltype(std::real(Treduced{}));

    *p_size_work = 0;
    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    // ---------------------------
    // may need array of pointers
    // ---------------------------
    size_t size_work = sizeof(Tfull*) * batch_count;

    bool const is_none_A = (trans_A == rocblas_operation_none);
    bool const is_none_B = (trans_B == rocblas_operation_none);

    I const nrows_A = (is_none_A) ? m : k;
    I const ncols_A = (is_none_A) ? k : m;

    I const nrows_B = (is_none_B) ? k : n;
    I const ncols_B = (is_none_B) ? n : k;

    I const nrows_C = m;
    I const ncols_C = n;

    rocblas_datatype constexpr r_type = rocblas_datatype_from_type<Treduced>;
    rocblas_datatype constexpr f_type = rocblas_datatype_from_type<Tfull>;

    bool constexpr is_fp16_compute
        = (r_type == rocblas_datatype_f16_c) || (r_type == rocblas_datatype_f16_r);

    bool constexpr is_bf16_compute
        = (r_type == rocblas_datatype_bf16_c) || (r_type == rocblas_datatype_bf16_r);

    bool constexpr is_fp32_store
        = (f_type == rocblas_datatype_f32_c) || (f_type == rocblas_datatype_f32_r);

    // -------------------------------------------
    // only process the special case for FP32 storage type and
    // reduced precision  FP16 or BF16 as compute type
    // -------------------------------------------
    bool const need_temp_storage = (is_fp32_store && (is_fp16_compute || is_bf16_compute));
    bool const is_complex = (f_type == rocblas_datatype_f32_c);

    if(need_temp_storage)
    {
        size_t const size_A_re_chop = sizeof(Treduced) * nrows_A * ncols_A * batch_count;
        size_t const size_B_re_chop = sizeof(Treduced) * nrows_B * ncols_B * batch_count;

        size_work += size_A_re_chop;
        size_work += size_B_re_chop;

        size_t size_amax_A_re = 0;
        size_t size_amax_A_im = 0;
        size_t size_amax_B_re = 0;
        size_t size_amax_B_im = 0;

        // --------------------------------
        // storage for amax to fit in FP16
        // --------------------------------
        if(is_fp16_compute)
        {
            size_amax_A_re = sizeof(Sr) * batch_count;
            size_amax_B_re = sizeof(Sr) * batch_count;

            size_work += size_amax_A_re;
            size_work += size_amax_B_re;

            if(is_complex)
            {
                size_amax_A_im = size_amax_A_re;
                size_amax_B_im = size_amax_B_re;

                size_work += size_amax_A_im;
                size_work += size_amax_B_im;
            }
        }

        if(is_complex)
        {
            size_t const size_A_im_chop = size_A_re_chop;
            size_t const size_B_im_chop = size_B_im_chop;

            size_work += size_A_im_chop;
            size_work += size_B_im_chop;
        }

        if(is_complex)
        {
            // ---------------------------------------------------
            // need to split complex C matrix
            // into the real part matrix and imaginary part matrix
            // ---------------------------------------------------
            size_t const size_C_re = sizeof(Sf) * nrows_C * ncols_C * batch_count;
            size_t const size_C_im = size_C_re;

            size_work += size_C_re;
            size_work += size_C_im;
        }
    }

    *p_size_work = size_work;
}

//
// simple interface to emulate rocblas_gemm_ex() or rocblas_gemm_ex_strided_batched()
//
template <typename Tfull, typename Treduced, typename I, typename Istride>
static rocblas_status rocblasCall_gemm_strided_batched_ex(rocblas_handle handle,

                                                          rocblas_operation const trans_A,
                                                          rocblas_operation const trans_B,

                                                          I const m,
                                                          I const n,
                                                          I const k,

                                                          const void* alpha,

                                                          const void* A,
                                                          rocblas_datatype type_A,
                                                          I const ld_A,
                                                          Istride const stride_A,

                                                          const void* B,
                                                          rocblas_datatype type_B,
                                                          I const ld_B,
                                                          Istride const stride_B,

                                                          const void* beta,

                                                          const void* C,
                                                          rocblas_datatype type_C,
                                                          I const ld_C,
                                                          Istride const stride_C,

                                                          const void* D,
                                                          rocblas_datatype type_D,
                                                          I const ld_D,
                                                          Istride const stride_D,

                                                          I const batch_count,

                                                          rocblas_datatype compute_type,
                                                          rocblas_gemm_algo algo,
                                                          int32_t solution_index,
                                                          uint32_t flags,

                                                          void* work,
                                                          size_t size_work)
{
    // ----------------------------------------
    // try evaluation as one of the supported types
    // ----------------------------------------
    auto const istat = (rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k, alpha,

                                                        A, type_A, ld_A, stride_A,

                                                        B, type_B, ld_B, stride_B,

                                                        beta,

                                                        C, type_C, ld_C, stride_C,

                                                        D, type_D, ld_D, stride_D,

                                                        batch_count,

                                                        compute_type, algo, solution_index, flags));
    if(istat != rocblas_status_not_implemented)
    {
        return (istat);
    }

    // ----------------------------------------------------
    // implement computation where storage type is
    // F32_C or F32_R and compute type is BF16 or FP16
    //
    // This is to match cublasGemmEx
    // with A/B/C type be CUDA_R_32F or CUDA_C_32F
    // but compute type is
    // CUBLAS_COMPUTE_F32_FAST_16BF
    // or
    // CUBLAS_COMPUTE_F32_FAST_16F
    // ----------------------------------------------------

    bool const is_complex = rocblas_is_complex<Tfull>;

    bool const is_A_fp32 = (type_A == rocblas_datatype_f32_r) || (type_A == rocblas_datatype_f32_c);
    bool const is_B_fp32 = (type_B == rocblas_datatype_f32_r) || (type_B == rocblas_datatype_f32_c);
    bool const is_C_fp32 = (type_C == rocblas_datatype_f32_r) || (type_C == rocblas_datatype_f32_c);

    bool const is_fp32_store = is_A_fp32 && is_B_fp32 && is_C_fp32 && (type_D == type_C);

    bool const is_fp16_compute
        = (compute_type == rocblas_datatype_f16_r) || (compute_type == rocblas_datatype_f16_c);

    bool const is_bf16_compute
        = (compute_type == rocblas_datatype_bf16_r) || (compute_type == rocblas_datatype_bf16_c);

    bool const is_supported = is_fp32_store && (is_fp16_compute || is_bf16_compute);
    if(!is_supported)
    {
        return (rocblas_status_not_implemented);
    }

    // =============================== //

    bool const is_none_A = (trans_A == rocblas_operation_none);
    bool const is_none_B = (trans_B == rocblas_operation_none);

    I const nrows_A = (is_none_A) ? m : k;
    I const ncols_A = (is_none_A) ? k : m;

    I const nrows_B = (is_none_B) ? k : n;
    I const ncols_B = (is_none_B) ? n : k;

    I const nrows_C = m;
    I const ncols_C = n;

    using Sf = decltype(std::real(Tfull{}));
    using Sr = decltype(std::real(Treduced{}));

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    double const fp32_max = std::numeric_limits<float>::max();
    double const fp16_max = 65504;
    Sr const dlimit = (is_fp16_compute) ? fp16_max : fp32_max;

    Sf* C_re = nullptr;
    Sf* C_im = nullptr;

    Treduced* A_re_chop = nullptr;
    Treduced* A_im_chop = nullptr;
    Treduced* B_re_chop = nullptr;
    Treduced* B_im_chop = nullptr;

    I const ldA_re_chop = nrows_A;
    I const ldA_im_chop = ldA_re_chop;

    Istride stride_A_re_chop = ldA_re_chop * ncols_A;
    Istride stride_A_im_chop = stride_A_re_chop;

    I const ldB_re_chop = nrows_B;
    I const ldB_im_chop = ldB_re_chop;

    Istride stride_B_re_chop = ldB_re_chop * ncols_B;
    Istride stride_B_im_chop = stride_B_re_chop;

    size_t const size_A_re_chop = sizeof(Sr) * stride_A_re_chop * batch_count;
    size_t const size_B_re_chop = sizeof(Sr) * stride_B_re_chop * batch_count;

    size_t const size_A_im_chop = (is_complex) ? size_A_re_chop : 0;
    size_t const size_B_im_chop = (is_complex) ? size_B_re_chop : 0;

    I const ldC = ld_C;
    I const ldC_re = nrows_C;
    I const ldC_im = ldC_re;

    Istride const shift_C_re = 0;
    Istride const shift_C_im = 0;

    Istride const stride_C_re = ldC_re * ncols_C;
    Istride const stride_C_im = stride_C_re;

    size_t const size_C_re = sizeof(Sf) * stride_C_re * batch_count;
    size_t const size_C_im = size_C_re;

    rocblas_datatype const A_re_type = rocblas_datatype_from_type<Sr>;
    rocblas_datatype const A_im_type = A_re_type;

    rocblas_datatype const B_re_type = rocblas_datatype_from_type<Sr>;
    rocblas_datatype const B_im_type = B_re_type;

    rocblas_datatype const C_re_type = rocblas_datatype_from_type<Sf>;
    rocblas_datatype const C_im_type = C_re_type;

    // ----------------------------------------------------
    // assume everything is executed with scalars on the host
    // ----------------------------------------------------
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    auto const is_valid_mode = (old_mode == rocblas_pointer_mode_host);
    if(!is_valid_mode)
    {
        return (rocblas_status_internal_error);
    }

    // --------------------------------------------
    // split into the real part and imaginary part
    // in reduced precision
    // --------------------------------------------
    A_re_chop = (Sr*)pfree;
    pfree += size_A_re_chop;
    B_re_chop = (Sr*)pfree;
    pfree += size_B_re_chop;
    if(is_complex)
    {
        A_im_chop = (Sr*)pfree;
        pfree += size_A_im_chop;
        B_im_chop = (Sr*)pfree;
        pfree += size_B_im_chop;
    }

    CHECK_MEM(pfree);

    Istride const shift_A = 0;
    Istride const shift_B = 0;
    Istride const shift_C = 0;

    Istride const shift_A_re_chop = 0;
    Istride const shift_A_im_chop = 0;

    Istride const shift_B_re_chop = 0;
    Istride const shift_B_im_chop = 0;

    complex2reim_clamp(handle, nrows_A, ncols_A,

                       A, shift_A, ld_A, stride_A,

                       A_re_chop, shift_A_re_chop, ldA_re_chop, stride_A_re_chop,

                       A_im_chop, shift_A_im_chop, ldA_im_chop, stride_A_im_chop,

                       batch_count,

                       dlimit);

    complex2reim_clamp(handle, nrows_B, ncols_B,

                       B, shift_B, ld_B, stride_B,

                       B_re_chop, shift_B_re_chop, ldB_re_chop, stride_B_re_chop,

                       B_im_chop, shift_B_im_chop, ldB_im_chop, stride_B_im_chop,

                       batch_count,

                       dlimit);

    // ----------------------------------------------
    // split the "C" into the real part and imag part
    // ----------------------------------------------
    if(is_complex)
    {
        Sf const dlimit_C = fp32_max;
        Sf* const amax_C_re_null = nullptr;
        Sf* const amax_C_im_null = nullptr;

        C_re = (Sf*)pfree;
        pfree += size_C_re;
        C_im = (Sf*)pfree;
        pfree += size_C_im;

        CHECK_MEM(pfree);

        complex2reim_clamp(handle, nrows_C, ncols_C,

                           C, shift_C, ldC, stride_C,

                           C_re, shift_C_re, ldC_re, stride_C_re,

                           C_im, shift_C_im, ldC_im, stride_C_im,

                           batch_count,

                           dlimit_C);
    }
    else
    {
        C_re = C;
        ldC_re = ld_C;
        stride_C_re = stride_C;
    }

    // ------------------------------------
    // (Cr + J * Cc) = beta * (Cr + J * Cc) +
    //              alpha * ( Ar + J * Ac ) * (Br + J * Bc)
    //
    // Cr = beta * Cr + alpha * { Ar * Br - Ac * Bc }
    // Cc = beta * Cc + alpha * { Ac * Br + Ar * Bc }
    // ------------------------------------

    // ------------------------------------
    // (1) Cr = beta * Cr + alpha * Ar * Br
    // (2) Cr =        Cr - alpha * Ac * Bc
    //
    // (3) Cc = beta * Cc + alpha * Ac * Br
    // (4) Cc =        Cc + alpha * Ar * Bc
    // ------------------------------------

    // -----------------------------------------
    // step (1) Cr = beta * Cr + alpha * Ar * Br
    // -----------------------------------------

    rocblas_datatype lcompute_type = rocblas_datatype_f32_r;

    {
        // ---------------------------------------------
        // no need to adjust alpha or beta
        // ---------------------------------------------

        ROCBLAS_CHECK(
            rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                            alpha,

                                            A_re_chop, A_re_type, ldA_re_chop, stride_A_re_chop,

                                            B_re_chop, B_re_type, ldB_re_chop, stride_B_re_chop,

                                            beta,

                                            C_re, C_re_type, ldC_re, stride_C_re,

                                            C_re, C_re_type, ldC_re, stride_C_re, // D matrix

                                            batch_count,

                                            lcompute_type, algo, solution_index, flags));
    }

    if(is_complex)
    {
        // ------------------------------------
        // step (2) Cr =        Cr - alpha * Ac * Bc
        // ------------------------------------

        {
            Treduced* p_alpha = (Treduced*)p_alpha;
            Treduced alpha_value = *p_alpha;
            Treduced neg_alpha_value = -alpha_value;

            Treduced beta_one = 1;

            ROCBLAS_CHECK(
                rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                                &neg_alpha_value,

                                                A_im_chop, A_im_type, ldA_im_chop, stride_A_im_chop,

                                                B_im_chop, B_im_type, ldB_im_chop, stride_B_im_chop,

                                                &beta_one,

                                                C_re, C_re_type, ldC_re, stride_C_re,

                                                C_re, C_re_type, ldC_re, stride_C_re, // D matrix

                                                batch_count,

                                                lcompute_type, algo, solution_index, flags));
        }

        // ------------------------------------
        // step (3) Cc = beta * Cc + alpha * Ac * Br
        // ------------------------------------

        {
            ROCBLAS_CHECK(
                rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                                alpha,

                                                A_im_chop, A_im_type, ldA_im_chop, stride_A_im_chop,

                                                B_re_chop, B_re_type, ldB_re_chop, stride_B_re_chop,

                                                beta,

                                                C_im, C_im_type, ldC_im, stride_C_im,

                                                C_im, C_im_type, ldC_im, stride_C_im, // D matrix

                                                batch_count,

                                                lcompute_type, algo, solution_index, flags));
        }

        // ------------------------------------
        // step (4) Cc =        Cc + alpha * Ar * Bc
        // ------------------------------------

        {
            Treduced beta_one = 1;

            ROCBLAS_CHECK(
                rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                                alpha,

                                                A_re_chop, A_re_type, ldA_re_chop, stride_A_re_chop,

                                                B_im_chop, B_im_type, ldB_im_chop, stride_B_im_chop,

                                                &beta_one,

                                                C_im, C_im_type, ldC_im, stride_C_im,

                                                C_im, C_im_type, ldC_im, stride_C_im, // D matrix

                                                batch_count,

                                                lcompute_type, algo, solution_index, flags));
        }

        // -------------------------------------------------------
        // convert from real and imag parts back to complex matrix
        // -------------------------------------------------------

        reim2complex_simple(handle, nrows_C, ncols_C,

                            C_re, shift_C_re, ldC_re, stride_C_re,

                            C_im, shift_C_im, ldC_im, stride_C_im,

                            C, shift_C, ldC, stride_C);
    }

    return (rocblas_status_success);
}

ROCSOLVER_END_NAMESPACE
