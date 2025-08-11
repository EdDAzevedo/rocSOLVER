
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

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "lib_host_helpers.hpp"

ROCSOLVER_BEGIN_NAMESPACE

static const unsigned int PLANAR_THREADS = 64;

// Convert complex interleaved matrix into complex planar, with
// separate buffers for each of the real and imaginary planes.
//
// Tcomplex is the input matrix type.  Treal is the output type of
// both the real and imaginary planes.
//
// I, Istride are integer types for indexing.
//
// Tscale is the type used for scaling the values during conversion.
// dlimit is a single value, while amax_re and amax_im are arrays in
// device memory of length batch_count.  Real and imaginary values
// are scaled during conversion as:
//
//   elem_out = dlimit / amax[batch_id] * elem_in
//
// This arithmetic is performed in the precision of Tscale.  Scaling
// is only performed for real or imaginary values if the
// corresponding amax array is non-null.
template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
void __global__ __launch_bounds__(PLANAR_THREADS)
    complex2reim_outofplace_kernel(const I m,
                                   const Tcomplex* A,
                                   const Istride shiftA,
                                   const I lda,
                                   const Istride strideA,
                                   Treal* A_re,
                                   const Istride shiftA_re,
                                   const I ld_re,
                                   const Istride strideA_re,
                                   Treal* A_im,
                                   const Istride shiftA_im,
                                   const I ld_im,
                                   const Istride strideA_im,
                                   const Tscale dlimit,
                                   const Tscale* amax_re,
                                   const Tscale* amax_im)
{
    auto n_idx = blockIdx.x;
    auto batch_id = blockIdx.z;

    const auto in = load_ptr_batch(A, batch_id, shiftA, strideA);
    auto out_re = load_ptr_batch(A_re, batch_id, shiftA_re, strideA_re);
    auto out_im = load_ptr_batch(A_im, batch_id, shiftA_im, strideA_im);

    for(unsigned int iter = 0; iter < ceil(m, PLANAR_THREADS); ++iter)
    {
        auto m_idx = idx2D(threadIdx.x, iter, PLANAR_THREADS);
        if(m_idx < m)
        {
            const auto read_idx = idx2D(m_idx, n_idx, lda);

            const auto write_re_idx = idx2D(m_idx, n_idx, ld_re);
            const auto write_im_idx = idx2D(m_idx, n_idx, ld_im);
            auto elem = in[read_idx];

            if(amax_re)
                out_re[write_re_idx] = static_cast<Tscale>(elem.x) * dlimit / amax_re[batch_id];
            else
                out_re[write_re_idx] = elem.x;

            if(amax_im)
                out_im[write_im_idx] = static_cast<Tscale>(elem.y) * dlimit / amax_im[batch_id];
            else
                out_im[write_im_idx] = elem.y;
        }
    }
}

// Convert complex interleaved matrix into complex planar, with
// separate buffers for each of the real and imaginary planes.
//
// Tcomplex is the input matrix type.  Treal is the output type of
// both the real and imaginary planes.
//
// I, Istride are integer types for indexing.
//
// Tscale is the type used for scaling the values during conversion.
// dlimit is a single value, while amax_re and amax_im are optional
// arrays in device memory of length batch_count.  Real and imaginary
// values are scaled during conversion as:
//
//   elem_out = dlimit / amax[batch_id] * elem_in
//
// This arithmetic is performed in the precision of Tscale.  Scaling
// is only performed for real or imaginary values if the
// corresponding amax array is non-null.
template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
void complex2reim_outofplace(rocblas_handle handle,
                             const I m,
                             const I n,
                             const Tcomplex* A,
                             const Istride shiftA,
                             const I lda,
                             const Istride strideA,
                             Treal* A_re,
                             const Istride shiftA_re,
                             const I ld_re,
                             const Istride strideA_re,
                             Treal* A_im,
                             const Istride shiftA_im,
                             const I ld_im,
                             const Istride strideA_im,
                             const I batch_count,
                             const Tscale dlimit,
                             const Tscale* amax_re,
                             const Tscale* amax_im)
{
    hipStream_t stream = nullptr;
    rocblas_get_stream(handle, &stream);

    dim3 blockDim{PLANAR_THREADS, 1, 1};
    dim3 gridDim{static_cast<unsigned int>(n), 1, static_cast<unsigned int>(batch_count)};

    complex2reim_outofplace_kernel<<<gridDim, blockDim, 0, stream>>>(
        m, A, shiftA, lda, strideA, A_re, shiftA_re, ld_re, strideA_re, A_im, shiftA_im, ld_im,
        strideA_im, dlimit, amax_re, amax_im);
}

// Convert complex planar matrix (real/imaginary in separate buffers
// for each plane) into complex interleaved.
//
// Treal is the input type of both the real and imaginary planes.
// Tcomplex is the output matrix type.
//
// I, Istride are integer types for indexing.
//
// Tscale is the type used for scaling the values during conversion.
// dlimit is a single value, while amax_re and amax_im are arrays in
// device memory of length batch_count.  Real and imaginary values
// are scaled during conversion as:
//
//   elem_out = dlimit / amax[batch_id] * elem_in
//
// This arithmetic is performed in the precision of Tscale.  Scaling
// is only performed for real or imaginary values if the
// corresponding amax array is non-null.
template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
void __global__ __launch_bounds__(PLANAR_THREADS)
    reim2complex_outofplace_kernel(const I m,
                                   const Treal* A_re,
                                   const Istride shiftA_re,
                                   const I ld_re,
                                   const Istride strideA_re,
                                   const Treal* A_im,
                                   const Istride shiftA_im,
                                   const I ld_im,
                                   const Istride strideA_im,
                                   Tcomplex* A,
                                   const Istride shiftA,
                                   const I lda,
                                   const Istride strideA,
                                   const Tscale dlimit,
                                   const Tscale* amax_re,
                                   const Tscale* amax_im)
{
    auto n_idx = blockIdx.x;
    auto batch_id = blockIdx.z;

    const auto in_re = load_ptr_batch(A_re, batch_id, shiftA_re, strideA_re);
    const auto in_im = load_ptr_batch(A_im, batch_id, shiftA_im, strideA_im);
    auto out = load_ptr_batch(A, batch_id, shiftA, strideA);

    for(unsigned int iter = 0; iter < ceil(m, PLANAR_THREADS); ++iter)
    {
        auto m_idx = idx2D(threadIdx.x, iter, PLANAR_THREADS);
        if(m_idx < m)
        {
            const auto write_idx = idx2D(m_idx, n_idx, lda);

            const auto read_re_idx = idx2D(m_idx, n_idx, ld_re);
            const auto read_im_idx = idx2D(m_idx, n_idx, ld_im);
            auto elem_re = in_re[read_re_idx];
            auto elem_im = in_im[read_im_idx];

            Tcomplex elem_out{};
            if(amax_re)
                elem_out.real(static_cast<Tscale>(elem_re) * dlimit / amax_re[batch_id]);
            else
                elem_out.real(elem_re);

            if(amax_im)
                elem_out.imag(static_cast<Tscale>(elem_im) * dlimit / amax_im[batch_id]);
            else
                elem_out.imag(elem_im);

            out[write_idx] = elem_out;
        }
    }
}

// Convert complex planar matrix (real/imaginary in separate buffers
// for each plane) into complex interleaved.
//
// Treal is the input type of both the real and imaginary planes.
// Tcomplex is the output matrix type.
//
// I, Istride are integer types for indexing.
//
// Tscale is the type used for scaling the values during conversion.
// dlimit is a single value, while amax_re and amax_im are arrays in
// device memory of length batch_count.  Real and imaginary values
// are scaled during conversion as:
//
//   elem_out = dlimit / amax[batch_id] * elem_in
//
// This arithmetic is performed in the precision of Tscale.  Scaling
// is only performed for real or imaginary values if the
// corresponding amax array is non-null.
template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
void reim2complex_outofplace(rocblas_handle handle,
                             const I m,
                             const I n,
                             const Treal* A_re,
                             const Istride shiftA_re,
                             const I ld_re,
                             const Istride strideA_re,
                             const Treal* A_im,
                             const Istride shiftA_im,
                             const I ld_im,
                             const Istride strideA_im,
                             Tcomplex* A,
                             const Istride shiftA,
                             const I lda,
                             const Istride strideA,
                             const I batch_count,
                             const Tscale dlimit,
                             const Tscale* amax_re,
                             const Tscale* amax_im)
{
    hipStream_t stream = nullptr;
    rocblas_get_stream(handle, &stream);

    dim3 blockDim{PLANAR_THREADS, 1, 1};
    dim3 gridDim{static_cast<unsigned int>(n), 1, static_cast<unsigned int>(batch_count)};

    reim2complex_outofplace_kernel<<<gridDim, blockDim, 0, stream>>>(
        m, A_re, shiftA_re, ld_re, strideA_re, A_im, shiftA_im, ld_im, strideA_im, A, shiftA, lda,
        strideA, dlimit, amax_re, amax_im);
}

ROCSOLVER_END_NAMESPACE
