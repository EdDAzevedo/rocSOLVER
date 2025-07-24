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

static const unsigned int AMAX_THREADS = 64;

template <typename Telem, typename I, typename Tresult>
typename std::enable_if<!rocblas_is_complex<Telem>, void>::type __device__ __host__
    set_max_magnitude(Telem e, const I tid, Tresult* lds_re, Tresult* lds_im)
{
    lds_re[tid] = std::max<Tresult>(lds_re[tid], std::abs(e));
}

template <typename Telem, typename I, typename Tresult>
typename std::enable_if<rocblas_is_complex<Telem>, void>::type __device__ __host__
    set_max_magnitude(Telem e, const I tid, Tresult* lds_re, Tresult* lds_im)
{
    lds_re[tid] = std::max<Tresult>(lds_re[tid], std::abs(e.real()));
    lds_im[tid] = std::max<Tresult>(lds_im[tid], std::abs(e.imag()));
}

// Return maximum magnitude of any element in an N by M matrix.
// Length M goes into gridDim.x, and batch size goes into gridDim.z
// at launch time.
//
// Telem is the type of an element in the matrix - either a real or
// complex.
//
// I, Istride are integer types for indexing.
//
// Tresult is the type of the result - for real matrices Tresult is
// expected to be the same as Telem.  For complex matrices, Tresult
// is expected to be the type of the real/imag part of the complex
// type.
template <typename Telem, typename I, typename Istride, typename Tresult>
void __global__ __launch_bounds__(AMAX_THREADS) amax_matrix_kernel(const I m,
                                                                   const Telem* A,
                                                                   const Istride shiftA,
                                                                   const I lda,
                                                                   const Istride strideA,
                                                                   Tresult* result_re,
                                                                   Tresult* result_im)
{
    const auto n_idx = blockIdx.x;
    const auto batch_id = blockIdx.z;
    const auto tid = threadIdx.x;

    const auto in = load_ptr_batch(A, batch_id, shiftA, strideA);

    extern __shared__ Tresult lds[];
    Tresult* lds_re = lds;
    Tresult* lds_im = lds + AMAX_THREADS;

    lds_re[tid] = static_cast<Tresult>(0.0);
    if constexpr(rocblas_is_complex<Telem>)
    {
        lds_im[tid] = static_cast<Tresult>(0.0);
    }

    // Each thread reads two elements and reduces the max abs value
    // to its location in LDS
    for(unsigned int iter = 0; iter < ceil(m / 2, AMAX_THREADS); ++iter)
    {
        auto m_idx = idx2D(tid, iter, AMAX_THREADS);
        if(m_idx < m)
        {
            const auto read_idx = idx2D(m_idx, n_idx, lda);
            auto elem = in[read_idx];
            set_max_magnitude(elem, tid, lds_re, lds_im);
        }

        // second half of read
        auto m_idx2 = m_idx + m / 2;
        if(m_idx2 < m)
        {
            const auto read_idx = idx2D(m_idx2, n_idx, lda);
            auto elem = in[read_idx];
            set_max_magnitude(elem, tid, lds_re, lds_im);
        }
    }

    __syncthreads();

    // reduce in LDS
    for(unsigned int stride = AMAX_THREADS / 2; stride > 0; stride /= 2)
    {
        if(tid < stride && tid + stride < m / 2)
        {
            lds_re[tid] = std::max(lds_re[tid], lds_re[tid + stride]);
            if constexpr(rocblas_is_complex<Telem>)
            {
                lds_im[tid] = std::max(lds_im[tid], lds_im[tid + stride]);
            }
        }
        __syncthreads();
    }
    __syncthreads();

    if(tid == 0)
    {
        atomicMax(result_re + batch_id, lds_re[0]);
        if constexpr(rocblas_is_complex<Telem>)
        {
            atomicMax(result_im + batch_id, lds_im[0]);
        }
    }
}

// Return maximum magnitude of any element (or real component) in an
// m by n matrix.  Multiple matrices can be batched together, and the
// resulting real/imaginary maximum of each matrix is written to an
// arrays of length batch_count on the device.
//
// Telem is the type of an element in the matrix - either a real or
// complex.
//
// I, Istride are integer types for indexing.
//
// Tresult is the type of the result - either Telem if Telem is real,
// or the real value type if Telem is complex.  result_re holds the
// maximum of the real values, and result_im holds the maximum
// imaginary value.  result_im is ignored if Telem is not complex.
//
// Result arrays are expected to be initialized to zero
template <typename Telem, typename I, typename Istride, typename Tresult>
void amax_matrix(rocblas_handle handle,
                 const I m,
                 const I n,
                 const Telem* A,
                 const Istride shiftA,
                 const I lda,
                 const Istride strideA,
                 const I batch_count,
                 Tresult* result_re,
                 Tresult* result_im)
{
    hipStream_t stream = nullptr;
    rocblas_get_stream(handle, &stream);

    const dim3 blockDim{std::min<unsigned int>(AMAX_THREADS, m / 2), 1, 1};
    const dim3 gridDim{static_cast<unsigned int>(n), 1, static_cast<unsigned int>(batch_count)};

    const unsigned int lds_bytes_real = AMAX_THREADS * sizeof(Tresult);
    const unsigned int reals_per_elem = rocblas_is_complex<Telem> ? 1 : 2;

    amax_matrix_kernel<<<gridDim, blockDim, lds_bytes_real * reals_per_elem, stream>>>(
        m, A, shiftA, lda, strideA, result_re, result_im);
}

ROCSOLVER_END_NAMESPACE
