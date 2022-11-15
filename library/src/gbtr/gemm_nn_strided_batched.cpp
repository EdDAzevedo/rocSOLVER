
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
#include "gemm_nn_strided_batched.hpp"
#include "gbtr_common.h"

template <typename T>
void gemm_nn_strided_batched_template(hipStream_t stream,

                                      int const m,
                                      int const n,
                                      int const k,
                                      int const batchCount,

                                      T const alpha,

                                      T const* const A_,
                                      int const ldA,
                                      long const strideA,

                                      T const* const B_,
                                      int const ldB,
                                      long const strideB,

                                      T const beta,

                                      T* C_,
                                      int const ldC,
                                      long const strideC)
{
#ifdef USE_GPU
    hipLaunchKernelGGL((gemm_nn_strided_batched_kernel), dim3(grid_dim), dim3(block_dim), 0, stream,

                       m, n, k, batchCount, alpha, A_, ldA, strideA, B_, ldB, strideB, beta, C_,
                       ldC, strideC);

#else
    gemm_nn_strided_batched_kernel(m, n, k, batchCount, alpha, A_, ldA, strideA, B_, ldB, strideB,
                                   beta, C_, ldC, strideC);
#endif
};

extern "C" {

void Dgemm_nn_strided_batched(hipStream_t stream,

                              int const m,
                              int const n,
                              int const k,
                              int const batchCount,

                              double const alpha,

                              double const* const A_,
                              int const ldA,
                              long const strideA,

                              double const* const B_,
                              int const ldB,
                              long const strideB,

                              double const beta,

                              double* C_,
                              int const ldC,
                              long const strideC)
{
    gemm_nn_strided_batched_template<double>(stream, m, n, k, batchCount, alpha, A_, ldA, strideA,
                                             B_, ldB, strideB, beta, C_, ldC, strideC);
};

void Zgemm_nn_strided_batched(hipStream_t stream,

                              int const m,
                              int const n,
                              int const k,
                              int const batchCount,

                              rocblas_double_complex const alpha,

                              rocblas_double_complex const* const A_,
                              int const ldA,
                              long const strideA,

                              rocblas_double_complex const* const B_,
                              int const ldB,
                              long const strideB,

                              rocblas_double_complex const beta,

                              rocblas_double_complex* C_,
                              int const ldC,
                              long const strideC)
{
    gemm_nn_strided_batched_template<rocblas_double_complex>(stream, m, n, k, batchCount, alpha, A_,
                                                             ldA, strideA, B_, ldB, strideB, beta,
                                                             C_, ldC, strideC);
};
}
