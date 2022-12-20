
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
#include "rocsolver_RfSetupDevice.hpp"

/*
----------------------------------------------------------------------
This routine assembles the internal data structures of the rocSolverRF
library.  It is often the first routine to be called after the call to
the rocsolverRfCreate() routine.

This routine accepts as input (on the device) the original matrix A,
the lower L and upper U triangular factors, as well as the left (P)
and right (Q) permutations resulting from the full LU factorization of
the first (i=1) linear system

   A_i x_i = f_i

The permutations P and Q represent the final composition of all the left
and right reordering applied to the original matrix A, respectively.
However, these permutations are often associated with partial pivoting
and reordering to minimize fill-in, respectively.

This routine needs to be called only for a single linear system

  
   A_i x_i = f_i


----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfSetupDevice(int n, // host input
                                         int nnzA, // host input

                                         /* Input (in the device memory) */
                                         int* csrRowPtrA_in,
                                         int* csrColIndA_in,
                                         double* csrValA_in,

                                         int nnzL, // host input

                                         /* Input (in the device memory) */
                                         int* csrRowPtrL_in,
                                         int* csrColIndL_in,
                                         double* csrValL_in,

                                         int nnzU, // host input

                                         /* Input (in the device memory) */
                                         int* csrRowPtrU_in,
                                         int* csrColIndU_in,
                                         double* csrValU_in,
                                         int* P_in,
                                         int* Q_in,

                                         /* Output */
                                         rocsolverRfHandle_t handle)
{
    bool constexpr MAKE_COPY = true;
    return (rocsolverRfSetupDevice_impl<MAKE_COPY>(
        n, nnzA, csrRowPtrA_in, csrColIndA_in, csrValA_in, nnzL, csrRowPtrL_in, csrColIndL_in,
        csrValL_in, nnzU, csrRowPtrU_in, csrColIndU_in, csrValU_in, P_in, Q_in, handle));
};
};
