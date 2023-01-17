
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
/*
 -------------------------------------------------------------
This routine performs the forward and backward solve with the upper
and lower triangular factors computed from the LU re-factorization
rocsolverRfRefactor() routine.

The routine can solve linear systems with multiple right-hand-sides (RHS):

  solve F = A X = (L U) X = L (U X) or   L Y = F, where Y = U X

This routine may be called multiple times, once for each of the linear
systems:

   A_i x_i = f_i

 -------------------------------------------------------------
*/

#include "rocsolver_RfSolve.hpp"
#include "assert.h"
#include "rf_pqrlusolve.hpp"

extern "C" {

rocsolverStatus_t rocsolverRfSolve(
    // Input (in the device memory)
    rocsolverRfHandle_t handle,
    int* d_P,
    int* d_Q,
    int nrhs,
    double* d_Temp, //  dense matrix of size (ldt * nrhs), ldt >= n
    int ldt,

    // Input/Output (in the device memory)

    // -----------------------------------------
    // dense matrix that contains right-hand side F
    // and solutions X of size (ldxf * nrhs)
    // -----------------------------------------
    double* d_XF,

    // Input
    int ldxf)
{
    return (rocsolverRfBatchSolve(handle, d_P, d_Q, nrhs, d_Temp, ldt, &d_XF, ldxf));
}
};
