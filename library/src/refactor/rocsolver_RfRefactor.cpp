
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

#include "rf_common.hpp"

/*
---------------------------------------------
This routine performs the LU re-factorization
    A = L * U
uses the available parallelism on the GPU.  It is assumed that a prior
call to rocsolverRfAnalyze() was done in order to find the available
parallelism.

This routine may be called multiple times, once for each of the linear
systems:
   A_i  x_i = f_i

There are some constraints to the combination of algorithms used
for refactorization and solving routines, rocsolverRfRefactor() and
rocsolverRfSolve().  The wrong combination generates the error code
ROCSOLVER_STATUS_INVALID_VALUE.
---------------------------------------------
 */

extern "C" {

rocsolverStatus_t rocsolverRfRefactor(rocsolverRfHandle_t handle)
{
    return (rocsolverRfBatchRefactor(handle));
};
};
