
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
#include "rf_pqrlusolve.h"

extern "C" {

rocsolverStatus_t rocsolverRfSolve(
    /* Input (in the device memory) */
    rocsolverRfHandle_t handle,
    int* P,
    int* Q,
    int nrhs,
    double* Temp, /* dense matrix of size (ldt * nrhs), ldt >= n */
    int ldt,

    /* Input/Output (in the device memory) */

    // -----------------------------------------
    // dense matrix that contains right-hand side F
    // and solutions X of size (ldxf * nrhs)
    // -----------------------------------------
    double* XF,

    /* Input */
    int ldxf)
{
  
    {
      bool const isok = (handle != nullptr) && (handle->hipsparse_handle != nullptr);
      if( !isok)
      {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
      };
    };

    {
        bool const isok = (P != nullptr) && (Q != nullptr) && (Temp != nullptr) && (XF != nullptr);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    int const n = handle->n;
    if((n < 0) || (nrhs < 0))
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    if((n == 0) || (nrhs == 0))
    {
        // no work
        return (ROCSOLVER_STATUS_SUCCESS);
    };

    {
        bool const isok_arguments = (XF != nullptr) && (ldxf >= n);
        if(!isok_arguments)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    int* const P_new2old = P;
    int* const Q_new2old = Q;

    {
    bool const isok_assumption = (P_new2old == handle->P_new2old) &&
                                 (Q_new2old == handle->P_new2old);
    if (!isok_assumption) {
       return( ROCSOLVER_STATUS_INVALID_VALUE );
       };
    };





    int nerrors = 0;
    for(int irhs = 0; irhs < nrhs; irhs++)
    {
        double* const Rs = nullptr;
        double* const brhs = &(XF[ldxf * irhs]);

        int const ibatch = 0;
        int* const LUp = handle->csrRowPtrLU;
        int* const LUi = handle->csrColIndLU;
        double* const LUx = handle->csrValLU_array[ibatch];

        int isok = rf_pqrlusolve(handle->hipsparse_handle, n, P_new2old, Q_new2old, Rs,
                                 LUp, LUi, LUx, brhs );
        if(!isok)
        {
            nerrors++;
        };
    };

    return((nerrors == 0) ?  ROCSOLVER_STATUS_SUCCESS :
                             ROCSOLVER_STATUS_INTERNAL_ERROR);
  
};
};
