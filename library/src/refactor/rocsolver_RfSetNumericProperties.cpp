
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
#include "rocsolver_refactor.h"

/*
-----------------------------------------------------------------------
This routine sets the numeric values used for checking for "zero" pivot
and for boosting it in the rocsolverRfRefactor() and rocsolverRfSolve()
routines. It may be called multiple times prior to rocsolverRfRefactor()
and rocsolverRfSolve() routines.  The numeric boosting will be used only
if boost > 0.0
-----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfSetNumericProperties(rocsolverRfHandle_t handle,
                                                  double effective_zero,
                                                  double boost_val)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    handle->effective_zero = effective_zero;
    handle->boost_val = boost_val;

    return (ROCSOLVER_STATUS_SUCCESS);
};
};
