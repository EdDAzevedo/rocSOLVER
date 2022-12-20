
/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hip_check.h"
#include "hipsparse_check.h"

#include "rocsolver_refactor.h"

/*
 -----------------------------------------------------------
 This routine sets the matrix format used in rocsolverRfSetupDevice(),
 rocsolverRfSetupHost(), rocsolverRfResetValues(),
 rocsolverRfExtractBundledFactorsHost(),
 rocsolverRfExtractSplitFactorsHost() routines.

 It may be called once prior to rocsolverRfSetupDevice() and
 rocsolverRfSetupHost() routines.
 -----------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfSetMatrixFormat(rocsolverRfHandle_t handle,
                                             rocsolverRfMatrixFormat_t matrix_format,
                                             rocsolverRfUnitDiagonal_t diag_format)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    {
        bool const isok = (diag_format == ROCSOLVERRF_UNIT_DIAGONAL_STORED_L)
            || (diag_format == ROCSOLVERRF_UNIT_DIAGONAL_STORED_U)
            || (diag_format == ROCSOLVERRF_UNIT_DIAGONAL_ASSUMED_L)
            || (diag_format == ROCSOLVERRF_UNIT_DIAGONAL_ASSUMED_U);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    {
        bool const isok = (matrix_format == ROCSOLVERRF_MATRIX_FORMAT_CSR)
            || (matrix_format == ROCSOLVERRF_MATRIX_FORMAT_CSC);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    handle->matrix_format = matrix_format;
    handle->diag_format = diag_format;

    return (ROCSOLVER_STATUS_SUCCESS);
};
};
