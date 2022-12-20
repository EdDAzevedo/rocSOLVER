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
This routine sets the mode used in the rocsolverRfResetValues() routine.
The fast mode may require extra memory and is recommended only if very
fast calls to rocsolverRfResetValues() are needed.

It may be called once prior to rocsolverRfAnalyze() routine.
 -----------------------------------------------------------
*/

rocsolverStatus_t rocsolverRfSetResetValuesFastMode(rocsolverRfHandle_t handle,
                                                    gluResetValuesFastMode_t fast_mode)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    bool const isok = (fast_mode == ROCSOLVERRF_RESET_VALUES_FAST_MODE_OFF)
        || (fast_mode == ROCSOLVERRF_RESET_VALUES_FAST_MODE_ON);
    if(!isok)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    handle->fast_mode = fast_mode;

    return (ROCSOLVER_STATUS_SUCCESS);
};
