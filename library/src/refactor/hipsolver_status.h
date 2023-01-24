
/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#pragma once
#ifndef HIPSOLVER_STATUS_H
#define HIPSOLVER_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    HIPSOLVER_STATUS_SUCCESS = 0,
    HIPSOLVER_STATUS_NOT_INITIALIZED = 1,
    HIPSOLVER_STATUS_ALLOC_FAILED = 2,
    HIPSOLVER_STATUS_INVALID_VALUE = 3,
    HIPSOLVER_STATUS_ARCH_MISMATCH = 4,
    HIPSOLVER_STATUS_MAPPING_ERROR = 5,
    HIPSOLVER_STATUS_EXECUTION_FAILED = 6,
    HIPSOLVER_STATUS_INTERNAL_ERROR = 7,
    HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    HIPSOLVER_STATUS_NOT_SUPPORTED = 9,
    HIPSOLVER_STATUS_ZERO_PIVOT = 10,
    HIPSOLVER_STATUS_INVALID_LICENSE = 11,
    HIPSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12,
    HIPSOLVER_STATUS_IRS_PARAMS_INVALID = 13,
    HIPSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14,
    HIPSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15,
    HIPSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16,
    HIPSOLVER_STATUS_IRS_INTERNAL_ERROR = 20,
    HIPSOLVER_STATUS_IRS_NOT_SUPPORTED = 21,
    HIPSOLVER_STATUS_IRS_OUT_OF_RANGE = 22,
    HIPSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23,
    HIPSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25,
    HIPSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26,
    HIPSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30,
    HIPSOLVER_STATUS_INVALID_WORKSPACE = 31
} hipsolverStatus_t;

#ifdef __cplusplus
};
#endif

#endif
