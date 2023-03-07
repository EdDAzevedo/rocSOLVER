
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef HIPSPARSE_CHECK_H
#define HIPSPARSE_CHECK_H

#include <exception>
#include <stdio.h>
#include <stdlib.h>

#include "hipsparse/hipsparse.h"

#ifndef hipsparseGetErrorName
#define hipsparseGetErrorName(istat)                                                            \
    (((istat) == HIPSPARSE_STATUS_SUCCESS)                ? "HIPSPARSE_STATUS_SUCCESS"          \
         : ((istat) == HIPSPARSE_STATUS_MAPPING_ERROR)    ? "HIPSPARSE_STATUS_MAPPING_ERROR"    \
         : ((istat) == HIPSPARSE_STATUS_ZERO_PIVOT)       ? "HIPSPARSE_STATUS_ZERO_PIVOT"       \
         : ((istat) == HIPSPARSE_STATUS_NOT_SUPPORTED)    ? "HIPSPARSE_STATUS_NOT_SUPPORTED"    \
         : ((istat) == HIPSPARSE_STATUS_NOT_INITIALIZED)  ? "HIPSPARSE_STATUS_NOT_INITIALIZED"  \
         : ((istat) == HIPSPARSE_STATUS_ALLOC_FAILED)     ? "HIPSPARSE_STATUS_ALLOC_FAILED"     \
         : ((istat) == HIPSPARSE_STATUS_INVALID_VALUE)    ? "HIPSPARSE_STATUS_INVALID_VALUE"    \
         : ((istat) == HIPSPARSE_STATUS_ARCH_MISMATCH)    ? "HIPSPARSE_STATUS_ARCH_MISMATCH"    \
         : ((istat) == HIPSPARSE_STATUS_EXECUTION_FAILED) ? "HIPSPARSE_STATUS_EXECUTION_FAILED" \
         : ((istat) == HIPSPARSE_STATUS_INTERNAL_ERROR)   ? "HIPSPARSE_STATUS_INTERNAL_ERROR"   \
         : ((istat) == HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)                              \
         ? "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"                                         \
         : ((istat) == HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES)                                 \
         ? "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES"                                            \
         : "unknown status code")
#endif

#ifndef HIPSPARSE_CHECK
#define HIPSPARSE_CHECK(fcn, error_code)                                                        \
    {                                                                                           \
        hipsparseStatus_t const istat = (fcn);                                                  \
        if(istat != HIPSPARSE_STATUS_SUCCESS)                                                   \
        {                                                                                       \
            printf("HIPSPARSE API failed at line %d in file %s with error: %s(%d)\n", __LINE__, \
                   __FILE__, hipsparseGetErrorName(istat), istat);                              \
            fflush(stdout);                                                                     \
            return ((error_code));                                                              \
        };                                                                                      \
    };
#endif

#ifndef THROW_IF_HIPSPARSE_ERROR
#define THROW_IF_HIPSPARSE_ERROR(fcn)                                                    \
    {                                                                                    \
        hipsparseStatus_t const istat = (fcn);                                           \
        if(istat != HIPSPARSE_STATUS_SUCCESS)                                            \
        {                                                                                \
            printf("HIPSPARSE failed at %s:%d, with error %s(%d)\n", __FILE__, __LINE__, \
                   hipsparseGetErrorName(istat), istat);                                 \
            fflush(stdout);                                                              \
            throw std::runtime_error(__FILE__);                                          \
        };                                                                               \
    };

#endif

#endif
