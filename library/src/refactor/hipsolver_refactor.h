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
#pragma once
#ifndef ROCSOLVER_REFACTOR_H
#define ROCSOLVER_REFACTOR_H

#ifdef __HIP_PLATFORM_AMD__
#include "rocsolverRf.h"
#else
#include "cusolverRf.h"
#endif

#ifdef __cplusplu
extern "C" {
#endif

#ifdef __HIP_PLATFORM_AMD__

#define hipsolverRfResetValues rocsolverRfResetValues
#define hipsolverRfRefactor rocsolverRfRefactor
#define hipsolverRfSolve rocsolverRfSolve
#define hipsolverRfSetupDevice rocsolverRfSetupDevice
#define hipsolverRfAnalyze rocsolverRfAnalyze

#define hipsolverRfHandle_t rocsolverRfHandle_t

#else

#define hipsolverRfResetValues cusolverRfResetValues
#define hipsolverRfRefactor cusolverRfRefactor
#define hipsolverRfSolve cusolverRfSolve
#define hipsolverRfSetupDevice cusolverRfSetupDevice
#define hipsolverRfAnalyze cusolverRfAnalyze

#define hipsolverRfHandle_t cusolverRfHandle_t

#endif

#ifdef __cplusplu
};
#endif

#endif
