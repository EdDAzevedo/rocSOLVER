
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
#pragma once
#ifndef HIPSOLVER_ENUM_H
#define HIPSOLVER_ENUM_H

typedef enum
{
    HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0, //default
    HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
} hipsolverRfResetValuesFastMode_t;

// typedef hipsolverRfResetValuesFastMode_t gluResetValuesFastMode_t;

/* HIPSOLVERRF matrix format */
typedef enum
{
    HIPSOLVERRF_MATRIX_FORMAT_CSR = 0, //default
    HIPSOLVERRF_MATRIX_FORMAT_CSC = 1
} hipsolverRfMatrixFormat_t;

// typedef hipsolverRfMatrixFormat_t gluMatrixFormat_t;

/* HIPSOLVERRF unit diagonal */
typedef enum
{
    HIPSOLVERRF_UNIT_DIAGONAL_STORED_L = 0, //default
    HIPSOLVERRF_UNIT_DIAGONAL_STORED_U = 1,
    HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,
    HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3
} hipsolverRfUnitDiagonal_t;

// typedef hipsolverRfUnitDiagonal_t gluUnitDiagonal_t;

/* HIPSOLVERRF factorization algorithm */
typedef enum
{
    HIPSOLVERRF_FACTORIZATION_ALG0 = 0, // default
    HIPSOLVERRF_FACTORIZATION_ALG1 = 1,
    HIPSOLVERRF_FACTORIZATION_ALG2 = 2,
} hipsolverRfFactorization_t;

// typedef hipsolverRfFactorization_t gluFactorization_t;

/* HIPSOLVERRF triangular solve algorithm */
typedef enum
{
    HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1, // default
    HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
} hipsolverRfTriangularSolve_t;

// typedef hipsolverRfTriangularSolve_t gluTriangularSolve_t;

/* HIPSOLVERRF numeric boost report */
typedef enum
{
    HIPSOLVERRF_NUMERIC_BOOST_NOT_USED = 0, //default
    HIPSOLVERRF_NUMERIC_BOOST_USED = 1
} hipsolverRfNumericBoostReport_t;
#endif
