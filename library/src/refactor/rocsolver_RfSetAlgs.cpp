
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

#include "rf_common.hpp"

/*
---------------------------------------------------------------------- 
This routine sets the algorithm used for the refactorization in
rocsolverRfRefactor() and the triangular solve in rocsolverRfSolve().
It may be called once prior to rocsolverRfAnalyze() routine.

Note the factorization algorithm need to be compatible with the solver
algorithm.
---------------------------------------------------------------------- 
*/

extern "C" {

rocsolverStatus_t rocsolverRfSetAlgs(rocsolverRfHandle_t handle,
                                     gluFactorization_t fact_alg,
                                     gluTriangularSolve_t solve_alg)
{
    // not fully implemented yet
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    {
        bool const is_valid_fact_alg = (fact_alg == ROCSOLVERRF_FACTORIZATION_ALG0)
            || (fact_alg == ROCSOLVERRF_FACTORIZATION_ALG1)
            || (fact_alg == ROCSOLVERRF_FACTORIZATION_ALG2);

        bool const is_valid_solve_alg = (solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1)
            || (solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG2)
            || (solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG3);

        bool const is_valid = (is_valid_fact_alg && is_valid_solve_alg);

        if(!is_valid)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    // ------------------------
    // check compatible options
    // ------------------------
    {
        bool const is_compatible_case1 = ((fact_alg == ROCSOLVERRF_FACTORIZATION_ALG0)
                                          && (solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1));

        bool const is_compatible_case2 = ((fact_alg == ROCSOLVERRF_FACTORIZATION_ALG1)
                                          && ((solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG2)
                                              || (solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG3)));

        bool const is_compatible_case3 = ((fact_alg == ROCSOLVERRF_FACTORIZATION_ALG2)
                                          && ((solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG2)
                                              || (solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG3)));

        bool const is_compatible = is_compatible_case1 || is_compatible_case2 || is_compatible_case3;
        if(!is_compatible)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    handle->fact_alg = fact_alg;
    handle->solve_alg = solve_alg;

    return (ROCSOLVER_STATUS_SUCCESS);
};
};
