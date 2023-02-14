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
This routine extracts lower (L) and upper (U) triangular factors from
the rocSovlerRF library handle into the host memory.  The factors
are compressed into a single matrix M = (L-I)+U, where the unitary
diagonal of (L) is not stored.  It is assumed that a prior call to the
rocsolverRfRefactor() was called to generate the triangular factors.
----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfExtractSplitFactorsHost(rocsolverRfHandle_t handle,
                                                     /* Output in host memory */
                                                     int* h_nnzL,
                                                     int** h_Lp,
                                                     int** h_Li,
                                                     double** h_Lx,
                                                     int* h_nnzU,
                                                     int** h_Up,
                                                     int** h_Ui,
                                                     double** h_Ux)

{
    // ------------
    // check handle
    // ------------
    {
        bool const isok = (handle != nullptr);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_NOT_INITIALIZED);
        };
    };

    // ---------------
    // check arguments
    // ---------------
    {
        bool const isok_L
            = (h_nnzL != nullptr) && (h_Lp != nullptr) && (h_Li != nullptr) && (h_Lx != nullptr);

        bool const isok_U
            = (h_nnzU != nullptr) && (h_Up != nullptr) && (h_Ui != nullptr) && (h_Ux != nullptr);

        bool const isok = (isok_L && isok_U);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    // -----------------------
    // extract M = (L - I) + U
    // -----------------------
    int nnzM = 0;
    int* Mp = nullptr;
    int* Mi = nullptr;
    double* Mx = nullptr;
    {
        rocsolverStatus_t istat = rocsolverRfExtractBundledFactorsHost(handle, &nnzM, &Mp, &Mi, &Mx);

        if(istat != ROCSOLVER_STATUS_SUCCESS)
        {
            return (istat);
        };
    };

    // --------------------
    // split M = (L-I) + U  into L and U
    // --------------------
    int const n = handle->n;
    int* const Lp = (int*)malloc(sizeof(int) * (n + 1));
    int* const Up = (int*)malloc(sizeof(int) * (n + 1));
    int* const nzLp = (int*)malloc(sizeof(int) * n);
    int* const nzUp = (int*)malloc(sizeof(int) * n);

    {
        bool const is_alloc_ok
            = (Lp != nullptr) && (Up != nullptr) && (nzLp != nullptr) && (nzUp != nullptr);
        if(!is_alloc_ok)
        {
            // -------------------------------------------
            // deallocate host memory to avoid memory leak
            // -------------------------------------------
            if(Lp != nullptr)
            {
                free(Lp);
            };
            if(Up != nullptr)
            {
                free(Up);
            };
            if(nzLp != nullptr)
            {
                free(nzLp);
            };
            if(nzUp != nullptr)
            {
                free(nzUp);
            };

            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    // -------------------------------------------------
    // 1st pass to determine number of non-zeros per row
    // -------------------------------------------------
    for(int i = 0; i < n; i++)
    {
        nzLp[i] = 0;
        nzUp[i] = 0;
    };

    int nnzL = 0;
    int nnzU = 0;
    for(int irow = 0; irow < n; irow++)
    {
        int const istart = Mp[irow];
        int const iend = Mp[irow + 1];
        int const nz = (iend - istart);

        int nzU = 0;
        for(int k = istart; k < iend; k++)
        {
            int const kcol = Mi[k];
            bool const is_upper = (irow <= kcol);
            if(is_upper)
            {
                nzU++;
            };
        };
        int const nzL = nz - nzU;

        nzLp[irow] = (nzL + 1); // add 1 for unit diagonal
        nzUp[irow] = nzU;

        nnzL += (nzL + 1);
        nnzU += nzU;
    };

    int* const Li = (int*)malloc(sizeof(int) * nnzL);
    int* const Ui = (int*)malloc(sizeof(int) * nnzU);
    double* const Lx = (double*)malloc(sizeof(double) * nnzL);
    double* const Ux = (double*)malloc(sizeof(double) * nnzU);

    {
        bool const is_alloc_ok
            = (Li != nullptr) && (Ui != nullptr) && (Lx != nullptr) && (Ux != nullptr);
        if(!is_alloc_ok)
        {
            // -----------------------------------------------
            // deallocate all host arrays to avoid memory leak
            // -----------------------------------------------
            if(Li != nullptr)
            {
                free(Li);
            };
            if(Ui != nullptr)
            {
                free(Ui);
            };
            if(Lx != nullptr)
            {
                free(Lx);
            };
            if(Ux != nullptr)
            {
                free(Ux);
            };

            if(Lp != nullptr)
            {
                free(Lp);
            };
            if(Up != nullptr)
            {
                free(Up);
            };
            if(nzLp != nullptr)
            {
                free(nzLp);
            };
            if(nzUp != nullptr)
            {
                free(nzUp);
            };

            if(Mp != nullptr)
            {
                free(Mp);
            };
            if(Mi != nullptr)
            {
                free(Mi);
            };
            if(Mx != nullptr)
            {
                free(Mx);
            };

            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    // ------------------------------------
    // prefix sum scan to setup Lp and Up
    // ------------------------------------
    int iL = 0;
    int iU = 0;
    for(int irow = 0; irow < n; irow++)
    {
        int const nzL = nzLp[irow];
        int const nzU = nzUp[irow];
        Lp[irow] = iL;
        iL += nzL;

        Up[irow] = iU;
        iU += nzU;
    };
    Up[n] = nnzU;
    Lp[n] = nnzL;

    // ---------------------------------------------------
    // second pass to populate  Li[], Lx[], Ui[], Ux[]
    // ---------------------------------------------------

    for(int irow = 0; irow < n; irow++)
    {
        nzLp[irow] = Lp[irow];
        nzUp[irow] = Up[irow];
    };

    double const one = 1;

    for(int irow = 0; irow < n; irow++)
    {
        int const istart = Mp[irow];
        int const iend = Mp[irow + 1];
        for(int k = istart; k < iend; k++)
        {
            int const kcol = Mi[k];
            double const mij = Mx[k];
            bool const is_upper = (irow <= kcol);
            if(is_upper)
            {
                int const ip = nzUp[irow];
                nzUp[irow]++;

                Ui[ip] = kcol;
                Ux[ip] = mij;
            }
            else
            {
                int const ip = nzLp[irow];
                nzLp[irow]++;

                Li[ip] = kcol;
                Lx[ip] = mij;
            };
        };
    };

    // ------------------------
    // set unit diagonal entry in L
    // ------------------------
    for(int irow = 0; irow < n; irow++)
    {
        int const kend = Lp[irow + 1];
        int const ip = kend - 1;
        Li[ip] = irow;
        Lx[ip] = one;
    };

    *h_nnzL = nnzL;
    *h_Lp = Lp;
    *h_Li = Li;
    *h_Lx = Lx;

    *h_nnzU = nnzU;
    *h_Up = Up;
    *h_Ui = Ui;
    *h_Ux = Ux;

    // -----------------
    // clean up matrix M
    // -----------------

    free(nzLp);
    free(nzUp);

    free(Mp);
    free(Mi);
    free(Mx);

    return (ROCSOLVER_STATUS_SUCCESS);
};
};
