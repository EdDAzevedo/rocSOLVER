
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
#ifndef RF_SUMLU_HPP
#define RF_SUMLU_HPP

#include "rf_common.hpp"
#include "rf_shellsort.hpp"

template <typename Iint, typename Ilong, typename T>
static __global__ void rf_setupLUp_kernel(Iint const nrow,
                                          Iint const ncol,

                                          Ilong const* const Lp,
                                          Ilong const* const Up,

                                          Ilong* const LUp)
{
    // ---------------------------
    // setup LUp row pointer array
    // ---------------------------

    {
        // ---------------------------------------------
        // just use a single thread block for simplicity
        // ---------------------------------------------
        bool const is_root = (blockIdx.x == 0);
        if(!is_root)
        {
            return;
        };
    };

    // -----------------------------------------
    // compute number of non-zeros per row in LU
    // -----------------------------------------

    Iint const nthreads = blockDim.x;
    Iint const i_start = threadIdx.x;
    Iint const i_inc = nthreads;
    for(Iint i = i_start; i < nrow; i += i_inc)
    {
        Iint const irow = i;
        Ilong const nnz_L = Lp[irow + 1] - Lp[irow];
        Ilong const nnz_U = Up[irow + 1] - Up[irow];
        Ilong const nnz_LU = (nnz_L - 1) + nnz_U;

        LUp[irow] = nnz_LU;
    };

    __syncthreads();

    // ---------------------------------------
    // prepare for prefix sum in shared memory
    // ---------------------------------------

    int constexpr MAX_THREADS = 1024;
    __shared__ Ilong isum[MAX_THREADS];

    for(Iint i = i_start; i < nthreads; i += i_inc)
    {
        // ---------------------------------------------
        // the i-th thread computes
        //
        //    isum[ i ] = sum( LUp[ istart..(iend-1) ] )
        // ---------------------------------------------
        Iint const nb = (nrow + (nthreads - 1)) / nthreads;
        Iint const istart = i * nb;
        Iint const iend = min(nrow, istart + nb);

        Ilong presum = 0;
        for(Iint irow = istart; irow < iend; irow++)
        {
            Ilong const nnz_LU = LUp[irow];
            presum += nnz_LU;
        };
        isum[i] = presum;
    };

    __syncthreads();

    // ------------------
    // compute prefix sum
    // use single thread for simplicity
    // ------------------

    Ilong offset = 0;
    Ilong nnz_LU = 0;
    bool const is_root_thread = (threadIdx.x == 0);
    if(is_root_thread)
    {
        for(Iint i = 0; i < nthreads; i++)
        {
            Ilong isum_i = isum[i];
            isum[i] = offset;
            offset += isum_i;
        };

        nnz_LU = offset;
        LUp[nrow] = nnz_LU;
    };

    __syncthreads();

    // ----------
    // update LUp
    // ----------

    for(Iint i = i_start; i < nthreads; i += i_inc)
    {
        Iint const nb = (nrow + (nthreads - 1)) / nthreads;
        Iint const istart = i * nb;
        Iint const iend = min(nrow, istart + nb);

        Ilong ipos = isum[i];
        for(Iint irow = istart; irow < iend; irow++)
        {
            Ilong const nz = LUp[irow];
            LUp[irow] = ipos;
            ipos += nz;
        };
    };

    __syncthreads();
}

template <typename Iint, typename Ilong, typename T>
static __global__ void rf_sumLU_kernel(Iint const nrow,
                                       Iint const ncol,

                                       Ilong const* const Lp,
                                       Iint const* const Li,
                                       T const* const Lx,

                                       Ilong const* const Up,
                                       Iint const* const Ui,
                                       T const* const Ux,

                                       Ilong* const LUp,
                                       Iint* const LUi,
                                       T* const LUx)
{
    // ------------------------------
    // Assume LUp array already setup
    // ------------------------------

    Ilong const nnzL = Lp[nrow] - Lp[0];
    Ilong const nnzU = Up[nrow] - Up[0];
    Ilong const nnzLU = LUp[nrow] - LUp[0];

    bool const isok = (nnzLU == (nnzL + nnzU - nrow));
    assert(isok);

    Iint const irow_start = threadIdx.x + blockIdx.x * blockDim.x;
    Iint const irow_inc = blockDim.x * gridDim.x;
    for(Iint irow = irow_start; irow < nrow; irow += irow_inc)
    {
        Ilong const kstart_L = Lp[irow];
        Ilong const kend_L = Lp[irow + 1];
        Iint const nz_L = (kend_L - kstart_L);

        Ilong const kstart_U = Up[irow];
        Ilong const kend_U = Up[irow + 1];
        Iint const nz_U = (kend_U - kstart_U);

        Ilong const kstart_LU = LUp[irow];
        Ilong const kend_LU = LUp[irow + 1];
        Iint const nz_LU = (kend_LU - kstart_LU);

        // --------------
        // copy L into LU
        // --------------
        Ilong ip = kstart_LU;
        for(Iint k = 0; k < nz_L; k++)
        {
            Ilong const jp = kstart_L + k;
            Iint const jcol = Li[jp];
            T const Lij = Lx[jp];
            bool const is_diag = (jcol == irow);
            if(!is_diag)
            {
                LUi[ip] = jcol;
                LUx[ip] = Lij;

                ip++;
            };
        };

        // --------------
        // copy U into LU
        // --------------
        for(Iint k = 0; k < nz_U; k++)
        {
            Ilong const jp = kstart_U + k;

            Iint const jcol = Ui[jp];
            T const Uij = Ux[jp];

            LUi[ip] = jcol;
            LUx[ip] = Uij;

            ip++;
        };

        // ----------------------------------------
        // check column indices are in sorted order
        // ----------------------------------------
        bool is_sorted = true;
        for(Iint k = 0; k < (nz_LU - 1); k++)
        {
            Ilong const ip = kstart_LU + k;
            is_sorted = (LUi[ip] < LUi[ip + 1]);
            if(!is_sorted)
            {
                break;
            };
        };

        if(!is_sorted)
        {
            // ----------------------------------
            // sort row in LU using shellsort algorithm
            // ----------------------------------
            Iint* iarr = &(LUi[kstart_LU]);
            T* darr = &(LUx[kstart_LU]);
            Iint num = nz_L;

            rf_shellsort(iarr, darr, num);
        };

    }; // end for irow
}

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rf_sumLU(rocsolverRfHandle_t handle,
                           Iint const nrow,
                           Iint const ncol,

                           Ilong const* const Lp,
                           Iint const* const Li,
                           T const* const Lx,

                           Ilong const* const Up,
                           Iint const* const Ui,
                           T const* const Ux,

                           Ilong* const LUp,
                           Iint* const LUi,
                           T* const LUx

)
{
    //  ----------------
    //  form (L - I) + U
    //  assume storage for LUp, LUi, LUx has been allocated
    // ---------------------------------------------------

    hipStream_t streamId;
    HIPSPARSE_CHECK(hipsparseGetStream(handle->hipsparse_handle, &streamId),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    // -----------------------------
    // Step 1: setup row pointer LUp
    // -----------------------------
    {
        Iint const nthreads = 1024;
        Iint const nblocks = 1; // special case

        rf_setupLUp_kernel<Iint, Ilong, T>
            <<<dim3(nthreads), dim3(nblocks), 0, streamId>>>(nrow, ncol, Lp, Up, LUp);
    };

    // ----------------------------------------------------
    // Step 2: copy entries of Li, Lx, Ui, Ux into LUi, LUx
    // ----------------------------------------------------
    {
        Iint const nthreads = 1024;
        Iint const nblocks = (nrow + (nthreads - 1)) / nthreads;

        rf_sumLU_kernel<Iint, Ilong, T><<<dim3(nthreads), dim3(nblocks), 0, streamId>>>(
            nrow, ncol, Lp, Li, Lx, Up, Ui, Ux, LUp, LUi, LUx);
    };

    return (ROCSOLVER_STATUS_SUCCESS);
}

#endif
