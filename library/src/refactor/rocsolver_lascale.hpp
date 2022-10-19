
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
#ifndef ROCSOLVER_LASCALE_HPP
#define ROCSOLVER_SCALE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "rocsolver/rocsolver.h"

#ifndef LASCALE_MAX_THDS
#define LASCALE_MAX_THDS 256
#endif


template< typename Iint, typename Ilong, typename T>
__device__
void
rocsolver_scale_device( 
		Iint const nrow,
		Iint const ncol,
		T const * const drow,
                T const * const dcol,
                Iint const * const Ap,
                Iint const * const Ai,
                T          * const Ax
                )
{

  /*
    -------------------------------------------------
    Perform row and column scaling of sparse matrix

    This is equivalent to computing (in-place)

    A = diag( drow(1:nrow),0) * A * diag( dcol(1:ncol),0)
    -------------------------------------------------
  */

  {
  bool const isok = (nrow >= 1) && (ncol >= 1) && 
                    (Ap != nullptr) && (Ai != nullptr) && (Ax != nullptr);
  if (!isok) { return; };
  };

  Ilong const global_idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  Iint const wid = (global_idx / warpSize );
  Iint const lid = (global_idx % warpSize );

  Iint const irow_start = wid;
  Iint const irow_inc = (hipBlockDim_x * hipGridDim_x )/warpSize;

  for(Iint irow=irow_start; irow < nrow; irow += irow_inc ) {
     Ilong const kstart = Ap[irow];
     Ilong const kend = Ap[irow+1];

     T const drow_i = (drow == nullptr) ? 1 : drow[irow];
     Iint const nz = (kend - kstart);

     for(Iint ik=lid; ik < nz; ik += warpSize ) {
         Ilong const k = kstart + ik;
         Iint const jcol = Ai[k];

         T aij = Ax[k];
         aij = (drow_i == 0) ? 0 : drow_i * aij;

         T const dcol_j = (dcol == nullptr) ? 1 : dcol[jcol];
         aij = (dcol_j == 0) ? 0 : aij * dcol_j;

         Ax[k] = aij;
         };
      };
}



template< typename Iint, typename Ilong, typename T>
__global__ void  __launch_bounds__(LASCALE_MAX_THDS)
rocsolver_scale_kernel( 
		Iint const nrow,
		Iint const ncol,
		T const * const drow,
                T const * const dcol,
                Iint const * const Ap,
                Iint const * const Ai,
                T          * const Ax
                )
{

  rocsolver_scale_device<Iint,Ilong,T>(
	nrow,
	ncol,
	drow,
	dcol,
	Ap,
	Ai,
	Ax
	);

}
                  
         

template< typename Iint, typename T>
void rocsolver_scale_template(
        hipStream_t const stream,
        Iint const nrow,
        Iint const ncol,
        T const * const drow,
        T const * const dcol,
        Iint const * const Ap,
        Iint const * const Ai,
        T          * const Ax
        )
{
  Iint constexpr nthreads = LASCALE_MAX_THDS;
  Iint constexpr nwaves = LASCALE_MAX_THDS/warpSize;

  Iint const nblocks = (nrow + (nwaves-1))/nwaves;

  rocsolver_scale_kernel<Iint,Iint,T><<<
                            dim3(nblocks),
                            dim3(nthreads),
                            0,
                            stream
                            >>>
                            (
                            nrow,
                            ncol,
                            drow,
                            dcol,
                            Ap,
                            Ai,
                            Ax
                          );
        

}

#endif
