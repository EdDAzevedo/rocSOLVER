#ifndef ROCREFACTOR_SCALE_HPP
#define ROCREFACTOR_SCALE_HPP

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#ifndef SCALE_MAX_THDS
#define SCALE_MAX_THDS 256
#endif


template< typename Iint, typename T>
ROCSOLVER_KERNEL void __launch_bounds__(SCALE_MAX_THDS)
rocrefactor_scale_kernel( 
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
    equivalent to
    diag( drow(1:nrow),0) * A * diag( dcol(1:ncol),0)
    -------------------------------------------------
  */
  Iint const irow_start = threadIdx.x + blockIdx.x * blockDim.x;
  Iint const irow_inc = blockDim.x * gridDim.x;

  for(Iint irow=irow_start; irow < nrow; irow += irow_inc ) {
     Iint const kstart = Ap[irow];
     Iint const kend = Ap[irow+1];

     T const drow_i = (drow == NULL) ? 1 : drow[irow];
     for(Iint k=kstart; k < kend; kstart++) {
         Iint const jcol = Ai[k];

         T aij = Ax[k];
         aij = (drow_i == 0) ? 0 : drow_i * aij;

         T dcol_j = (dcol == NULL) ? 1 : dcol[jcol];
         aij = (dcol_j == 0) ? 0 : aij * dcol_j;

         Ax[k] = aij;
         };
      };
}
                  
         

template< typename Iint, typename T>
void rocrefactor_scale_template(
        hipStream_t const stream;
        Iint const nrow,
        Iint const ncol,
        T const * const drow,
        T const * const dcol,
        Iint const * const Ap,
        Iint const * const Ai,
        T          * const Ax
        )
{
  Iint const nthreads = SCALE_MAX_THDS;
  Iint const nblocks = (nrow + (nthreads-1))/nthreads;

  ROCSOLVER_LAUNCH_KERNEL(  rocrefactor_scale_kernel<Iint,T>,
                            dim3(nblocks),
                            dim3(nthreads),
                            0,
                            stream,

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
