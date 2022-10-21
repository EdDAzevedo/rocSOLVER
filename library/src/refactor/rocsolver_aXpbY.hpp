#ifndef ROCSOLVER_AXPBY_HPP
#define ROCSOLVER_AXPBY_HPP

#ifndef AXPBY_MAX_THDS
#define AXPBY_MAX_THDS 256
#endif

template <typename Iint, typename Ilong, typename T>
__global__ __launch_bounds__(AXPBY_MAX_THDS)
void rocsolver_aXpbY_kernel(           
                                         Iint const nrow,
                                         Iint const ncol,
                                         T const alpha,
                                         Iint const* const Xp,
                                         Iint const* const Xi,
                                         T const* const Xx,
                                         T const beta,
                                         Iint const* const Yp,
                                         Iint const* const Yi,
                                         T const Yx)
{
/*
 ------------------------------------------------
 Perform  Y = alpha * X + beta * Y
 where sparsity pattern of matrix X is a subset of
 sparsity pattern of matrix Y
 ------------------------------------------------
*/
    {
        bool const isok = (nrow >= 1) && (ncol >= 1) && (Xp != NULL) && (Xi != NULL) && (Xx != NULL)
            && (Yp != NULL) && (Yi != NULL) && (Yx != NULL);
        if(!isok)
        {
            return;
        };
    }

#include "rf_search.hpp"

    Iint const irow_start = threadIdx.x + blockIdx.x * blockDim.x;
    Iint const irow_inc = blockDim.x * gridDim.x;

    for(Iint irow = irow_start; irow < nrow; irow += irow_inc)
    {
        Ilong const kx_start = Xp[irow];
        Ilong const kx_end = Xp[irow + 1];
        Ilong const ky_start = Yp[irow];
        Ilong const ky_end = Yp[irow + 1];

        if(beta == 0)
        {
            for(Ilong ky = ky_start; ky < ky_end; ky++)
            {
                Yx[ky] = 0;
            };
        };

        if(alpha == 0)
        {
         /*
         -------------------
         just scale matrix Y
         -------------------
         */
            for(Ilong ky = ky_start; ky < ky_end; ky++)
            {
                Yx[ky] *= beta;
            };
        }
        else
        {
            for(Ilong kx = kx_start; kx < kx_end; kx++)
            {
                Iint const jcol = Xi[kx];

                /*
        ---------------------
        perform search
        ---------------------
        */
                bool is_found = false;
                {
                    Iint const key = jcol;
                    Iint const len = (ky_end - ky_start);
                    Iint const* const arr = &(Yp[ky_start]);

                    Iint const ipos = rf_search(len, arr, key);
                    is_found = (0 <= ipos) && (ipos < len) && (arr[ipos] == key);
                };

                if(is_found)
                {
                    Ilong const ky = ky_start + ipos;
                    Y[ky] = alpha * Xx[kx] + beta * Yx[ky];
                };
            };
        };
    };
}

template< typename Iint, typename Ilong, typename T>
void rocsolver_aXpbY_template( 
                 hipStream_t stream,
                 Iint const nrow,
                 Iint const ncol,
                 T const alpha,
                 Iint const * const Xp,
                 Iint const * const Xi,
                 T    const * const Xx,
                 T const beta,
                 Iint const * const Yp,
                 Iint const * const Yi,
                 Iint       * const Yx
                 )
{
     Iint const nthreads = AXPBY_MAX_THDS;
     Iint const nblocks = (nrow + (nthreads-1))/nthreads;

     rocsolver_aXpbY_template<Iint,Ilong,T><<< 
                             dim3(nblocks), 
                             dim3(nthreads),
                             0,
                             stream >>>(
                                nrow,
                                ncol,
                                alpha,
                                Xp,
                                Xi,
                                Xx,
                                beta,
                                Yp,
                                Yi,
                                Yx
                                );
}
           
               

#endif
