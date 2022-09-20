#ifndef ROCREFACTOR_AXPBY_HPP
#define ROCREFACTOR_AXPBY_HPP

#ifndef AXPBY_MAX_THDS
#define AXPBY_MAX_THDS 256
#endif

template <typename Iint, typename Ilong, typename T>
__global__ launch_bounds(AXPBY_MAX_THDS)
void rocrefactor_aXpbY_kernel(           
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

    auto search = [](Iint const len, Iint const* const arr, Iint const key) -> Iint {

        /*
         ---------------------------------------
         search array  arr[0], ..., arr[ len-1] 
         for matching value "key"

         return the index value of matching position
         ---------------------------------------
         */
        Iint constexpr small_len = 8;
        Iint ipos = len;
        if((len <= 0) || (arr == NULL))
        {
            return (ipos);
        };

        if(len <= small_len)
        {
          /*  
            -----------------
            use simple linear search  
            -----------------
           */
            #pragma unroll
            for(Iint k = 0; k < len; k++)
            {
                bool const is_found = (arr[k] == key);
                if(is_found)
                {
                    ipos = k;
                    break;
                };
            };
        }
        else
        {
         /*
          -----------------
          use binary search
          -----------------
          */
            Iint lo = 0;
            Iint hi = len;

	    #pragma unroll
            for(int i = 0; i < 32; i++)
            {
                if(lo >= hi)
                {
                    break;
                };

                Iint mid = (lo + hi) / 2;
                bool const is_found = (arr[mid] == key);
                if(is_found)
                {
                    ipos = mid;
                    break;
                };

                if(arr[mid] < key)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                };
            };
        };
        return (ipos);
    };

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

                    Iint const ipos = search(len, arr, key);
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
void rocrefactor_aXpbY_template( 
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

     rocrefactor_aXpbY_template<Iint,Ilong,T><<< 
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
