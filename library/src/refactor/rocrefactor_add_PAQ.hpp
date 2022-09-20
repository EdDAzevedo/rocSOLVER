#ifndef ROCREFACTOR_ADD_PAQ_HPP
#define ROCREFACTOR_ADD_PAQ_HPP

template <typename Iint, typename Ilong, typename T>
__global__ void rocrefactor_add_PAQ_kernel(Iint const nrow,
                                           Iint const ncol,
                                           Iint* const* P_new2old,
                                           Iint* const* Q_old2new,
                                           Iint const* const Ap,
                                           Iint const* const Ai,
                                           T const* const Ax,
                                           Iint const* const LUp,
                                           Iint const* const LUi,
                                           T* const LUx)
{
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

    Iint const irow_start = threadsIdx.x + blockIdx.x * blockDim.x;
    Iint const irow_inc = blockDim.x * gridDim.x;

    for(Iint irow = irow_start; irow < nrow; irow += irow_inc)
    {
        Ilong kstart_LU = LUp[irow];
        Ilong kend_LU = LUp[irow + 1];
        Iint  nz_LU = kend_LU - kstart_LU;

        for(Iint k=0; k < nz_LU; k++) {
        {
            Ilong const k_lu = kstart_LU + k;
            LUx[k_lu] = 0;
        };

        Iint const irow_old = P_new2old[irow];
        Ilong const kstart_A = Ap[irow_old];
        Ilong const kend_A = Ap[irow_old + 1];
        Iint const nz_A = kend_A - kstart_A;

        for(Iint k=0; k < nz_A; k++) {
        {
            Ilong const ka = kstart_A + k;

            Iint const jcol_old = Ai[ka];
            Iint const jcol = Q_old2new[jcol_old];

            Iint ipos = -1;
            {
                Iint const len = nz_LU;
                Iint const * const arr = &(LUi[kstart_LU]);
                Iint const key = jcol;

                ipos = search(len, arr, key);
            };
            bool const is_found = (0 <= ipos) && (ipos < len) && (arr[ipos] == key);
            assert( is_found );

            k_lu = kstart_LU + ipos;

            T const aij = Ax[ka];
            LUx[k_lu] += aij;
        };
    };
}

#endif
