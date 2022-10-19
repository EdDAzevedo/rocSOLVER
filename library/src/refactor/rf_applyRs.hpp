#pragma once
#ifndef RF_APPLYRS_HPP
#define RF_APPLYRS_HPP

#include <hipsparse/hipsparse.h>
#include "hip_check.h"
#include "hipsparse_check.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE 1024
#endif

template< typename T>
__global__
void rf_applyRs_kernel(
	int const n,
        T const * const Rs,
        T * const b 
        )
{


 if ((n <= 0) || (Rs == NULL)) { return; };

 unsigned int i_start = threadIdx.x + blockIdx.x * blockDim.x;
 unsigned int i_inc = blockDim.x * gridDim.x;

 for(unsigned int i=i_start; i < n; i += i_inc) {
   if (Rs[i] != 0) { b[i] = b[i] / Rs[i]; };
   };
}


template<typename T>
void rf_applyRs_template(
        hipsparseHandle_t handle,
        int const n,
        T const * const d_Rs,
        T * const d_b
        )
{

 assert( d_b != NULL );
 unsigned int nthreads = BLOCKSIZE;
 unsigned int min_nblocks = (n + (nthreads-1))/nthreads;
 unsigned int nblocks = (min_nblocks <= 0) ? 1 : min_nblocks;

  hipStream_t streamId;
  HIPSPARSE_CHECK( hipsparseGetStream( handle, &streamId ) );

 rf_applyRs_kernel<<< dim3(nblocks), dim3(nthreads), 0, streamId >>>(
             n, 
             d_Rs,
             d_b
             );
}
   



#endif

