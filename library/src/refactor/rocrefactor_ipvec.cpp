#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "rocrefactor_ipvec.h"


#ifndef BLOCKSIZE 
#define BLOCKSIZE 256
#endif

__global__ 
void rocrefactor_ipvec_kernel( 
                     int const n,
                     int const * const Q_new2old, /* input */
                     int       * const Q_old2new /* output */
                     )
{


  int const i_start = blockIdx.x + gridIdx.x * blockDim.x;
  int const i_inc = blockDim.x * gridDim.x;

  for(int i=i_start; i < n; i += i_inc) {
      int const inew = i;
      int const iold = Q_new2old[ inew ];
      Q_old2new[ iold ] = inew;
      };

}

extern "C"
void rocrefactor_ipvec(
                hipStream_t stream,
                int const n,
                int const * const Q_new2old,
                int       * const Q_old2new
                )
{
  if (n <= 0) { return; };
  if ((Q_new2old == NULL) || (Q_old2new == NULL)) { return; };

  int nblocks = (n + (BLOCKSIZE-1))/BLOCKSIZE; 
  rocrefactor_ipvec<<< dim3(nblocks), dim3(BLOCKSIZE), 0, stream >>>(
                 n,
                 Q_new2old,
                 Q_old2new
                 );
}
  
 
