#ifndef ROCREFACTOR_IPVEC_H
#define ROCREFACTOR_IPVEC_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

void rocrefactor_ipvec( 
                 hipStream_t stream,
                 int const n,
                 int const * const Q_new2old,
                 int       * const Q_old2new
                 );

#ifdef __cplusplus
};
#endif


#endif
