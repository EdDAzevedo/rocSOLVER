#ifndef RF_IPVEC_H
#define RF_IPVEC_H


#include <hipsparse/hipsparse.h>

#ifdef __cplusplus
extern "C" {
#endif

void rfDipvec(  
               hipsparseHandle_t handle,
               int const n,
               int const * const d_P_new2old,
               double * d_b,
               double * d_x );

void rfSipvec( 
               hipsparseHandle_t handle,
               int const n,
               int const * const d_P_new2old,
               float * d_b,
               float * d_x );





#ifdef __cplusplus
};
#endif


#endif
