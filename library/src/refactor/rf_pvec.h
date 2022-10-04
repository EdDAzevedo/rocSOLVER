#ifndef RF_PVEC_H
#define RF_PVEC_H


#include <hipsparse/hipsparse.h>

#ifdef __cplusplus
extern "C" {
#endif

void rfDpvec(  
               hipsparseHandle_t handle,
               int const n,
               int const * const d_P_new2old,
               double * d_xold,
               double * d_xnew );

void rfSpvec( 
               hipsparseHandle_t handle,
               int const n,
               int const * const d_P_new2old,
               float * d_xold,
               float * d_xnew );





#ifdef __cplusplus
};
#endif


#endif
