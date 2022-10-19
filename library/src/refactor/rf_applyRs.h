#ifndef RF_APPLYRS_H
#define RF_APPLYRS_H

#include <hipsparse/hipsparse.h>

#ifdef __cplusplus
extern "C" {
#endif



void rfDapplyRs( 
          hipsparseHandle_t handle,
          int const n,
          double const * const Rs,
          double *b
          );

void rfSapplyRs( 
          hipsparseHandle_t handle,
          int const n,
          float const * const Rs,
          float *b
          );

#ifdef __cplusplus
};
#endif

#endif
