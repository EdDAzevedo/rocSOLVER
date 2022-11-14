/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/
#ifndef CAL_INDEX_H
#define CAL_INDEX_H

#include <assert.h>

#define is_ijk(nrow,ncol,ld,stride,batchCount) \
                 (((ld) >= (nrow)) && ((stride) >= (ld) * (ncol)))

#define is_ikj(nrow,ncol,ld,stride,batchCount) \
                 (((ld) >= (stride) * (batchCount) ) && ((stride) >= (nrow)))

#define is_kij(nrow,ncol,ld,stride,batchCount) \
                 (((stride) == 1) && ((ld) >= (nrow)))


#define CAL_INDEX(nrow,ncol, ld,stride, batchCount, ci,cj,ck) { \
                 bool const is_ijk0 = is_ijk(nrow,ncol,ld,stride,batchCount); \
                 bool const is_ikj0 = is_ikj(nrow,ncol,ld,stride,batchCount); \
                 bool const is_kij0 = is_kij(nrow,ncol,ld,stride,batchCount); \
                 int ncase = 0; \
                 if (is_ijk0) { ncase++; }; \
                 if (is_ikj0) { ncase++; }; \
                 if (is_kij0) { ncase++; }; \
                 bool const is_valid = (ncase == 1); \
                 assert( is_valid ); \
                 if (is_ijk0) { \
                      /*  A(i,j,k)  A_[ i + j*ld + k * stride ] */ \
                      ci = 1; cj = ld; ck = stride; \
                      }; \
                 if (is_ikj0) { \
                     /*  A(i,j,k)  A_[ i + k * stride + j * ld ] */ \
                     ci = 1; cj = ld; ck = stride; \
                     }; \
                 if (is_kij0) { \
                     /* A(i,j,k)  A_[ k + i * batchCount + j * (batchCount * ld) ] */ \
                     ci = batchCount; ck = 1; \
                     cj = batchCount; cj *= ld; \
                     }; \
                  }



#endif
