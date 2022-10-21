#ifndef RF_MIRROR_POINTER_H
#define RF_MIRROR_POINTER_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "hip_check.h"

#ifdef __cplusplus
extern "C" {
#endif

void * rf_mirror_pointer( 
                        void * const h_ptr, 
                        size_t const nbytes 
                        );

#ifdef __cplusplus
};
#endif

#endif
