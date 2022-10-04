#pragma once
#ifndef HIP_CHECK_H
#define HIP_CHECK_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifndef HIP_CHECK
#define HIP_CHECK( fcn, error_code ) { hipStatus_t istat = (fcn); \
                                       if (istat != HIP_SUCCESS ) { return( error_code ); }; \
                                      };
#endif
                                           



#endif

