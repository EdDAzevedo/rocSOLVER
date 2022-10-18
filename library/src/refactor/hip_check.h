#pragma once
#ifndef HIP_CHECK_H
#define HIP_CHECK_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifndef HIP_CHECK
#define HIP_CHECK( fcn, error_code ) { hipError_t istat = (fcn); \
			       if (istat != HIP_SUCCESS ) { \
				printf("HIP API failed at line %d in file %s with error: %s (%d)\n", \
				 __LINE__, __FILE__, hipGetErrorString(istat), istat); \
				return( error_code ); }; \
			      };
#endif
                                           



#endif

