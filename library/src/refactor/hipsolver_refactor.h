#ifndef ROCSOLVER_REFACTOR_H
#define ROCSOLVER_REFACTOR_H

#ifdef __HIP_PLATFORM_AMD__
#include "rocsolverRf.h"
#else
#include "cusolverRf.h"
#endif

#ifdef __cplusplu
extern "C" {
#endif


#ifdef __HIP_PLATFORM_AMD__

#define hipsolverRfResetValues rocsolverRfResetValues
#define hipsolverRfRefactor rocsolverRfRefactor
#define hipsolverRfSolve rocsolverRfSolve
#define hipsolverRfSetupDevice rocsolverRfSetupDevice
#define hipsolverRfAnalyze rocsolverRfAnalyze

#define hipsolverRfHandle_t rocsolverRfHandle_t

#else

#define hipsolverRfResetValues cusolverRfResetValues
#define hipsolverRfRefactor cusolverRfRefactor
#define hipsolverRfSolve cusolverRfSolve
#define hipsolverRfSetupDevice cusolverRfSetupDevice
#define hipsolverRfAnalyze cusolverRfAnalyze

#define hipsolverRfHandle_t cusolverRfHandle_t

#endif







#ifdef __cplusplu
};
#endif

#endif
