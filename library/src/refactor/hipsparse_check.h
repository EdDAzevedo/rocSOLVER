#ifndef HIPSPARSE_CHECK_H
#define HIPSPARSE_CHECK_H
#include "hipsparse/hipsparse.h"

#ifndef HIPSPARSE_CHECK
#define HIPSPARSE_CHECK( fcn, error_code ) { hipsparseStatus_t istat = (fcn); \
				     if (istat != HIPSPARSE_STATUS_SUCCESS) { \
				       printf("HIPSPARSE API failed at line %d in file %s with error: %d\n", \
                                               __LINE__, __FILE__, istat ); \
                                               return( (error_code) ); }; \
                                            };
#endif


#endif
