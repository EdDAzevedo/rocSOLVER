#ifndef HIPSPARSE_CHECK_H
#define HIPSPARSE_CHECK_H
#include "hipsparse.h"

#ifndef HIPSPARSE_CHECK
#define HIPSPARSE_CHECK( fcn, error_code ) { hipsparseStatus_t istat = (fcn); \
                                             if (istat != HIPSPARSE_STATUS_SUCCESS) { return( (error_code) ); }; \
                                            };
#endif


#endif
