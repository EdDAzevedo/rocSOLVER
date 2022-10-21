#include "rf_mirror_pointer.h"

void *rf_mirror_pointer( void * const h_ptr, size_t const nbytes ) 
{
 if ((nbytes <= 0) || (h_ptr == NULL))  { return( NULL ); };

 void * d_ptr = NULL;
 HIP_CHECK( hipMalloc( (void **) &d_ptr, nbytes ));
 assert( d_ptr != NULL );

 HIP_CHECK( hipMemcpy( d_ptr, h_ptr, nbytes, hipMemcpyHostToDevice ));

 return( d_ptr );
}


