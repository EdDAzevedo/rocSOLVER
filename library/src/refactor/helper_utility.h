#ifndef HELPER_UTILITY_H
#define HELPER_UTILITY_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
#else
#define __HIP_PLATFORM_AMD__
#endif

#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>

#include "hip_check.h"
#include "hipsparse_check.h"

static bool is_device_pointer( const void *ptr) 
{
   hipPointerAttribute_t attributes;
   hipError_t istat = ( hipPointerGetAttributes( &attributes, ptr ) );
   if (istat != hipSuccess) { return(false); };

   return( (attributes.memoryType == hipMemoryTypeDevice) ||
           (attributes.memoryType == hipMemoryTypeArray) );
}
   
static bool is_host_pointer( const void *ptr) 
{
   hipPointerAttribute_t attributes;
   hipError_t istat = ( hipPointerGetAttributes( &attributes, ptr ) );
   if (istat != hipSuccess) { return(false); };

   return( (attributes.memoryType == hipMemoryTypeHost) );


}


static bool is_unified_pointer( const void *ptr) 
{
   hipPointerAttribute_t attributes;
   hipError_t istat = ( hipPointerGetAttributes( &attributes, ptr ) );

   if (istat != hipSuccess) { return(false); };
   return( (attributes.memoryType == hipMemoryTypeUnified) );


}



static bool is_manged_pointer( const void *ptr) 
{
   hipPointerAttribute_t attributes;
   hipError_t istat = ( hipPointerGetAttributes( &attributes, ptr ) );

   if (istat != hipSuccess) { return(false); };
   return( (attributes.isManaged) ); 

}

#endif
