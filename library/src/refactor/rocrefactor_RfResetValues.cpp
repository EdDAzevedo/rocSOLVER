#include "rocrefactor_RfResetValues.hpp"

extern "C"
rocsolverStatus_t 
rocsolverRfResetValues(
                     int n,
                     int nnzA,
                     int* csrRowPtrA,
                     int* csrColIndA,
                     double* csrValA,
                     int* P,
                     int* Q,
          
                     rocsolverRfHandle_t handle
                     )
{
  return(  
         rocrefactor_RfResetValues<int,long,double>(
                n,
                nnzA,
                csrRowPtrA,
                csrColIndA,
                csrValA,
                P,
                Q 
                ) 
          );
}
