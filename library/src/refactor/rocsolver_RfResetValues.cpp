#include "rocsolver_refactor.h"
#include "rocsolver_RfResetValues.hpp"

extern "C" {

rocsolverStatus_t rocsolverRfResetValues(
	/* Input (in the device memory) */
	int n,
	int nnzA,
	int* csrRowPtrA,
	int* csrColIndA,
	double* csrValA,
	int* P,
	int* Q,
	/* Output */
	rocsolverRfHandle_t handle 
	)
{
   return(  
     rocsolver_RfResetValues_template<int,int,double>(
	n,
	nnzA,
	csrRowPtrA,
	csrColIndA,
	csrValA,
	P,
	Q,
	handle
	)
    );
};


}
