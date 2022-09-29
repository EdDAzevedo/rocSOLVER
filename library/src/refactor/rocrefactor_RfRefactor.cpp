/*
 ---------------------------------------------
 This routine performs the LU re-factorization
 ---------------------------------------------
 */
rocsolverStatus_t rocrefactor_RfRefactor( rocsolverRfHandle_t handle )
{

 if (handle == nullptr) { 
       return( ROCSOLVER_STATUS_NOT_INITIALIZED ); 
       };




   csrilu02Info_t info;
   HIPSPARSE_CHECK( hipsparseCreateCsrilu02Info( &info ), 
              ROCSOLVER_STATUS_EXECUTION_FAILED );

         int *csrSortedRowPtrA = handle->csrRowPtrLU;
         int *csrSortedColIndA = handle->csrColIndLU;
         double *csrSortedValA = handle->csrValLU;

         int n = handle->n;
         int nnz = handle->nnz_LU;
         hipsparseMatDescr_t descrA = handle->descrLU;
          


         HIPSPARSE_CHECK( hipsparseDcsrilu02_bufferSize(
	                       handle->hipsparse_handle,
			       n,
			       nnz,
			       descrA,
			       csrSortedValA,
			       csrSortedRowPtrA,
			       csrSortedColIndA,
			       info,
			       &BufferSizeInBytes_int ),
                 ROCSOLVER_STATUS_EXECUTION_FAILED );

        double *pBuffer = nullptr;
        {
        size_t const BufferSizeInBytes = BufferSizeInBytes_int;
        HIP_CHECK( hipMalloc( (void **) &pBuffer, BufferSizeInBytes ),
                    ROCSOLVER_STATUS_ALLOC_ERROR);
        };

        /*
	 ---------------------------------------------------------------------------------
	 perform analysis

	 note policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL or HIPSPARSE_SOLVE_POLICY_NO_LEVEL
	 ---------------------------------------------------------------------------------
	 */
	 hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

	 HIPSPARSE_CHECK( hipsparseDcsrilu02_analysis( handle->hipsparse_handle,
	                                               m,
						       nnz,
						       descrA,
						       csrSortedValA,
						       csrSortedRowPtrA,
						       csrSortedColIndA,
						       info,
						       policy,
						       pBuffer
						       ),
              ROCSOLVER_STATUS_EXECUTION_FAILED );


	/*
	 ---------------------
	 perform factorization
	 ---------------------
	 */
        HIPSPARSE_CHECK( hipsparseDcsrilu02( handle->hipsparse_handle,
	                                              n,
						      nnz,
						      descrA,
						      csrSortedValA,
						      csrSortedRowPtrA,
						      csrSortedColIndA,
						      info,
						      policy,
						      pBuffer 
						      ),
            ROCSOLVER_STATUS_EXECUTION_FAILED );

         HIP_CHECK( hipFree( pBuffer ), ROCSOLVER_STATUS_ALLOC_ERROR );
        /*
	 --------------------
	 check for zero pivot
	 --------------------
         */
	 int pivot = -(n+1);

	 HIPSPARSE_CHECK( hipsparseXcsrilu02_zeroPivot(handle,
	                                               info,
						       &pivot ),
                 ROCSOLVER_STATUS_EXECUTION_FAILED);

         bool isok = (pivot == -1);

         if (!isok) {
            return( ROCSOLVER_STATUS_ZERO_PIVOT );
          };





 return( ROCSOLVER_STATUS_SUCCESS );
}
