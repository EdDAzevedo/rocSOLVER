/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GETRF_NPVT_HPP
#define GETRF_NPVT_HPP

#include "gbtr_common.h"

template <typename T, typename I>
DEVICE_FUNCTION void getrf_npvt_device(I const m, I const n, T* A_, I const lda, I* pinfo)
{
#define A(ia, ja) A_[indx2f(ia, ja, lda)]
    /*
!     ----------------------------------------
!     Perform LU factorization without pivoting
!     Matrices L and U over-writes matrix A
!     ----------------------------------------
*/

    I const min_mn = (m < n) ? m : n;
    I info = 0;

    /*
! 
! % ----------------------------------------------------------
! % note in actual code, L and U over-writes original matrix A
! % ----------------------------------------------------------
! for j=1:min_mn,
!   jp1 = j + 1;
! 
!   U(j,j) = A(j,j);
!   L(j,j) = 1;
! 
!   L(jp1:m,j) = A(jp1:m,j) / U(j,j);
!   U(j,jp1:n) = A(j,jp1:n);
! 
!   A(jp1:m,jp1:n) = A(jp1:m, jp1:n) - L(jp1:m,j) * U(j, jp1:n);
! end;
*/

    T const zero = 0;
    T const one = 1;

    for(I j = 1; j <= min_mn; j++)
    {
        I const jp1 = j + 1;
        bool const is_diag_zero = (A(j, j) == zero);

        T const inv_Ujj = (is_diag_zero) ? one : one / A(j, j);
        info = (is_diag_zero) && (info == 0) ? j : info;

        /*
!        ---------------------------------
!        A(jp1:m,j) = A(jp1:m,j) * inv_Ujj
!        ---------------------------------
*/
        for(I ia = jp1; ia <= m; ia++)
        {
            A(ia, j) *= inv_Ujj;
        };

        for(I ja = jp1; ja <= n; ja++)
        {
            for(I ia = jp1; ia <= m; ia++)
            {
                A(ia, ja) = A(ia, ja) - A(ia, j) * A(j, ja);
            };
        };
    };

    if(info != 0)
    {
        *pinfo = info;
    };

    return;
};

#undef A
#endif
