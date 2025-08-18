#include "rocblas/rocblas.h"

#include "../../clients/samples/gpubuf.h"
#include "auxiliary/rocauxiliary_gemm_ex.hpp"

#include <stdexcept>
#include <vector>

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
using namespace rocsolver;

template <typename Tfull, typename Treduced, typename I, typename Istride>
void test_gemm_ex(rocblas_handle handle, I const m, I const n, I const k)
{
    bool constexpr is_complex = rocblas_is_complex<Tfull>;

    using Sf = decltype(std::real(Tfull{}));
    using Sr = decltype(std::real(Treduced{}));

    bool constexpr is_fp16 = std::is_same<Sr, rocblas_half>::value;
    Sf const dlimit = (is_fp16) ? 65504 : std::numeric_limits<Sf>::max();

    auto clamp = [](auto aij, auto amin, auto amax) {
        return ((aij < amin) ? amin : (aij > amax) ? amax : aij);
    };

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const nrows_A = m;
    I const ncols_A = k;
    I const nrows_B = k;
    I const ncols_B = n;
    I const nrows_C = m;
    I const ncols_C = n;

    I const lda = nrows_A;
    I const ldb = nrows_B;
    I const ldc = nrows_C;

    rocblas_operation const trans_A = rocblas_operation_none;
    rocblas_operation const trans_B = rocblas_operation_none;

    thrust::host_vector<Tfull> h_A(lda * ncols_A);
    thrust::host_vector<Tfull> h_B(ldb * ncols_B);
    thrust::host_vector<Tfull> h_C(ldc * ncols_C);

    I const batch_count = 1;

#pragma omp parallel for
    for(I j = 0; j < ncols_A; j++)
    {
        for(I i = 0; i < nrows_A; i++)
        {
            auto const ij = idx2D(i, j, lda);

            Sf const aij_re = (i + 1.0) / nrows_A;
            if constexpr(is_complex)
            {
                Sf const aij_im = (j + 1.0) / ncols_A;
                Tfull aij{aij_re, aij_im};
                h_A[ij] = aij;
            }
            else
            {
                h_A[ij] = aij_re;
            }
        }
    }

#pragma omp parallel for
    for(I j = 0; j < ncols_B; j++)
    {
        for(I i = 0; i < nrows_B; i++)
        {
            auto const ij = idx2D(i, j, ldb);

            Sf const bij_re = i + 1.0;
            if constexpr(is_complex)
            {
                Sf const bij_im = j + 1.0;
                Tfull bij{bij_re, bij_im};
                h_B[ij] = bij;
            }
            else
            {
                h_B[ij] = bij_re;
            }
        }
    }

#pragma omp parallel for
    for(I j = 0; j < ncols_C; j++)
    {
        for(I i = 0; i < nrows_C; i++)
        {
            auto const ij = idx2D(i, j, ldc);
            if constexpr(is_complex)
            {
                Tfull const cij{1.5, 2.0};
                h_C[ij] = cij;
            }
            else
            {
                h_C[ij] = 1.5;
            }
        }
    }

    thrust::device_vector<Tfull> d_A{h_A};
    thrust::device_vector<Tfull> d_B{h_B};
    thrust::device_vector<Tfull> d_C{h_C};

    thrust::host_vector<Tfull> h_alpha(1);
    h_alpha[0] = 1.0;
    thrust::host_vector<Tfull> h_beta(1);
    h_beta[0] = 1.0;

    thrust::device_vector<Tfull> d_alpha{h_alpha};
    thrust::device_vector<Tfull> d_beta{h_beta};

    // ------------------
    // compute on CPU host
    // ------------------

#pragma omp parallel for
    for(I j = 0; j < ncols_C; j++)
    {
        for(I i = 0; i < nrows_C; i++)
        {
            // --------------------------------------------
            // compute just (i,j) entry of A_chop * B_chop
            // --------------------------------------------
            Sf cij_re = 0;
            Sf cij_im = 0;

            Sr const zero = static_cast<Sr>(0.0);

            for(I k = 0; k < ncols_A; k++)
            {
                auto const aik = h_A[idx2D(i, k, lda)];
                auto const bkj = h_B[idx2D(k, j, ldb)];

                // -------------------------
                // chop to reduced precision
                // -------------------------
                Sr const aik_re_chop = static_cast<Sr>(clamp(std::real(aik), -dlimit, dlimit));
                Sr const aik_im_chop
                    = (is_complex) ? static_cast<Sr>(clamp(std::imag(aik), -dlimit, dlimit)) : zero;

                Sr const bkj_re_chop
                    = (is_complex) ? static_cast<Sr>(clamp(std::real(bkj), -dlimit, dlimit)) : zero;
                Sr const bkj_im_chop
                    = (is_complex) ? static_cast<Sr>(clamp(std::imag(bkj), -dlimit, dlimit)) : zero;

                // ----------------------------------
                // arithmetic calculations using FP32
                // ----------------------------------
                Sf const a_re = aik_re_chop;
                Sf const a_im = aik_im_chop;
                Sf const b_re = bkj_re_chop;
                Sf const b_im = bkj_im_chop;

                // (c_re + J * c_im) = (c_re + J * c_im) + (a_re + J * a_im) * (b_re + J * b_im)
                cij_re += (a_re * b_re - a_im * b_im);
                cij_im += (a_re * b_im + a_im * b_re);
            }

            auto const ij_c = idx2D(i, j, ldc);
            h_C[ij_c] *= h_beta[0];
            if constexpr(is_complex)
            {
                h_C[ij_c] += h_alpha[0] * Tfull{cij_re, cij_im};
            }
            else
            {
                h_C[ij_c] += h_alpha[0] * cij_re;
            }
        }
    }

    // ---------------------------
    // perform computations on GPU
    // ---------------------------
    {
        size_t size_work = 0;

        rocblasCall_gemm_strided_batched_ex_getMemorySize<Tfull, Treduced>(trans_A, trans_B, m, n, k,
                                                                           batch_count, &size_work);

        thrust::device_vector<std::byte> d_work(size_work);

        Istride const stride_A = Istride(lda) * ncols_A;
        Istride const stride_B = Istride(ldb) * ncols_B;
        Istride const stride_C = Istride(ldc) * ncols_C;

        rocblas_datatype const type_C = rocblas_datatype_from_type<Tfull>;
        rocblas_datatype const type_A = type_C;
        rocblas_datatype const type_B = type_C;

        rocblas_datatype const compute_type = rocblas_datatype_from_type<Sr>;

        int32_t const solution_index = 0;
        uint32_t const flags = 0;
        auto const istat = rocblasCall_gemm_strided_batched_ex(
            handle, trans_A, trans_B, m, n, k,

            h_alpha.data(),

            d_A.data().get(), type_A, lda, stride_A,

            d_B.data().get(), type_B, ldb, stride_B,

            h_beta.data(),

            d_C.data().get(), type_C, ldc, stride_C,

            d_C.data().get(), type_C, ldc, stride_C, // D is C

            batch_count,

            compute_type, rocblas_gemm_algo_standard, solution_index, flags,

            d_work.data().get(), size_work);
        assert(istat == rocblas_status_success);
    }

    // -------------
    // check results
    // -------------
    {
        rocblas_datatype const type_C = rocblas_datatype_from_type<Tfull>;
        rocblas_datatype const compute_type = rocblas_datatype_from_type<Sr>;

        thrust::host_vector<Tfull> h_C_gpu{d_C};

        double err_max = 0;
        double err_L2 = 0;

#pragma omp parallel for collapse(2) reduction(max : err_max) reduction(+ : err_L2)
        for(I j = 0; j < ncols_C; j++)
        {
            for(I i = 0; i < nrows_C; i++)
            {
                auto const ij_c = idx2D(i, j, ldc);
                auto const cij_cpu = h_C[ij_c];
                auto const cij_gpu = h_C_gpu[ij_c];
                double const abserr = std::abs(cij_cpu - cij_gpu);

                err_max = std::max(err_max, abserr);
                err_L2 += abserr * abserr;
            }
        }
        err_L2 = std::sqrt(err_L2);
        printf("m = %d, n=%d, k=%d, type_C = %d, compute_type = %d\n", m, n, k, (int)type_C,
               (int)compute_type);
        printf("err_max = %le, err_L2 = %le \n", err_max, err_L2);
    }
}

int main()
{
    const rocblas_int m = 17504;
    const rocblas_int n = 17504;
    const rocblas_int k = 32;

    using Tfull = rocblas_float_complex;
    using Treduced = rocblas_half;

    size_t work_size = 0;
    rocblasCall_gemm_strided_batched_ex_getMemorySize<Tfull, Treduced>(
        rocblas_operation_none, rocblas_operation_none, m, n, k, 1, &work_size);

    printf("work size %zu\n", work_size);

    gpubuf_t workmem;
    if(workmem.alloc(work_size) != hipSuccess)
        throw std::runtime_error("failed to alloc work");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    rocblas_float_complex* ptr = nullptr;

    std::vector<rocblas_float_complex> A_host(16);
    std::vector<rocblas_float_complex> B_host(16);
    std::vector<rocblas_float_complex> C_host(16);
    std::vector<rocblas_float_complex> D_host_ref(16);
    std::vector<rocblas_float_complex> D_host_ex(16);
    std::fill(A_host.begin(), A_host.end(), rocblas_float_complex{0.25, 0.5});
    std::fill(B_host.begin(), B_host.end(), rocblas_float_complex{-0.25, 0.5});
    std::fill(C_host.begin(), C_host.end(), rocblas_float_complex{1.5, 2.0});

    gpubuf_t<rocblas_float_complex> A;
    size_t A_bytes = A_host.size() * sizeof(rocblas_float_complex);
    if(A.alloc(A_bytes) != hipSuccess)
        throw std::runtime_error("failed to alloc A");

    gpubuf_t<rocblas_float_complex> B;
    size_t B_bytes = B_host.size() * sizeof(rocblas_float_complex);
    if(B.alloc(B_bytes) != hipSuccess)
        throw std::runtime_error("failed to alloc B");

    gpubuf_t<rocblas_float_complex> C;
    size_t C_bytes = C_host.size() * sizeof(rocblas_float_complex);
    if(C.alloc(C_bytes) != hipSuccess)
        throw std::runtime_error("failed to alloc C");

    gpubuf_t<rocblas_float_complex> D_ref;
    if(D_ref.alloc(C_bytes) != hipSuccess)
        throw std::runtime_error("failed to alloc D_ref");
    if(hipMemset(D_ref.data(), 0, C_bytes) != hipSuccess)
        throw std::runtime_error("failed to memset D_ref");

    gpubuf_t<rocblas_float_complex> D_ex;
    if(D_ex.alloc(C_bytes) != hipSuccess)
        throw std::runtime_error("failed to alloc D_ex");
    if(hipMemset(D_ex.data(), 0, C_bytes) != hipSuccess)
        throw std::runtime_error("failed to memset D_ex");

    if(hipMemcpy(A.data(), A_host.data(), A_bytes, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("failed to memcpy A");
    if(hipMemcpy(B.data(), B_host.data(), B_bytes, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("failed to memcpy B");
    if(hipMemcpy(C.data(), C_host.data(), C_bytes, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("failed to memcpy C");

    rocblas_float_complex alpha_complex{1.0, 0.0};
    rocblas_float_complex beta_complex{1.0, 0.0};
    rocblas_half alpha_real{1.0};
    rocblas_half beta_real{1.0};

    auto status = rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,

                                  4, 4, 4,

                                  &alpha_complex,

                                  A.data(), rocblas_datatype_f32_c, 4,

                                  B.data(), rocblas_datatype_f32_c, 4,

                                  &beta_complex,

                                  C.data(), rocblas_datatype_f32_c, 4,

                                  D_ref.data(), rocblas_datatype_f32_c, 4,

                                  rocblas_datatype_f32_c, rocblas_gemm_algo_standard, 0, 0);

    printf("status ref : %d\n", static_cast<int>(status));

    if(hipMemcpy(D_host_ref.data(), D_ref.data(), C_bytes, hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("failed to copy D_ref back");

    for(auto elem : D_host_ref)
    {
        printf("(%f, %f) ", static_cast<double>(elem.real()), static_cast<double>(elem.imag()));
    }
    puts("");

    status = rocblasCall_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none,

        4, 4, 4,

        &alpha_real,

        A.data(), rocblas_datatype_f16_r, 4, 0,

        B.data(), rocblas_datatype_f32_c, 4, 0,

        &beta_real,

        C.data(), rocblas_datatype_f32_c, 4, 0,

        D_ex.data(), rocblas_datatype_f32_c, 4, 0,

        1,

        rocblas_datatype_f32_c, rocblas_gemm_algo_standard, 0, 0,

        workmem.data(), work_size);

    printf("status ex : %d\n", static_cast<int>(status));

    if(hipMemcpy(D_host_ex.data(), D_ex.data(), C_bytes, hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("failed to copy D_ex back");

    for(auto elem : D_host_ex)
    {
        printf("(%f, %f) ", static_cast<double>(elem.real()), static_cast<double>(elem.imag()));
    }
    puts("");

    {
        auto const m = 4;
        auto const n = 4;
        auto const k = 4;
        test_gemm_ex<rocblas_float_complex, rocblas_bfloat16, rocblas_int, rocblas_stride>(handle,
                                                                                           m, n, k);

        test_gemm_ex<float, rocblas_bfloat16, rocblas_int, rocblas_stride>(handle, m, n, k);

        test_gemm_ex<rocblas_float_complex, rocblas_half, rocblas_int, rocblas_stride>(handle, m, n,
                                                                                       k);

        test_gemm_ex<float, rocblas_half, rocblas_int, rocblas_stride>(handle, m, n, k);
    }

    return 0;
}
