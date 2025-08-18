#include "rocblas/rocblas.h"

#include "../../clients/samples/gpubuf.h"
#include "auxiliary/rocauxiliary_gemm_ex.hpp"

#include <stdexcept>
#include <vector>

using namespace rocsolver;

int main()
{
    const rocblas_int m = 100;
    const rocblas_int n = 100;
    const rocblas_int k = 100;

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

    std::vector<rocblas_float_complex> A_host(m * k);
    std::vector<rocblas_float_complex> B_host(k * n);
    std::vector<rocblas_float_complex> C_host(m * n);
    std::vector<rocblas_float_complex> D_host_ref(m * n);
    std::vector<rocblas_float_complex> D_host_ex(m * n);
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

                                  m, n, k,

                                  &alpha_complex,

                                  A.data(), rocblas_datatype_f32_c, k,

                                  B.data(), rocblas_datatype_f32_c, n,

                                  &beta_complex,

                                  C.data(), rocblas_datatype_f32_c, n,

                                  D_ref.data(), rocblas_datatype_f32_c, n,

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

        m, n, k,

        &alpha_real,

        A.data(), rocblas_datatype_f16_r, k, 0,

        B.data(), rocblas_datatype_f32_c, n, 0,

        &beta_real,

        C.data(), rocblas_datatype_f32_c, n, 0,

        D_ex.data(), rocblas_datatype_f32_c, n, 0,

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

    return 0;
}
