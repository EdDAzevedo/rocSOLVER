#include "rocblas/rocblas.h"

#include "../../clients/samples/gpubuf.h"
#include "auxiliary/rocauxiliary_gemm_ex.hpp"

#include <stdexcept>

using namespace rocsolver;

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

    auto status = rocblasCall_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none,

        m, n, k,

        ptr,

        ptr, rocblas_datatype_f32_c, n, 0,

        ptr, rocblas_datatype_f32_c, m, 0,

        ptr,

        ptr, rocblas_datatype_f32_c, n, 0,

        ptr, rocblas_datatype_f32_c, n, 0,

        1,

        rocblas_datatype_f16_r, rocblas_gemm_algo_standard, 0, 0,

        workmem.data(), work_size);

    printf("status: %d\n", static_cast<int>(status));

    return 0;
}
