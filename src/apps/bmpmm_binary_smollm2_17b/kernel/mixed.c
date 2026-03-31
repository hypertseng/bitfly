#include "mixed.h"
#include "../../common/bmpmm_lowp_mixed_common.h"

static void binary_reference_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                                    unsigned long M, unsigned long K, unsigned long N)
{
    for (unsigned long i = 0; i < M; ++i)
        for (unsigned long j = 0; j < N; ++j)
        {
            int32_t acc = 0;
            for (unsigned long k = 0; k < K; ++k)
                acc += (int32_t)a[i * K + k] * (int32_t)b[k * N + j];
            c[i * N + j] = (int16_t)acc;
        }
}

int64_t mixed_compute_time = 0;

void binary_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                         unsigned long M, unsigned long K, unsigned long N)
{
    binary_reference_matmul(c, a, b, M, K, N);
}

int binary_mixed_matmul_with_cfg(int16_t *c, const int8_t *a, const int8_t *b,
                                 unsigned long M, unsigned long K, unsigned long N,
                                 const binary_exec_cfg_t *exec_cfg)
{
    int64_t local_compute = 0;
    int ok = bmpmm_lowp_mixed_matmul_with_cfg("bmpmm_binary", c, a, b, M, K, N, exec_cfg, &local_compute);
    mixed_compute_time += local_compute;
    return ok;
}
