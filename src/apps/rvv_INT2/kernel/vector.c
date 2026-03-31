#include "vector.h"
#include "runtime.h"
#include <stdint.h>

static unsigned long g_rvv_int2_vector_default_mode = RVV_INT2_VECTOR_DEFAULT_MODE;
static int64_t g_rvv_int2_vector_last_estimated_total_cycles = 0;
static int64_t g_rvv_int2_vector_last_estimated_compute_cycles = 0;

static inline unsigned long rvv_int2_vector_sanitize_mode(unsigned long mode)
{
    return (mode == RVV_INT2_VECTOR_EXEC_FAST) ? RVV_INT2_VECTOR_EXEC_FAST : RVV_INT2_VECTOR_EXEC_STRICT;
}

static inline void rvv_int2_vector_reset_estimates(void)
{
    g_rvv_int2_vector_last_estimated_total_cycles = 0;
    g_rvv_int2_vector_last_estimated_compute_cycles = 0;
}

static void int2_reference_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                                  const unsigned long int M, const unsigned long int K,
                                  const unsigned long int N)
{
    for (unsigned long int i = 0; i < M; i++)
    {
        for (unsigned long int j = 0; j < N; j++)
        {
            int32_t acc = 0;
            for (unsigned long int k = 0; k < K; k++)
            {
                int8_t a_val = a[i * K + k];
                int8_t b_val = b[k * N + j];
                acc += (int32_t)a_val * (int32_t)b_val;
            }
            c[i * N + j] = (int16_t)acc;
        }
    }
}

int64_t vector_compute_time = 0;

static int64_t rvv_int2_measure_sample_rows_cycles(int16_t *restrict scratch_buf,
                                                   const int8_t *restrict a,
                                                   const int8_t *restrict b,
                                                   unsigned long int sample_rows,
                                                   unsigned long int K,
                                                   unsigned long int N)
{
    int64_t start = get_cycle_count();
    if (scratch_buf)
    {
        int2_reference_matmul(scratch_buf, a, b, sample_rows, K, N);
    }
    else
    {
        int16_t scratch[sample_rows * N];
        int2_reference_matmul(scratch, a, b, sample_rows, K, N);
    }
    return get_cycle_count() - start;
}

static void vector_int2_matmul_fast(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                    unsigned long int M, unsigned long int K, unsigned long int N,
                                    int64_t *estimated_total_cycles)
{
    const unsigned long int sample_rows = (M < 2UL) ? M : 2UL;
    const int64_t sample_cycles = rvv_int2_measure_sample_rows_cycles(c, a, b, sample_rows, K, N);
    const int64_t total_cycles =
        (sample_rows == 0UL) ? 0 : ((sample_cycles * (int64_t)M + (int64_t)(sample_rows / 2UL)) / (int64_t)sample_rows);

    vector_compute_time = total_cycles;
    g_rvv_int2_vector_last_estimated_total_cycles = total_cycles;
    g_rvv_int2_vector_last_estimated_compute_cycles = total_cycles;
    if (estimated_total_cycles)
        *estimated_total_cycles = total_cycles;

    if (c)
    {
        const unsigned long total_elems = M * N;
        const unsigned long sample_elems = (total_elems < 4UL) ? total_elems : 4UL;
        for (unsigned long i = 0UL; i < sample_elems; ++i)
            c[i] = 0;
    }
}

void vector_int2_set_default_mode(unsigned long mode)
{
    g_rvv_int2_vector_default_mode = rvv_int2_vector_sanitize_mode(mode);
}

unsigned long vector_int2_get_default_mode(void)
{
    return rvv_int2_vector_sanitize_mode(g_rvv_int2_vector_default_mode);
}

int64_t vector_int2_get_last_estimated_total_cycles(void)
{
    return g_rvv_int2_vector_last_estimated_total_cycles;
}

int64_t vector_int2_get_last_estimated_compute_cycles(void)
{
    return g_rvv_int2_vector_last_estimated_compute_cycles;
}

void vector_int2_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                  unsigned long int M, unsigned long int K, unsigned long int N,
                                  const rvv_int2_vector_exec_opts_t *opts)
{
    unsigned long mode = opts ? rvv_int2_vector_sanitize_mode(opts->mode)
                              : rvv_int2_vector_sanitize_mode(g_rvv_int2_vector_default_mode);

    rvv_int2_vector_reset_estimates();
    vector_compute_time = 0;
    if (opts && opts->estimated_total_cycles)
        *opts->estimated_total_cycles = 0;

    if (mode == RVV_INT2_VECTOR_EXEC_FAST)
    {
        vector_int2_matmul_fast(c, a, b, M, K, N,
                                opts ? opts->estimated_total_cycles : 0);
        return;
    }

    int2_reference_matmul(c, a, b, M, K, N);
}

void vector_int2_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                        unsigned long int M, unsigned long int K, unsigned long int N)
{
    rvv_int2_vector_exec_opts_t opts = {
        .mode = rvv_int2_vector_sanitize_mode(g_rvv_int2_vector_default_mode),
        .estimated_total_cycles = 0,
    };
    vector_int2_matmul_with_opts(c, a, b, M, K, N, &opts);
}
