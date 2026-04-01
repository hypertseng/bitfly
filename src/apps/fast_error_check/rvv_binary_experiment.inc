#define vector_compute_time rvv_binary_vector_compute_time
#define vector_int8_matmul rvv_binary_vector_int8_matmul_prod
#define vector_int8_matmul_with_opts rvv_binary_vector_int8_matmul_with_opts_prod
#define vector_int8_set_default_mode rvv_binary_vector_set_default_mode_prod
#define vector_int8_get_default_mode rvv_binary_vector_get_default_mode_prod
#define vector_int8_get_last_estimated_total_cycles rvv_binary_vector_get_last_estimated_total_cycles_prod
#define vector_int8_get_last_estimated_compute_cycles rvv_binary_vector_get_last_estimated_compute_cycles_prod
#define rvv_binary_vector_sanitize_mode rvv_binary_vector_sanitize_mode_prod
#define rvv_binary_vector_reset_estimates rvv_binary_vector_reset_estimates_prod
#define vector_int8_matmul_strict rvv_binary_vector_int8_matmul_strict_prod
#define rvv_binary_measure_block rvv_binary_measure_block_prod
#define rvv_binary_measure_block_probe rvv_binary_measure_block_probe_prod
#define rvv_binary_measure_block_sampled rvv_binary_measure_block_sampled_prod
#define rvv_binary_probe_matmul_vec rvv_binary_probe_matmul_vec_prod
#define rvv_binary_linear_estimate rvv_binary_linear_estimate_prod
#define rvv_binary_scale_cycles_by_cols rvv_binary_scale_cycles_by_cols_prod
#define int8_reference_matmul_window int8_reference_matmul_window_prod
#define vector_int8_matmul_fast rvv_binary_vector_int8_matmul_fast_prod
#define matmul_vec_slice_init rvv_binary_matmul_vec_slice_init_prod
#define matmul_vec rvv_binary_matmul_vec_prod

#include "../rvv_binary/kernel/vector.c"

void rvv_binary_vector_set_default_mode(unsigned long mode)
{
    rvv_binary_vector_set_default_mode_prod(mode);
}

unsigned long rvv_binary_vector_get_default_mode(void)
{
    return rvv_binary_vector_get_default_mode_prod();
}

int64_t rvv_binary_vector_get_last_estimated_total_cycles(void)
{
    return rvv_binary_vector_get_last_estimated_total_cycles_prod();
}

int64_t rvv_binary_vector_get_last_estimated_compute_cycles(void)
{
    return rvv_binary_vector_get_last_estimated_compute_cycles_prod();
}

void rvv_binary_vector_int8_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                             unsigned long int M, unsigned long int K, unsigned long int N,
                                             const rvv_binary_vector_exec_opts_t *opts)
{
    rvv_binary_vector_int8_matmul_with_opts_prod(c, a, b, M, K, N, opts);
}

void rvv_binary_vector_int8_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                   unsigned long int M, unsigned long int K, unsigned long int N)
{
    rvv_binary_vector_int8_matmul_prod(c, a, b, M, K, N);
}
