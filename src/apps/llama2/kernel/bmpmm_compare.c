#include "bmpmm_compare.h"

int llama_bmpmm_matmul_with_cfg_opts(int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long M, unsigned long K, unsigned long N,
                                     const llama_bmpmm_exec_cfg_t *exec_cfg,
                                     const llama_bmpmm_exec_opts_t *opts)
{
    int64_t local_compute = 0;
    return bmpmm_lowp_mixed_matmul_with_cfg_opts("llama2", c, a, b, M, K, N, exec_cfg, &local_compute, opts);
}

int64_t llama_bmpmm_get_last_estimated_total_cycles(void)
{
    return bmpmm_lowp_get_last_estimated_total_cycles();
}

int64_t llama_bmpmm_get_last_estimated_compute_cycles(void)
{
    return bmpmm_lowp_get_last_estimated_compute_cycles();
}
