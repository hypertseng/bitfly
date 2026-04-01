#include "rvv_compare.h"

#define vector_int8_matmul llama_rvv_binary_vector_int8_matmul
#define vector_int8_matmul_with_opts llama_rvv_binary_vector_int8_matmul_with_opts
#define vector_int8_set_default_mode llama_rvv_binary_vector_int8_set_default_mode
#define vector_int8_get_default_mode llama_rvv_binary_vector_int8_get_default_mode
#define vector_int8_get_last_estimated_total_cycles llama_rvv_binary_vector_int8_get_last_estimated_total_cycles
#define vector_int8_get_last_estimated_compute_cycles llama_rvv_binary_vector_int8_get_last_estimated_compute_cycles
#define matmul_vec_slice_init llama_rvv_binary_matmul_vec_slice_init
#define matmul_vec llama_rvv_binary_matmul_vec
#define vector_compute_time llama_rvv_binary_vector_compute_time
#include "../../rvv_binary/kernel/vector.c"
#undef vector_int8_matmul
#undef vector_int8_matmul_with_opts
#undef vector_int8_set_default_mode
#undef vector_int8_get_default_mode
#undef vector_int8_get_last_estimated_total_cycles
#undef vector_int8_get_last_estimated_compute_cycles
#undef matmul_vec_slice_init
#undef matmul_vec
#undef vector_compute_time

static int64_t g_llama_rvv_last_estimated_total_cycles = 0;
static int64_t g_llama_rvv_last_estimated_compute_cycles = 0;

static unsigned long llama_rvv_canonical_prec(unsigned long prec)
{
    (void)prec;
    return LLAMA2_RVV_PREC_INT8;
}

const char *llama_rvv_prec_name(unsigned long prec)
{
    (void)prec;
    return "INT8";
}

int llama_rvv_matmul_with_cfg_opts(int16_t *c, const int8_t *a, const int8_t *b,
                                   unsigned long M, unsigned long K, unsigned long N,
                                   const llama_rvv_exec_cfg_t *exec_cfg,
                                   const llama_rvv_exec_opts_t *opts)
{
    const unsigned long prec = exec_cfg ? llama_rvv_canonical_prec(exec_cfg->prec) : ~0UL;

    g_llama_rvv_last_estimated_total_cycles = 0;
    g_llama_rvv_last_estimated_compute_cycles = 0;
    if (opts && opts->estimated_total_cycles)
        *opts->estimated_total_cycles = 0;

    if (prec != LLAMA2_RVV_PREC_INT8)
        return 0;

    {
        rvv_binary_vector_exec_opts_t int8_opts = {
            .mode = opts ? opts->mode : LLAMA2_RVV_EXEC_STRICT,
            .estimated_total_cycles = opts ? opts->estimated_total_cycles : 0,
        };
        llama_rvv_binary_vector_int8_matmul_with_opts(c, a, b, M, K, N, &int8_opts);
        g_llama_rvv_last_estimated_total_cycles = llama_rvv_binary_vector_int8_get_last_estimated_total_cycles();
        g_llama_rvv_last_estimated_compute_cycles = llama_rvv_binary_vector_int8_get_last_estimated_compute_cycles();
        return 1;
    }
}

int64_t llama_rvv_get_last_estimated_total_cycles(void)
{
    return g_llama_rvv_last_estimated_total_cycles;
}

int64_t llama_rvv_get_last_estimated_compute_cycles(void)
{
    return g_llama_rvv_last_estimated_compute_cycles;
}
