#ifndef LLAMA2_RVV_COMPARE_H
#define LLAMA2_RVV_COMPARE_H

#include <stdint.h>

#define LLAMA2_RVV_EXEC_STRICT 0UL
#define LLAMA2_RVV_EXEC_FAST 1UL

#define LLAMA2_RVV_PREC_INT8 1UL
#define LLAMA2_RVV_PREC_BINARY 0UL
#define LLAMA2_RVV_PREC_INT2 2UL
#define LLAMA2_RVV_PREC_INT4 3UL

typedef struct
{
    unsigned long mtile;
    unsigned long ntile;
    unsigned long ktile;
    unsigned long gm;
    unsigned long gn;
    unsigned long prec;
} llama_rvv_exec_cfg_t;

typedef struct
{
    unsigned long mode;
    int64_t *estimated_total_cycles;
} llama_rvv_exec_opts_t;

int llama_rvv_matmul_with_cfg_opts(int16_t *c, const int8_t *a, const int8_t *b,
                                   unsigned long M, unsigned long K, unsigned long N,
                                   const llama_rvv_exec_cfg_t *exec_cfg,
                                   const llama_rvv_exec_opts_t *opts);

int64_t llama_rvv_get_last_estimated_total_cycles(void);
int64_t llama_rvv_get_last_estimated_compute_cycles(void);
const char *llama_rvv_prec_name(unsigned long prec);

#endif
