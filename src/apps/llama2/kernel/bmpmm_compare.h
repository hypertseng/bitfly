#ifndef LLAMA2_BMPMM_COMPARE_H
#define LLAMA2_BMPMM_COMPARE_H

#include <stdint.h>
#include "../../common/bmpmm_bench_common.h"
#include "../../common/bmpmm_lowp_mixed_common.h"

typedef bmpmm_exec_cfg_t llama_bmpmm_exec_cfg_t;
typedef bmpmm_lowp_exec_opts_t llama_bmpmm_exec_opts_t;

int llama_bmpmm_matmul_with_cfg_opts(int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long M, unsigned long K, unsigned long N,
                                     const llama_bmpmm_exec_cfg_t *exec_cfg,
                                     const llama_bmpmm_exec_opts_t *opts);

int64_t llama_bmpmm_get_last_estimated_total_cycles(void);
int64_t llama_bmpmm_get_last_estimated_compute_cycles(void);

#endif
