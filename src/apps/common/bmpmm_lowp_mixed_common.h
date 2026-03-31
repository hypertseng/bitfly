#ifndef BMPMM_LOWP_MIXED_COMMON_H
#define BMPMM_LOWP_MIXED_COMMON_H

#include <stdint.h>
#include "bmpmm_bench_common.h"

#define BMPMM_LOWP_EXEC_STRICT 0UL
#define BMPMM_LOWP_EXEC_FAST 1UL

#ifndef BMPMM_LOWP_DEFAULT_MODE
#define BMPMM_LOWP_DEFAULT_MODE BMPMM_LOWP_EXEC_STRICT
#endif

typedef struct
{
    // `fast` only estimates runtime from sampled inner-kernel execution.
    // It does not materialize a valid full output matrix.
    unsigned long mode;
    int64_t *estimated_total_cycles;
} bmpmm_lowp_exec_opts_t;

int bmpmm_lowp_mixed_matmul_with_cfg(const char *app_tag,
                                     int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long M, unsigned long K, unsigned long N,
                                     const bmpmm_exec_cfg_t *exec_cfg,
                                     int64_t *compute_cycles);

int bmpmm_lowp_mixed_matmul_with_cfg_opts(const char *app_tag,
                                          int16_t *c, const int8_t *a, const int8_t *b,
                                          unsigned long M, unsigned long K, unsigned long N,
                                          const bmpmm_exec_cfg_t *exec_cfg,
                                          int64_t *compute_cycles,
                                          const bmpmm_lowp_exec_opts_t *opts);

void bmpmm_lowp_set_default_mode(unsigned long mode);
unsigned long bmpmm_lowp_get_default_mode(void);
int64_t bmpmm_lowp_get_last_estimated_total_cycles(void);
int64_t bmpmm_lowp_get_last_estimated_compute_cycles(void);

#endif
