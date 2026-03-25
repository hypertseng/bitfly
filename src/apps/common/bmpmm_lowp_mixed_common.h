#ifndef BMPMM_LOWP_MIXED_COMMON_H
#define BMPMM_LOWP_MIXED_COMMON_H

#include <stdint.h>
#include "bmpmm_bench_common.h"

int bmpmm_lowp_mixed_matmul_with_cfg(const char *app_tag,
                                     int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long M, unsigned long K, unsigned long N,
                                     const bmpmm_exec_cfg_t *exec_cfg,
                                     int64_t *compute_cycles);

#endif
