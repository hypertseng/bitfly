#ifndef BMPMM_INT2_MIXED_H
#define BMPMM_INT2_MIXED_H

#include <stdint.h>
#include "../../common/bmpmm_bench_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void int2_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                           unsigned long M, unsigned long K, unsigned long N);
    int int2_mixed_matmul_with_cfg(int16_t *c, const int8_t *a, const int8_t *b,
                                   unsigned long M, unsigned long K, unsigned long N,
                                   const bmpmm_exec_cfg_t *exec_cfg);
    extern int64_t mixed_compute_time;

#ifdef __cplusplus
}
#endif

#endif
