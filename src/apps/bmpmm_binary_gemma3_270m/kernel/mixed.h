#ifndef _BMPMM_MIXED_H_
#define _BMPMM_MIXED_H_

#include <stdint.h>
#include "../../common/bmpmm_bench_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef bmpmm_exec_cfg_t binary_exec_cfg_t;

    extern int64_t mixed_compute_time;

    void binary_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                             unsigned long int M, unsigned long int K,
                             unsigned long int N);

    int binary_mixed_matmul_with_cfg(int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long int M, unsigned long int K,
                                     unsigned long int N,
                                     const binary_exec_cfg_t *exec_cfg);

#ifdef __cplusplus
}
#endif

#endif // _BMPMM_MIXED_H_
