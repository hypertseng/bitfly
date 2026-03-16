#ifndef _BMPMM_MIXED_H_
#define _BMPMM_MIXED_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        unsigned long int mtile;
        unsigned long int ntile;
        unsigned long int ktile;
        unsigned long int gm;
        unsigned long int gn;
        unsigned long int prec;
    } binary_exec_cfg_t;

    // Execute directly with an externally provided tiling/reuse configuration.
    // This is intended for running pre-searched parameter tuples.
    int binary_mixed_matmul_with_cfg(int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long int M, unsigned long int K,
                                     unsigned long int N,
                                     const binary_exec_cfg_t *exec_cfg);

    // debug / profiling counter
    extern int64_t mixed_compute_time;

#ifdef __cplusplus
}
#endif

#endif // _BMPMM_MIXED_H_
