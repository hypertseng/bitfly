#ifndef _BMPMM_MIXED_H_
#define _BMPMM_MIXED_H_

#include <stdint.h>

#define RVV_BINARY_EXEC_STRICT 0UL
#define RVV_BINARY_EXEC_FAST 1UL

#ifndef RVV_BINARY_DEFAULT_MODE
#define RVV_BINARY_DEFAULT_MODE RVV_BINARY_EXEC_STRICT
#endif

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

    typedef struct
    {
        // `fast` only estimates runtime from sampled inner-kernel execution.
        // It does not materialize a valid full output matrix.
        unsigned long int mode;
        int64_t *estimated_total_cycles;
    } binary_exec_opts_t;

    // Execute directly with an externally provided tiling/reuse configuration.
    // This is intended for running pre-searched parameter tuples.
    int binary_mixed_matmul_with_cfg(int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long int M, unsigned long int K,
                                     unsigned long int N,
                                     const binary_exec_cfg_t *exec_cfg);

    int binary_mixed_matmul_with_cfg_opts(int16_t *c, const int8_t *a, const int8_t *b,
                                          unsigned long int M, unsigned long int K,
                                          unsigned long int N,
                                          const binary_exec_cfg_t *exec_cfg,
                                          const binary_exec_opts_t *opts);

    void binary_set_default_mode(unsigned long int mode);
    unsigned long int binary_get_default_mode(void);
    int64_t binary_get_last_estimated_total_cycles(void);
    int64_t binary_get_last_estimated_compute_cycles(void);


#ifdef __cplusplus
}
#endif

#endif // _BMPMM_MIXED_H_
