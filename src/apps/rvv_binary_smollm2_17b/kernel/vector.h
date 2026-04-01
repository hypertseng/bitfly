#ifndef _BMPMM_VECTOR_H_
#define _BMPMM_VECTOR_H_

#include <stdint.h>

#define RVV_BINARY_VECTOR_EXEC_STRICT 0UL
#define RVV_BINARY_VECTOR_EXEC_FAST 1UL

#ifndef RVV_BINARY_VECTOR_DEFAULT_MODE
#define RVV_BINARY_VECTOR_DEFAULT_MODE RVV_BINARY_VECTOR_EXEC_STRICT
#endif

typedef struct
{
    // `fast` estimates runtime from sampled RVV block execution.
    // It does not materialize a valid full output matrix.
    unsigned long mode;
    int64_t *estimated_total_cycles;
} rvv_binary_vector_exec_opts_t;

#ifdef __cplusplus
extern "C"
{
#endif

    // RVV vector-based int8xint8->int16 matrix multiplication
    void vector_int8_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                            unsigned long int M, unsigned long int K, unsigned long int N);
    void vector_int8_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                      unsigned long int M, unsigned long int K, unsigned long int N,
                                      const rvv_binary_vector_exec_opts_t *opts);
    void vector_int8_set_default_mode(unsigned long mode);
    unsigned long vector_int8_get_default_mode(void);
    int64_t vector_int8_get_last_estimated_total_cycles(void);
    int64_t vector_int8_get_last_estimated_compute_cycles(void);

    // low‑level helpers used by the vector kernel
    void matmul_vec_slice_init(void);
    void matmul_vec(int16_t *c, const int8_t *a, const int8_t *b,
                    const unsigned long int K, const unsigned long int N);

    // profiling counter
    extern int64_t vector_compute_time;

#ifdef __cplusplus
}
#endif

#endif // _BMPMM_VECTOR_H_
