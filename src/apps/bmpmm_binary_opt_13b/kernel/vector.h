#ifndef _BMPMM_VECTOR_H_
#define _BMPMM_VECTOR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // RVV vector-based int8xint8->int16 matrix multiplication
    void vector_int8_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                            unsigned long int M, unsigned long int K, unsigned long int N);

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
