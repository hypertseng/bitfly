#ifndef BMPMM_INT2_VECTOR_H
#define BMPMM_INT2_VECTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void vector_int2_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                            unsigned long int M, unsigned long int K, unsigned long int N);

    extern int64_t vector_compute_time;

#ifdef __cplusplus
}
#endif

#endif // BMPMM_INT2_VECTOR_H
