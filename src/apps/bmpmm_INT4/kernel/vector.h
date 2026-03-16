#ifndef BMPMM_INT4_VECTOR_H
#define BMPMM_INT4_VECTOR_H

#include <stdint.h>

void vector_int4_matmul(int16_t *restrict c, const int8_t *restrict a,
                        const int8_t *restrict b,
                        unsigned long int M, unsigned long int K,
                        unsigned long int N);

#endif // BMPMM_INT4_VECTOR_H
