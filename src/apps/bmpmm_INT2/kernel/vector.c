#include "vector.h"
#include <stdint.h>

static void int2_reference_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                                  const unsigned long int M, const unsigned long int K,
                                  const unsigned long int N)
{
    for (unsigned long int i = 0; i < M; i++)
    {
        for (unsigned long int j = 0; j < N; j++)
        {
            int32_t acc = 0;
            for (unsigned long int k = 0; k < K; k++)
            {
                int8_t a_val = a[i * K + k];
                int8_t b_val = b[k * N + j];
                acc += (int32_t)a_val * (int32_t)b_val;
            }
            c[i * N + j] = (int16_t)acc;
        }
    }
}

int64_t vector_compute_time = 0;

void vector_int2_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                        unsigned long int M, unsigned long int K, unsigned long int N)
{
    int2_reference_matmul(c, a, b, M, K, N);
}
