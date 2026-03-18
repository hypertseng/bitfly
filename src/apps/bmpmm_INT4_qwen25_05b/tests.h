#ifndef BMPMM_INT4_TESTS_H
#define BMPMM_INT4_TESTS_H

#include <stdint.h>
/* bring in common KernelData definition */
#include "kernel/data.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void run_test(const char *test_name, int M, int K, int N);
    void compare_results(int M, int K, int N);

    /* forward declarations for INT4 kernels */
    void int4_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                           const unsigned long int M, const unsigned long int K,
                           const unsigned long int N);
    void vector_int4_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                            unsigned long int M, unsigned long int K, unsigned long int N);

#ifdef __cplusplus
}
#endif

#endif // BMPMM_INT4_TESTS_H
