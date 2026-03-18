#ifndef BMPMM_TESTS_H
#define BMPMM_TESTS_H

#include "kernel/bmpmm.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void run_test(const char *test_name, int M, int K, int N);
    void compare_results(int M, int K, int N);
    int validate_mixed_cfg(int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif // BMPMM_TESTS_H
