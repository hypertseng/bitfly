#ifndef BMPMM_INT2_TESTS_H
#define BMPMM_INT2_TESTS_H

#include "kernel/data.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void run_test(const char *test_name, int M, int K, int N);
    void compare_results(int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif // BMPMM_INT2_TESTS_H
