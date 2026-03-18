#include <stdio.h>
#include <string.h>
#include "tests.h"
#include "runtime.h"

// forward declarations for INT2 kernels
void int2_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                       const unsigned long int M, const unsigned long int K,
                       const unsigned long int N);
void vector_int2_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                        unsigned long int M, unsigned long int K, unsigned long int N);

void run_test(const char *test_name, int M, int K, int N)
{
    printf("Running test: %s | M=%lu, K=%lu, N=%lu\n", test_name, (unsigned long)M, (unsigned long)K, (unsigned long)N);

    KernelData data = get_kernel_data(K);
    if (!data.activation_lp)
        return;

    start_timer();
    if (strcmp(test_name, "mixed") == 0)
    {
        int2_mixed_matmul(data.result_lp, data.activation_lp, data.weight_lp, M, K, N);
    }
    else if (strcmp(test_name, "vector") == 0)
    {
        vector_int2_matmul(data.result_hp, data.activation_hp, data.weight_hp, M, K, N);
    }
    else
    {
        printf("Unknown test mode: %s\n", test_name);
        return;
    }
    stop_timer();

    int64_t runtime = get_timer();
    float performance = 2.0f * M * K * N / runtime;
    float utilization = 100.0f * performance / (16.0f * NR_LANES);
    printf("Execution took %ld cycles.\n", runtime);
    printf("Performance: %.2f OP/cycle (%.2f%% utilization)\n\n", performance, utilization);
}

void compare_results(int M, int K, int N)
{
    int errors = 0;
    KernelData data = get_kernel_data(K);

    int cnt = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int idx = i * N + j;
            int got = data.result_hp[idx];
            int expect = data.result_torch[idx];
            if (got != expect)
            {
                printf("Mismatch at [%d][%d]: got %d vs %d\n", i, j, got, expect);
                errors++;
            }
            cnt++;
            if (cnt == 20)
                goto out;
        }
    }
out:
    if (errors == 0)
        printf("All checked results matched with no mismatches!\n");
    else
        printf("Total mismatches found: %d\n", errors);
}
