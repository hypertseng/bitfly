#include <stdio.h>
#include <string.h>
#include "tests.h"

static int compare_mixed_sampled(int M, int N)
{
    int errors = 0;
    KernelData data = get_kernel_data(N);
    const int total = M * N;
    const int max_checks = 512;
    const int checks = (total < max_checks) ? total : max_checks;
    const int step = (total > checks) ? (total / checks) : 1;
    int idx = 0;
    int checked = 0;

    while (checked < checks && idx < total)
    {
        int i = idx / N;
        int j = idx % N;
        int got = data.result_lp[idx];
        int expect = data.result_torch[idx];
        if (got != expect)
        {
            if (errors < 16)
            {
                printf("[mixed-g2] mismatch at [%d][%d]: got %d vs %d\n", i, j, got, expect);
            }
            errors++;
        }

        idx += step;
        checked++;
    }

    if (errors == 0)
    {
        printf("[mixed-g2] sampled check passed (%d/%d elements).\n", checked, total);
    }
    else
    {
        printf("[mixed-g2] sampled mismatches: %d (checked %d/%d elements)\n", errors, checked, total);
    }

    return errors;
}

static binary_exec_cfg_t default_exec_cfg(int K)
{
    binary_exec_cfg_t cfg;
    cfg.mtile = 8;
    cfg.ntile = 64;
    cfg.ktile = ((unsigned long)K >= 128UL) ? 128UL : (((unsigned long)K / 8UL) * 8UL);
    if (cfg.ktile < 8)
        cfg.ktile = 8;
    cfg.gm = 1;
    cfg.gn = 2;
    cfg.prec = 0;
    return cfg;
}

void run_test(const char *test_name, int M, int K, int N)
{
    printf("Running test: %s | M=%lu, K=%lu, N=%lu\n", test_name, (unsigned long)M, (unsigned long)K, (unsigned long)N);

    KernelData data = get_kernel_data(K);
    if (!data.activation_lp)
        return;

    int64_t runtime = 0;

    start_timer();
    if (strcmp(test_name, "mixed") == 0)
    {
        binary_exec_cfg_t cfg = default_exec_cfg(K);
        if (!binary_mixed_matmul_with_cfg(data.result_lp, data.activation_lp, data.weight_lp,
                                          (unsigned long)M, (unsigned long)K, (unsigned long)N,
                                          &cfg))
        {
            printf("[mixed] execution failed with cfg (mt=%lu nt=%lu kt=%lu gm=%lu gn=%lu p=%lu)\n",
                   cfg.mtile, cfg.ntile, cfg.ktile, cfg.gm, cfg.gn, cfg.prec);
            stop_timer();
            return;
        }
    }
    else if (strcmp(test_name, "vector") == 0)
    {
        vector_int8_matmul(data.result_hp, data.activation_hp, data.weight_hp, M, K, N);
    }
    else
    {
        printf("Unknown test mode: %s\n", test_name);
        return;
    }
    stop_timer();

    runtime = get_timer();
    float performance = 2.0 * M * K * N / runtime;
    float utilization = 100 * performance / (16.0 * NR_LANES);
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
            int idx_hp = i * N + j;
            int got = data.result_hp[idx_hp];
            int expect = data.result_torch[idx_hp];
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
    {
        printf("All checked results matched with no mismatches!\n");
    }
    else
    {
        printf("Total mismatches found: %d\n", errors);
    }
}

int validate_mixed_cfg(int M, int K, int N)
{
    KernelData data = get_kernel_data(K);
    binary_exec_cfg_t cfg = default_exec_cfg(K);

    if (!data.activation_lp)
    {
        printf("[mixed-g2] missing kernel data for K=%d\n", K);
        return 0;
    }

    printf("[mixed-cfg] cfg: mt=%lu nt=%lu kt=%lu gm=%lu gn=%lu p=%lu\n",
           cfg.mtile, cfg.ntile, cfg.ktile, cfg.gm, cfg.gn, cfg.prec);

    memset(data.result_lp, 0, (size_t)M * (size_t)N * sizeof(int16_t));
    if (!binary_mixed_matmul_with_cfg(data.result_lp, data.activation_lp, data.weight_lp,
                                      (unsigned long)M, (unsigned long)K, (unsigned long)N,
                                      &cfg))
    {
        printf("[mixed-cfg] execution failed\n");
        return 0;
    }

    return compare_mixed_sampled(M, N) == 0;
}
