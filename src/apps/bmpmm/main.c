#include "runtime.h"
#include "util.h"

#include "kernel/bmpmm.h"

extern int8_t activation_lp_K_16[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_K_16[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_K_16[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t activation_hp_K_16[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_hp_K_16[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_hp_K_16[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_torch_K_16[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_K_32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_K_32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_K_32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t activation_hp_K_32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_hp_K_32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_hp_K_32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_torch_K_32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_K_64[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_K_64[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_K_64[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t activation_hp_K_64[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_hp_K_64[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_hp_K_64[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_torch_K_64[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_K_128[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_K_128[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_K_128[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t activation_hp_K_128[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_hp_K_128[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_hp_K_128[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_torch_K_128[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_K_256[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_K_256[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_K_256[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t activation_hp_K_256[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_hp_K_256[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_hp_K_256[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_torch_K_256[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_K_480[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_K_480[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_K_480[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t activation_hp_K_480[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_hp_K_480[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_hp_K_480[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_torch_K_480[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

mixed_kernel_func get_mixed_kernel(int K);

KernelData get_kernel_data(int K)
{
    switch (K)
    {
    case 16:
        return (KernelData){
            .activation_lp = activation_lp_K_16,
            .weight_lp = weight_lp_K_16,
            .result_lp = result_lp_K_16,
            .activation_hp = activation_hp_K_16,
            .weight_hp = weight_hp_K_16,
            .result_hp = result_hp_K_16,
            .result_torch = result_torch_K_16};
    case 32:
        return (KernelData){
            .activation_lp = activation_lp_K_32,
            .weight_lp = weight_lp_K_32,
            .result_lp = result_lp_K_32,
            .activation_hp = activation_hp_K_32,
            .weight_hp = weight_hp_K_32,
            .result_hp = result_hp_K_32,
            .result_torch = result_torch_K_32};
    case 64:
        return (KernelData){
            .activation_lp = activation_lp_K_64,
            .weight_lp = weight_lp_K_64,
            .result_lp = result_lp_K_64,
            .activation_hp = activation_hp_K_64,
            .weight_hp = weight_hp_K_64,
            .result_hp = result_hp_K_64,
            .result_torch = result_torch_K_64};
    case 128:
        return (KernelData){
            .activation_lp = activation_lp_K_128,
            .weight_lp = weight_lp_K_128,
            .result_lp = result_lp_K_128,
            .activation_hp = activation_hp_K_128,
            .weight_hp = weight_hp_K_128,
            .result_hp = result_hp_K_128,
            .result_torch = result_torch_K_128};
    case 256:
        return (KernelData){
            .activation_lp = activation_lp_K_256,
            .weight_lp = weight_lp_K_256,
            .result_lp = result_lp_K_256,
            .activation_hp = activation_hp_K_256,
            .weight_hp = weight_hp_K_256,
            .result_hp = result_hp_K_256,
            .result_torch = result_torch_K_256};
    case 480:
        return (KernelData){
            .activation_lp = activation_lp_K_480,
            .weight_lp = weight_lp_K_480,
            .result_lp = result_lp_K_480,
            .activation_hp = activation_hp_K_480,
            .weight_hp = weight_hp_K_480,
            .result_hp = result_hp_K_480,
            .result_torch = result_torch_K_480};
    default:
        printf("Unsupported K value: %lu\n", K);
        return (KernelData){0};
    }
}

void run_test(const char *test_name, int M, int K, int N)
{
    printf("Running test: %s | M=%lu, K=%lu, N=%lu\n", test_name, M, K, N);

    KernelData data = get_kernel_data(K);
    if (!data.activation_lp)
        return;

    int64_t runtime;

    if (strcmp(test_name, "mixed") == 0)
    {
        mixed_kernel_func kernel = get_mixed_kernel(K);
        if (!kernel)
            return;

        start_timer();
        kernel(data.result_lp, data.activation_lp, data.weight_lp);
        stop_timer();
        runtime = get_timer();
    }
    else if (strcmp(test_name, "vector") == 0)
    {
        start_timer();
        vector_int8_matmul(data.result_hp, data.activation_hp, data.weight_hp, M, K, N);
        stop_timer();
        runtime = get_timer();
    }
    else
    {
        printf("Unknown test mode: %s\n", test_name);
        return;
    }

    float performance = 2.0 * M * K * N / runtime;
    float utilization = 100 * performance / (16.0 * NR_LANES);

    printf("Execution took %ld cycles.\n", runtime);
    printf("Performance: %.2f OP/cycle (%.2f%% utilization)\n\n", performance, utilization);
}

void compare_results(int M, int N, const int16_t *result_lp, const int32_t *result_torch)
{
    int errors = 0;

    // === 打印 result_lp === 只打印4行
    printf("result_lp(the first four rows):\n");
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < N / 4; ++j)
        {
            for (int k = 0; k < 4; k++)
                printf("%d ", result_lp[j * 4 * M + 4 * i + k]);
        }
        printf("\n");
    }
    printf("\n");

    // === 打印 result_torch ===
    printf("result_torch(the first four rows):\n");
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%d ", result_torch[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    // 比较结果
    int cnt = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int idx_lp = (j / 4) * 4 * M + 4 * i + j % 4;
            int idx_hp = i * N + j; // 行主序

            if ((int32_t)result_lp[idx_lp] != result_torch[idx_hp])
            {
                printf("Mismatch at [%d][%d]: got %d vs %d\n",
                       i, j,
                       (int32_t)result_lp[idx_lp],
                       result_torch[idx_hp]);
                errors++;
            }
            cnt++;
            if (cnt == 20)
                return; // 仅检查前 20 个
        }
    }

    // 输出统计结果
    if (errors == 0)
    {
        printf("All results matched with no mismatches!\n");
    }
    else
    {
        printf("Total mismatches found: %d\n", errors);
    }
}

int main()
{
    const int M = 16;
    const int N = 32;
    const int K_DIMS[] = {16, 32, 64, 128, 256, 480};
    // const int K_DIMS[] = {64};
    const int NUM_K_DIMS = sizeof(K_DIMS) / sizeof(K_DIMS[0]);

    for (int i = 0; i < NUM_K_DIMS; ++i)
    {
        int K = K_DIMS[i];
        KernelData data = get_kernel_data(K);
        printf("\n");
        printf("------------------------------------------------------------\n");
        printf("Calculating a (%d x %d) x (%d x %d) Binary mixed precision matrix multiplication...\n", M,
               K, K, N);
        printf("------------------------------------------------------------\n");
        printf("\n");
        run_test("mixed", M, K, N);
        // printf("\n");
        // printf("------------------------------------------------------------\n");
        // printf("Calculating a (%d x %d) x (%d x %d) RVV int8 matrix multiplication...\n", M,
        //        K, K, N);
        // printf("------------------------------------------------------------\n");
        // printf("\n");
        // run_test("vector", M, K, N);

        if (K == 16)
            compare_results(M, N, data.result_lp, data.result_torch);
    }

    return 0;
}
