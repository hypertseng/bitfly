#include "runtime.h"
#include "util.h"

#include "kernel/bmpmm.h"

#define DECLARE_KERNEL_DATA(K) \
extern int8_t activation_lp_len_##K[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); \
extern int8_t weight_lp_len_##K[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); \
extern int16_t result_lp_len_##K[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); \
extern int8_t activation_hp_len_##K[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); \
extern int8_t weight_hp_len_##K[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); \
extern int16_t result_hp_len_##K[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); \
extern int16_t result_torch_len_##K[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

// 使用
DECLARE_KERNEL_DATA(8)
DECLARE_KERNEL_DATA(16)
DECLARE_KERNEL_DATA(32)
DECLARE_KERNEL_DATA(64)
DECLARE_KERNEL_DATA(128)
DECLARE_KERNEL_DATA(256)
DECLARE_KERNEL_DATA(480)


KernelData get_kernel_data(int x)
{
    switch (x)
    {
    // case 1:
    //     return (KernelData){
    //         .activation_lp = activation_lp_len_1,
    //         .weight_lp = weight_lp_len_1,
    //         .result_lp = result_lp_len_1,
    //         .activation_hp = activation_hp_len_1,
    //         .weight_hp = weight_hp_len_1,
    //         .result_hp = result_hp_len_1,
    //         .result_torch = result_torch_len_1};
    case 8:
        return (KernelData){
            .activation_lp = activation_lp_len_8,
            .weight_lp = weight_lp_len_8,
            .result_lp = result_lp_len_8,
            .activation_hp = activation_hp_len_8,
            .weight_hp = weight_hp_len_8,
            .result_hp = result_hp_len_8,
            .result_torch = result_torch_len_8};
    case 16:
        return (KernelData){
            .activation_lp = activation_lp_len_16,
            .weight_lp = weight_lp_len_16,
            .result_lp = result_lp_len_16,
            .activation_hp = activation_hp_len_16,
            .weight_hp = weight_hp_len_16,
            .result_hp = result_hp_len_16,
            .result_torch = result_torch_len_16};
    case 32:
        return (KernelData){
            .activation_lp = activation_lp_len_32,
            .weight_lp = weight_lp_len_32,
            .result_lp = result_lp_len_32,
            .activation_hp = activation_hp_len_32,
            .weight_hp = weight_hp_len_32,
            .result_hp = result_hp_len_32,
            .result_torch = result_torch_len_32};
    case 64:
        return (KernelData){
            .activation_lp = activation_lp_len_64,
            .weight_lp = weight_lp_len_64,
            .result_lp = result_lp_len_64,
            .activation_hp = activation_hp_len_64,
            .weight_hp = weight_hp_len_64,
            .result_hp = result_hp_len_64,
            .result_torch = result_torch_len_64};
    case 128:
        return (KernelData){
            .activation_lp = activation_lp_len_128,
            .weight_lp = weight_lp_len_128,
            .result_lp = result_lp_len_128,
            .activation_hp = activation_hp_len_128,
            .weight_hp = weight_hp_len_128,
            .result_hp = result_hp_len_128,
            .result_torch = result_torch_len_128};
    case 256:
        return (KernelData){
            .activation_lp = activation_lp_len_256,
            .weight_lp = weight_lp_len_256,
            .result_lp = result_lp_len_256,
            .activation_hp = activation_hp_len_256,
            .weight_hp = weight_hp_len_256,
            .result_hp = result_hp_len_256,
            .result_torch = result_torch_len_256};
    case 480:
        return (KernelData){
            .activation_lp = activation_lp_len_480,
            .weight_lp = weight_lp_len_480,
            .result_lp = result_lp_len_480,
            .activation_hp = activation_hp_len_480,
            .weight_hp = weight_hp_len_480,
            .result_hp = result_hp_len_480,
            .result_torch = result_torch_len_480};
    // case 512:
    //     return (KernelData){
    //         .activation_lp = activation_lp_len_512,
    //         .weight_lp = weight_lp_len_512,
    //         .result_lp = result_lp_len_512,
    //         .activation_hp = activation_hp_len_512,
    //         .weight_hp = weight_hp_len_512,
    //         .result_hp = result_hp_len_512,
    //         .result_torch = result_torch_len_512};
    // case 1024:
    //     return (KernelData){
    //         .activation_lp = activation_lp_len_1024,
    //         .weight_lp = weight_lp_len_1024,
    //         .result_lp = result_lp_len_1024,
    //         .activation_hp = activation_hp_len_1024,
    //         .weight_hp = weight_hp_len_1024,
    //         .result_hp = result_hp_len_1024,
    //         .result_torch = result_torch_len_1024};
    default:
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
        start_timer();
        binary_mixed_matmul(data.result_lp, data.activation_lp, data.weight_lp, M, K, N);
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
    else if (strcmp(test_name, "scalar") == 0)
    {
        start_timer();
        scalar_matmul(data.result_hp, data.activation_hp, data.weight_hp, M, K, N);
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

void compare_results(int M, int K, int N)
{
    int errors = 0;
    KernelData data = get_kernel_data(K);
    // === 打印 result_lp === 只打印1行
    printf("result_lp(the first row):\n");
    for (int j = 0; j < (N + 31) / 32; ++j){
        for (int k = 0; k < 32 / 4; k++){
            for (int i = 0; i < 4; i++)
            {
                int idx = j * 16 * 32 + k * 16 * 4 + i;
                printf("%d ", data.result_lp[idx]);
            }
        }
    }
    printf("\n");

    // === 打印 result_torch ===
    printf("result_torch(the first rows):\n");
    for (int j = 0; j < N; ++j)
    {
        printf("%d ", data.result_torch[j]);
    }
    printf("\n");

    // 比较结果
    int cnt = 0;
    for (int i = 0; i < M; ++i)
    {
        int tm = i / 16;
        for (int j = 0; j < N; ++j)
        {
            int tn = j / 32;
            int idx_lp = tm * 16 * N + tn * 16 * 32 + ((j % 32) / 4) * 4 * 16 + 4 * (i % 16) + (j % 32) % 4;
            int idx_hp = i * N + j; // 行主序

            if ((int32_t)data.result_lp[idx_lp] != data.result_torch[idx_hp])
            {
                printf("Mismatch at [%d][%d]: got %d vs %d\n",
                       i, j,
                       (int32_t)data.result_lp[idx_lp],
                       data.result_torch[idx_hp]);
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
    // const int M_DIMS[] = {1, 16, 32, 64, 128, 256, 512};
    // const int M_DIMS[] = {50};
    const int M = 16;
    const int N = 32;
    // const int K_DIMS[] = {8, 16, 32, 64, 128};
    const int K_DIMS[] = {256, 480};
    // const int K = 500;
    const int NUM_K_DIMS = sizeof(K_DIMS) / sizeof(K_DIMS[0]);

    for (int i = 0; i < NUM_K_DIMS; ++i)
    {
        int K = K_DIMS[i];
        printf("\n");
        printf("------------------------------------------------------------\n");
        printf("Calculating a (%d x %d) x (%d x %d) Binary mixed precision matrix multiplication...\n", M,
               K, K, N);
        printf("------------------------------------------------------------\n");
        printf("\n");
        run_test("mixed", M, K, N);
        printf("\n");
        printf("------------------------------------------------------------\n");
        printf("Calculating a (%d x %d) x (%d x %d) RVV int8 matrix multiplication...\n", M,
               K, K, N);
        printf("------------------------------------------------------------\n");
        printf("\n");
        run_test("vector", M, K, N);
        printf("------------------------------------------------------------\n");
        printf("Calculating a (%d x %d) x (%d x %d) RVV scalar matrix multiplication...\n", M,
               K, K, N);
        printf("------------------------------------------------------------\n");
        printf("\n");
        run_test("scalar", M, K, N);

        // compare_results(M, K, N);
    }

    return 0;
}

