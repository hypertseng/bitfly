#include "runtime.h"
#include "util.h"
#include <stdint.h>
#include <string.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "../common/bmpmm_bench_common.h"
#include "kernel/data.h"
#include "kernel/bench_cases.h"
#include "kernel/vector.h"

int main()
{
    printf("[rvv_binary] precision=binary\n");
    const char *current_model = 0;
    int64_t rvv_model_cycles = 0;
    for (int i = 0; i < BMPMM_BENCH_CASE_COUNT; ++i)
    {
        const bmpmm_bench_case_t *sc = &kBenchCases[i];
        BenchKernelData data = get_bench_kernel_data(i);
        if (!current_model || strcmp(current_model, sc->model) != 0)
        {
            if (current_model)
            {
                printf("[rvv_binary] model_total model=%s rvv_cycles=%ld\n", current_model, (long)rvv_model_cycles);
            }
            current_model = sc->model;
            rvv_model_cycles = 0;
            printf("\n============================================================\n");
            printf("[rvv_binary] model=%s scale=%s\n", sc->model, sc->scale);
            printf("============================================================\n");
        }
        printf("\n------------------------------------------------------------\n");
        printf("[rvv_binary] case%d layer=%s shape=(%lu,%lu,%lu)\n", i + 1, sc->layer, sc->M, sc->N, sc->K);

        vector_compute_time = 0;
        start_timer();
        vector_int8_matmul(data.result_hp, data.activation_hp, data.weight_hp, sc->M, sc->K, sc->N);
        stop_timer();
        int64_t rvv_runtime = get_timer();
        rvv_model_cycles += rvv_runtime;
        printf("[rvv_binary] rvv_runtime=%ld rvv_compute=%ld\n", (long)rvv_runtime, (long)vector_compute_time);
    }
    if (current_model)
    {
        printf("[rvv_binary] model_total model=%s rvv_cycles=%ld\n", current_model, (long)rvv_model_cycles);
    }
    return 0;
}
