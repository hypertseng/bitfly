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

#include "../common/model_bench_common.h"
#include "kernel/data.h"
#include "kernel/bench_cases.h"
#include "kernel/vector.h"

int main()
{
    printf("[rvv_INT2] precision=int2\n");
    const char *current_model = 0;
    int64_t rvv_model_cycles = 0;
    const int fast_mode = (vector_int2_get_default_mode() == RVV_INT2_VECTOR_EXEC_FAST);
    model_runtime_cache_entry_t runtime_cache[MODEL_RUNTIME_CACHE_CAP] = {0};
    int runtime_cache_count = 0;
    for (int i = 0; i < BMPMM_BENCH_CASE_COUNT; ++i)
    {
        const model_bench_case_t *sc = &kBenchCases[i];
        BenchKernelData data = get_bench_kernel_data(i);
        if (!current_model || strcmp(current_model, sc->model) != 0)
        {
            if (current_model)
            {
                printf("[rvv_INT2] model_total model=%s rvv_cycles=%ld\n", current_model, (long)rvv_model_cycles);
            }
            current_model = sc->model;
            rvv_model_cycles = 0;
            printf("\n============================================================\n");
            printf("[rvv_INT2] model=%s scale=%s\n", sc->model, sc->scale);
            printf("============================================================\n");
        }
        printf("\n------------------------------------------------------------\n");
        printf("[rvv_INT2] case%d layer=%s shape=(%lu,%lu,%lu)\n", i+1, sc->layer, sc->M, sc->N, sc->K);

        model_runtime_cache_entry_t *cached = model_runtime_cache_lookup(runtime_cache, runtime_cache_count, sc);
        if (cached)
        {
            rvv_model_cycles += cached->runtime;
            printf("[rvv_INT2] duplicate_shape_skip case%d reuse_case%d\n", i + 1, cached->first_case_index + 1);
            printf("[rvv_INT2] rvv_runtime=%ld rvv_compute=%ld\n", (long)cached->runtime, (long)cached->aux_cycles);
            continue;
        }

        vector_compute_time = 0;
        start_timer();
        vector_int2_matmul(data.result_hp, data.activation_hp, data.weight_hp, sc->M, sc->K, sc->N);
        stop_timer();
        int64_t rvv_runtime = fast_mode ? vector_int2_get_last_estimated_total_cycles() : get_timer();
        int64_t rvv_compute = fast_mode ? vector_int2_get_last_estimated_compute_cycles() : vector_compute_time;
        rvv_model_cycles += rvv_runtime;
        model_runtime_cache_store(runtime_cache, &runtime_cache_count, MODEL_RUNTIME_CACHE_CAP,
                                  sc, i, rvv_runtime, rvv_compute, data.result_hp);
        printf("[rvv_INT2] rvv_runtime=%ld rvv_compute=%ld\n", (long)rvv_runtime, (long)rvv_compute);
    }
    if (current_model)
    {
        printf("[rvv_INT2] model_total model=%s rvv_cycles=%ld\n", current_model, (long)rvv_model_cycles);
    }
    return 0;
}
