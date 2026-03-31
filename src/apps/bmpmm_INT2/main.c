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
#include "../common/bmpmm_lowp_mixed_common.h"
#include "kernel/data.h"
#include "kernel/mixed.h"
#include "kernel/bench_cases.h"

int main()
{
    printf("[bmpmm_INT2] precision=int2\n");
    const char *current_model = 0;
    int64_t bmpmm_model_cycles = 0;
    const int fast_mode = (bmpmm_lowp_get_default_mode() == BMPMM_LOWP_EXEC_FAST);
    bmpmm_runtime_cache_entry_t runtime_cache[BMPMM_RUNTIME_CACHE_CAP] = {0};
    int runtime_cache_count = 0;
    for (int i = 0; i < BMPMM_BENCH_CASE_COUNT; ++i)
    {
        const bmpmm_bench_case_t *sc = &kBenchCases[i];
        BenchKernelData data = get_bench_kernel_data(i);
        if (!current_model || strcmp(current_model, sc->model) != 0)
        {
            if (current_model)
            {
                printf("[bmpmm_INT2] model_total model=%s bmpmm_cycles=%ld\n", current_model, (long)bmpmm_model_cycles);
            }
            current_model = sc->model;
            bmpmm_model_cycles = 0;
            printf("\n============================================================\n");
            printf("[bmpmm_INT2] model=%s scale=%s\n", sc->model, sc->scale);
            printf("============================================================\n");
        }
        printf("\n------------------------------------------------------------\n");
        printf("[bmpmm_INT2] case%d layer=%s shape=(%lu,%lu,%lu), cfg=(mt=%lu,nt=%lu,kt=%lu,gm=%lu,gn=%lu,p=%lu)\n", i+1, sc->layer, sc->M, sc->N, sc->K, sc->cfg.mtile, sc->cfg.ntile, sc->cfg.ktile, sc->cfg.gm, sc->cfg.gn, sc->cfg.prec);

        bmpmm_runtime_cache_entry_t *cached = bmpmm_runtime_cache_lookup(runtime_cache, runtime_cache_count, sc, 1);
        if (cached)
        {
            bmpmm_model_cycles += cached->runtime;
            printf("[bmpmm_INT2] duplicate_shape_skip case%d reuse_case%d\n", i + 1, cached->first_case_index + 1);
            printf("[bmpmm_INT2] bmpmm_runtime=%ld\n", (long)cached->runtime);
            continue;
        }

        start_timer();
        int ok = int2_mixed_matmul_with_cfg(data.result_lp, data.activation_lp, data.weight_lp, sc->M, sc->K, sc->N, &sc->cfg);
        stop_timer();
        if (!ok) {
            printf("[bmpmm_INT2] ERROR: bmpmm failed for case%d (%s)\n", i + 1, sc->layer);
            continue;
        }
        int64_t bmpmm_runtime = fast_mode ? bmpmm_lowp_get_last_estimated_total_cycles() : get_timer();
        int64_t bmpmm_compute = fast_mode ? bmpmm_lowp_get_last_estimated_compute_cycles() : 0;
        bmpmm_model_cycles += bmpmm_runtime;
        bmpmm_runtime_cache_store(runtime_cache, &runtime_cache_count, BMPMM_RUNTIME_CACHE_CAP,
                                  sc, i, bmpmm_runtime, bmpmm_compute, data.result_lp);
        printf("[bmpmm_INT2] bmpmm_runtime=%ld\n", (long)bmpmm_runtime);
    }
    if (current_model)
    {
        printf("[bmpmm_INT2] model_total model=%s bmpmm_cycles=%ld\n", current_model, (long)bmpmm_model_cycles);
    }
    return 0;
}
