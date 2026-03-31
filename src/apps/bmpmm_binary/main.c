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
#include "kernel/bmpmm.h"
#include "kernel/bench_cases.h"

int main()
{
    printf("[bmpmm_binary] precision=binary\n");
    const char *current_model = 0;
    int64_t bmpmm_model_cycles = 0;
    for (int i = 0; i < BMPMM_BENCH_CASE_COUNT; ++i)
    {
        const bmpmm_bench_case_t *sc = &kBenchCases[i];
        BenchKernelData data = get_bench_kernel_data(i);
        if (!current_model || strcmp(current_model, sc->model) != 0)
        {
            if (current_model)
            {
                printf("[bmpmm_binary] model_total model=%s bmpmm_cycles=%ld\n", current_model, (long)bmpmm_model_cycles);
            }
            current_model = sc->model;
            bmpmm_model_cycles = 0;
            printf("\n============================================================\n");
            printf("[bmpmm_binary] model=%s scale=%s\n", sc->model, sc->scale);
            printf("============================================================\n");
        }
        printf("\n------------------------------------------------------------\n");
        printf("[bmpmm_binary] case%d layer=%s shape=(%lu,%lu,%lu), cfg=(mt=%lu,nt=%lu,kt=%lu,gm=%lu,gn=%lu,p=%lu)\n", i + 1, sc->layer, sc->M, sc->N, sc->K, sc->cfg.mtile, sc->cfg.ntile, sc->cfg.ktile, sc->cfg.gm, sc->cfg.gn, sc->cfg.prec);

        start_timer();
        int ok = binary_mixed_matmul_with_cfg(data.result_lp, data.activation_lp, data.weight_lp, sc->M, sc->K, sc->N, &sc->cfg);
        stop_timer();
        if (!ok) {
            printf("[bmpmm_binary] ERROR: bmpmm failed for case%d (%s)\n", i + 1, sc->layer);
            continue;
        }
        int64_t bmpmm_runtime = get_timer();
        bmpmm_model_cycles += bmpmm_runtime;
        printf("[bmpmm_binary] bmpmm_runtime=%ld\n", (long)bmpmm_runtime);
    }
    if (current_model)
    {
        printf("[bmpmm_binary] model_total model=%s bmpmm_cycles=%ld\n", current_model, (long)bmpmm_model_cycles);
    }
    return 0;
}
