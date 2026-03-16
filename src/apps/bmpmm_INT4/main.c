#include "runtime.h"
#include "util.h"
#include <stdint.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "../common/bmpmm_bench_common.h"
#include "kernel/data.h"
#include "kernel/mixed.h"

BMPMM_DEFINE_BENCH_CASES_G(kBenchCases, 3UL, 32UL, 1UL, 1UL, 1UL, 1UL);

int main()
{
    printf("[bmpmm_INT4] precision=int4\n");

    for (int i = 0; i < 5; ++i)
    {
        const bmpmm_bench_case_t *sc = &kBenchCases[i];
        BenchKernelData data = get_bench_kernel_data(i);

        printf("\n------------------------------------------------------------\n");
        printf("[bmpmm_INT4] case%d shape=(%lu,%lu,%lu), cfg=(mt=%lu,nt=%lu,kt=%lu,gm=%lu,gn=%lu,p=%lu)\n",
               i + 1, sc->M, sc->N, sc->K,
               sc->cfg.mtile, sc->cfg.ntile, sc->cfg.ktile,
               sc->cfg.gm, sc->cfg.gn, sc->cfg.prec);

        start_timer();
        int ok = int4_mixed_matmul_with_cfg(
            data.result_lp, data.activation_lp, data.weight_lp,
            sc->M, sc->K, sc->N,
            &sc->cfg);
        stop_timer();

        if (!ok)
        {
            printf("[bmpmm_INT4] ERROR: execution failed for case%d\n", i + 1);
            continue;
        }

        const int64_t runtime = get_timer();
        const float performance = 2.0f * (float)sc->M * (float)sc->K * (float)sc->N / (float)runtime;
        const float utilization = 100.0f * performance / (16.0f * NR_LANES);

        printf("[bmpmm_INT4] runtime=%ld cycles, perf=%.2f OP/cycle, util=%.2f%%\n",
               (long)runtime, performance, utilization);
        printf("[bmpmm_INT4] sample C(col-major) = {%d, %d, %d, %d}\n",
               data.result_lp[0], data.result_lp[1], data.result_lp[2], data.result_lp[3]);
    }

    return 0;
}
