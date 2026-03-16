#include "runtime.h"
#include "util.h"
#include <stdint.h>

#include "kernel/bmpmm.h"

#define TOPK 5

typedef struct
{
    unsigned long M;
    unsigned long N;
    unsigned long K;
    binary_exec_cfg_t cfg;
    int8_t *a;
    int8_t *b;
    int16_t *c;
} shape_cfg_t;

extern int8_t activation_lp_top1[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_top1[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_top1[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_top2[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_top2[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_top2[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_top3[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_top3[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_top3[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_top4[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_top4[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_top4[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern int8_t activation_lp_top5[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_lp_top5[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int16_t result_lp_top5[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

static const shape_cfg_t kTopShapes[TOPK] = {
    {128, 128, 896, {.mtile = 8, .ntile = 64, .ktile = 128, .gm = 1, .gn = 2, .prec = 0}, activation_lp_top1, weight_lp_top1, result_lp_top1},
    {128, 256, 640, {.mtile = 8, .ntile = 64, .ktile = 128, .gm = 1, .gn = 2, .prec = 0}, activation_lp_top2, weight_lp_top2, result_lp_top2},
    {128, 256, 1536, {.mtile = 8, .ntile = 64, .ktile = 128, .gm = 1, .gn = 2, .prec = 0}, activation_lp_top3, weight_lp_top3, result_lp_top3},
    {128, 256, 2048, {.mtile = 8, .ntile = 64, .ktile = 128, .gm = 1, .gn = 2, .prec = 0}, activation_lp_top4, weight_lp_top4, result_lp_top4},
    {128, 320, 960, {.mtile = 8, .ntile = 64, .ktile = 128, .gm = 2, .gn = 1, .prec = 0}, activation_lp_top5, weight_lp_top5, result_lp_top5},
};

int main()
{
    printf("[bmpmm_binary] precision=binary\n");

    for (int i = 0; i < TOPK; ++i)
    {
        const shape_cfg_t *sc = &kTopShapes[i];

        printf("\n------------------------------------------------------------\n");
        printf("[bmpmm_binary] top%d shape=(%lu,%lu,%lu), cfg=(mt=%lu,nt=%lu,kt=%lu,gm=%lu,gn=%lu,p=%lu)\n",
               i + 1, sc->M, sc->N, sc->K,
               sc->cfg.mtile, sc->cfg.ntile, sc->cfg.ktile,
               sc->cfg.gm, sc->cfg.gn, sc->cfg.prec);

        start_timer();
        int ok = binary_mixed_matmul_with_cfg(
            sc->c, sc->a, sc->b,
            sc->M, sc->K, sc->N,
            &sc->cfg);
        stop_timer();

        if (!ok)
        {
            printf("[bmpmm_binary] ERROR: execution failed for top%d\n", i + 1);
            continue;
        }

        const int64_t runtime = get_timer();
        const float performance = 2.0f * (float)sc->M * (float)sc->K * (float)sc->N / (float)runtime;
        const float utilization = 100.0f * performance / (16.0f * NR_LANES);

        printf("[bmpmm_binary] runtime=%ld cycles, perf=%.2f OP/cycle, util=%.2f%%\n",
               (long)runtime, performance, utilization);
        printf("[bmpmm_binary] sample C(col-major) = {%d, %d, %d, %d}\n",
               sc->c[0], sc->c[1], sc->c[2], sc->c[3]);
    }

    return 0;
}
