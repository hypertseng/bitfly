#ifndef BMPMM_LOWP_MIXED_COMMON_H
#define BMPMM_LOWP_MIXED_COMMON_H

#include <stdint.h>
#include "runtime.h"
#include "bmpmm_bench_common.h"
#include "bmpmm_operator_template.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define BMPMM_STR_IMPL(x) #x
#define BMPMM_STR(x) BMPMM_STR_IMPL(x)

#define BMPCFG_CASE(PREC, K, M, N, GM, GN) \
    case K: \
        asm volatile("bmpcfg " BMPMM_STR(PREC) ", " BMPMM_STR(K) ", " BMPMM_STR(M) ", " BMPMM_STR(N) ", " BMPMM_STR(GM) ", " BMPMM_STR(GN) "\n\t" : : : "memory"); \
        return 1

#define BMPCFG_K_SWITCH(PREC, M, N, GM, GN, K) \
    switch (K) \
    { \
        BMPCFG_CASE(PREC, 8, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 16, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 24, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 32, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 40, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 48, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 56, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 64, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 72, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 80, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 88, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 96, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 104, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 112, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 120, M, N, GM, GN); \
        BMPCFG_CASE(PREC, 128, M, N, GM, GN); \
    default: \
        return 0; \
    }

static inline unsigned long bmpmm_lowp_min_ul(unsigned long a, unsigned long b)
{
    return (a < b) ? a : b;
}

static inline unsigned long bmpmm_lowp_align_up_ul(unsigned long x, unsigned long a)
{
    return ((x + a - 1) / a) * a;
}

static inline unsigned long bmpmm_planes_from_prec(unsigned long prec)
{
    return (prec == 3UL) ? 4UL : ((prec == 1UL || prec == 2UL) ? 2UL : 1UL);
}

static inline int bmpcfg_emit_prec(unsigned long prec, unsigned long K,
                                   unsigned long mtile, unsigned long ntile,
                                   unsigned long gm, unsigned long gn)
{
    if (mtile != 8UL || ntile != 64UL)
        return 0;

    switch (prec)
    {
    case 0:
        switch (gm)
        {
        case 1: switch (gn) { case 1: BMPCFG_K_SWITCH(0, 8, 64, 1, 1, K); case 2: BMPCFG_K_SWITCH(0, 8, 64, 1, 2, K); case 3: BMPCFG_K_SWITCH(0, 8, 64, 1, 3, K); case 4: BMPCFG_K_SWITCH(0, 8, 64, 1, 4, K); default: return 0; }
        case 2: switch (gn) { case 1: BMPCFG_K_SWITCH(0, 8, 64, 2, 1, K); case 2: BMPCFG_K_SWITCH(0, 8, 64, 2, 2, K); case 3: BMPCFG_K_SWITCH(0, 8, 64, 2, 3, K); case 4: BMPCFG_K_SWITCH(0, 8, 64, 2, 4, K); default: return 0; }
        case 3: switch (gn) { case 1: BMPCFG_K_SWITCH(0, 8, 64, 3, 1, K); case 2: BMPCFG_K_SWITCH(0, 8, 64, 3, 2, K); case 3: BMPCFG_K_SWITCH(0, 8, 64, 3, 3, K); case 4: BMPCFG_K_SWITCH(0, 8, 64, 3, 4, K); default: return 0; }
        case 4: switch (gn) { case 1: BMPCFG_K_SWITCH(0, 8, 64, 4, 1, K); case 2: BMPCFG_K_SWITCH(0, 8, 64, 4, 2, K); case 3: BMPCFG_K_SWITCH(0, 8, 64, 4, 3, K); case 4: BMPCFG_K_SWITCH(0, 8, 64, 4, 4, K); default: return 0; }
        default: return 0;
        }
    case 1:
        switch (gm)
        {
        case 1: switch (gn) { case 1: BMPCFG_K_SWITCH(1, 8, 64, 1, 1, K); case 2: BMPCFG_K_SWITCH(1, 8, 64, 1, 2, K); case 3: BMPCFG_K_SWITCH(1, 8, 64, 1, 3, K); case 4: BMPCFG_K_SWITCH(1, 8, 64, 1, 4, K); default: return 0; }
        case 2: switch (gn) { case 1: BMPCFG_K_SWITCH(1, 8, 64, 2, 1, K); case 2: BMPCFG_K_SWITCH(1, 8, 64, 2, 2, K); case 3: BMPCFG_K_SWITCH(1, 8, 64, 2, 3, K); case 4: BMPCFG_K_SWITCH(1, 8, 64, 2, 4, K); default: return 0; }
        case 3: switch (gn) { case 1: BMPCFG_K_SWITCH(1, 8, 64, 3, 1, K); case 2: BMPCFG_K_SWITCH(1, 8, 64, 3, 2, K); case 3: BMPCFG_K_SWITCH(1, 8, 64, 3, 3, K); case 4: BMPCFG_K_SWITCH(1, 8, 64, 3, 4, K); default: return 0; }
        case 4: switch (gn) { case 1: BMPCFG_K_SWITCH(1, 8, 64, 4, 1, K); case 2: BMPCFG_K_SWITCH(1, 8, 64, 4, 2, K); case 3: BMPCFG_K_SWITCH(1, 8, 64, 4, 3, K); case 4: BMPCFG_K_SWITCH(1, 8, 64, 4, 4, K); default: return 0; }
        default: return 0;
        }
    case 2:
        switch (gm)
        {
        case 1: switch (gn) { case 1: BMPCFG_K_SWITCH(2, 8, 64, 1, 1, K); case 2: BMPCFG_K_SWITCH(2, 8, 64, 1, 2, K); case 3: BMPCFG_K_SWITCH(2, 8, 64, 1, 3, K); case 4: BMPCFG_K_SWITCH(2, 8, 64, 1, 4, K); default: return 0; }
        case 2: switch (gn) { case 1: BMPCFG_K_SWITCH(2, 8, 64, 2, 1, K); case 2: BMPCFG_K_SWITCH(2, 8, 64, 2, 2, K); case 3: BMPCFG_K_SWITCH(2, 8, 64, 2, 3, K); case 4: BMPCFG_K_SWITCH(2, 8, 64, 2, 4, K); default: return 0; }
        case 3: switch (gn) { case 1: BMPCFG_K_SWITCH(2, 8, 64, 3, 1, K); case 2: BMPCFG_K_SWITCH(2, 8, 64, 3, 2, K); case 3: BMPCFG_K_SWITCH(2, 8, 64, 3, 3, K); case 4: BMPCFG_K_SWITCH(2, 8, 64, 3, 4, K); default: return 0; }
        case 4: switch (gn) { case 1: BMPCFG_K_SWITCH(2, 8, 64, 4, 1, K); case 2: BMPCFG_K_SWITCH(2, 8, 64, 4, 2, K); case 3: BMPCFG_K_SWITCH(2, 8, 64, 4, 3, K); case 4: BMPCFG_K_SWITCH(2, 8, 64, 4, 4, K); default: return 0; }
        default: return 0;
        }
    case 3:
        switch (gm)
        {
        case 1: switch (gn) { case 1: BMPCFG_K_SWITCH(3, 8, 64, 1, 1, K); case 2: BMPCFG_K_SWITCH(3, 8, 64, 1, 2, K); case 3: BMPCFG_K_SWITCH(3, 8, 64, 1, 3, K); case 4: BMPCFG_K_SWITCH(3, 8, 64, 1, 4, K); default: return 0; }
        case 2: switch (gn) { case 1: BMPCFG_K_SWITCH(3, 8, 64, 2, 1, K); case 2: BMPCFG_K_SWITCH(3, 8, 64, 2, 2, K); case 3: BMPCFG_K_SWITCH(3, 8, 64, 2, 3, K); case 4: BMPCFG_K_SWITCH(3, 8, 64, 2, 4, K); default: return 0; }
        case 3: switch (gn) { case 1: BMPCFG_K_SWITCH(3, 8, 64, 3, 1, K); case 2: BMPCFG_K_SWITCH(3, 8, 64, 3, 2, K); case 3: BMPCFG_K_SWITCH(3, 8, 64, 3, 3, K); case 4: BMPCFG_K_SWITCH(3, 8, 64, 3, 4, K); default: return 0; }
        case 4: switch (gn) { case 1: BMPCFG_K_SWITCH(3, 8, 64, 4, 1, K); case 2: BMPCFG_K_SWITCH(3, 8, 64, 4, 2, K); case 3: BMPCFG_K_SWITCH(3, 8, 64, 4, 3, K); case 4: BMPCFG_K_SWITCH(3, 8, 64, 4, 4, K); default: return 0; }
        default: return 0;
        }
    default:
        return 0;
    }
}

typedef struct
{
    const char *app_tag;
    unsigned long planes;
    int invalid_cfg;
    int64_t *compute_cycles;
} bmpmm_lowp_template_ctx_t;

static inline void bmpmm_lowp_emit_cfg(const bmpmm_template_cfg_t *cfg, unsigned long k_cfg, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    if (!bmpcfg_emit_prec(cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn))
    {
        printf("[%s] ERROR: unsupported bmpcfg tuple p=%lu k=%lu mt=%lu nt=%lu gm=%lu gn=%lu\n",
               ctx->app_tag, cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn);
        ctx->invalid_cfg = 1;
    }
}

static inline const void *bmpmm_lowp_addr_a(const void *A, const bmpmm_template_cfg_t *cfg,
                                            unsigned long m_tile_idx, unsigned long k0, void *user)
{
    (void)user;
    const int8_t *a = (const int8_t *)A;
    const unsigned long k_aligned = bmpmm_lowp_align_up_ul(cfg->K, 8UL);
    return a + m_tile_idx * cfg->mtile * k_aligned + k0;
}

static inline const void *bmpmm_lowp_addr_b(const void *B, const bmpmm_template_cfg_t *cfg,
                                            unsigned long n_tile_idx, unsigned long k0, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    const int8_t *b = (const int8_t *)B;
    const unsigned long n_groups_total = (cfg->N + 7UL) / 8UL;
    const unsigned long n_group0 = (n_tile_idx * cfg->ntile) / 8UL;
    const unsigned long k_blk0 = k0 / 8UL;
    return b + (k_blk0 * ctx->planes * n_groups_total + n_group0) * 8UL;
}

static inline void *bmpmm_lowp_addr_c(void *C, const bmpmm_template_cfg_t *cfg,
                                      unsigned long m_tile_idx, unsigned long n_tile_idx, void *user)
{
    (void)user;
    int16_t *c = (int16_t *)C;
    return c + (n_tile_idx * cfg->ntile) * cfg->M + (m_tile_idx * cfg->mtile);
}

static inline void bmpmm_lowp_load_w(const void *ptr, unsigned long w_slot, void *user)
{
    (void)user;
    (void)w_slot;
    asm volatile("bmple 0(%0), w\n\t" : : "r"(ptr) : "memory");
}

static inline void bmpmm_lowp_load_a(const void *ptr, unsigned long a_slot, void *user)
{
    (void)user;
    (void)a_slot;
    asm volatile("bmple 0(%0), a\n\t" : : "r"(ptr) : "memory");
}

static inline void bmpmm_lowp_compute(void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    int64_t start = get_cycle_count();
    asm volatile("bmpmm\n\t" : : : "memory");
    if (ctx->compute_cycles)
        *ctx->compute_cycles += get_cycle_count() - start;
}

static inline void bmpmm_lowp_store_c(void *ptr, unsigned long a_slot, unsigned long w_slot, void *user)
{
    (void)user;
    (void)a_slot;
    (void)w_slot;
    asm volatile("bmpse 0(%0)\n\t" : : "r"(ptr) : "memory");
}

static inline int bmpmm_lowp_mixed_matmul_with_cfg(const char *app_tag,
                                                   int16_t *c, const int8_t *a, const int8_t *b,
                                                   unsigned long M, unsigned long K, unsigned long N,
                                                   const bmpmm_exec_cfg_t *exec_cfg,
                                                   int64_t *compute_cycles)
{
    if (!exec_cfg)
        return 0;

    if (exec_cfg->mtile == 0 || exec_cfg->ntile == 0 || exec_cfg->ktile == 0 ||
        exec_cfg->gm == 0 || exec_cfg->gn == 0 || exec_cfg->prec > 3)
        return 0;

    if ((exec_cfg->ktile & 7UL) != 0)
        return 0;

    bmpmm_template_cfg_t cfg = {
        .M = M,
        .K = K,
        .N = N,
        .mtile = exec_cfg->mtile,
        .ntile = exec_cfg->ntile,
        .ktile = exec_cfg->ktile,
        .gm = exec_cfg->gm,
        .gn = exec_cfg->gn,
        .prec = exec_cfg->prec,
    };

    bmpmm_template_ops_t ops = {
        .emit_cfg = bmpmm_lowp_emit_cfg,
        .addr_a = bmpmm_lowp_addr_a,
        .addr_b = bmpmm_lowp_addr_b,
        .addr_c = bmpmm_lowp_addr_c,
        .load_w = bmpmm_lowp_load_w,
        .load_a = bmpmm_lowp_load_a,
        .compute = bmpmm_lowp_compute,
        .store_c = bmpmm_lowp_store_c,
    };

    bmpmm_lowp_template_ctx_t ctx = {
        .app_tag = app_tag,
        .planes = bmpmm_planes_from_prec(cfg.prec),
        .invalid_cfg = 0,
        .compute_cycles = compute_cycles,
    };

    if (!bmpmm_template_execute(&cfg, &ops, a, b, c, &ctx) || ctx.invalid_cfg)
        return 0;

    return 1;
}

#endif
