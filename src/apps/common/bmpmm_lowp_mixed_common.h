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

#define BMPCFG_GM_GN_SWITCH(PREC, NVAL, K) \
    switch (gm) \
    { \
    case 1: switch (gn) { case 1: BMPCFG_K_SWITCH(PREC, 8, NVAL, 1, 1, K); case 2: BMPCFG_K_SWITCH(PREC, 8, NVAL, 1, 2, K); case 3: BMPCFG_K_SWITCH(PREC, 8, NVAL, 1, 3, K); case 4: BMPCFG_K_SWITCH(PREC, 8, NVAL, 1, 4, K); default: return 0; } \
    case 2: switch (gn) { case 1: BMPCFG_K_SWITCH(PREC, 8, NVAL, 2, 1, K); case 2: BMPCFG_K_SWITCH(PREC, 8, NVAL, 2, 2, K); case 3: BMPCFG_K_SWITCH(PREC, 8, NVAL, 2, 3, K); case 4: BMPCFG_K_SWITCH(PREC, 8, NVAL, 2, 4, K); default: return 0; } \
    case 3: switch (gn) { case 1: BMPCFG_K_SWITCH(PREC, 8, NVAL, 3, 1, K); case 2: BMPCFG_K_SWITCH(PREC, 8, NVAL, 3, 2, K); case 3: BMPCFG_K_SWITCH(PREC, 8, NVAL, 3, 3, K); case 4: BMPCFG_K_SWITCH(PREC, 8, NVAL, 3, 4, K); default: return 0; } \
    case 4: switch (gn) { case 1: BMPCFG_K_SWITCH(PREC, 8, NVAL, 4, 1, K); case 2: BMPCFG_K_SWITCH(PREC, 8, NVAL, 4, 2, K); case 3: BMPCFG_K_SWITCH(PREC, 8, NVAL, 4, 3, K); case 4: BMPCFG_K_SWITCH(PREC, 8, NVAL, 4, 4, K); default: return 0; } \
    default: return 0; \
    }

static inline int bmpcfg_emit_prec(unsigned long prec, unsigned long K,
                                   unsigned long mtile, unsigned long ntile,
                                   unsigned long gm, unsigned long gn)
{
    if (mtile != 8UL)
        return 0;

    switch (ntile)
    {
    case 16UL:
        switch (prec)
        {
        case 0: BMPCFG_GM_GN_SWITCH(0, 16, K);
        case 1: BMPCFG_GM_GN_SWITCH(1, 16, K);
        case 2: BMPCFG_GM_GN_SWITCH(2, 16, K);
        case 3: BMPCFG_GM_GN_SWITCH(3, 16, K);
        default: return 0;
        }
    case 64UL:
        switch (prec)
        {
        case 0: BMPCFG_GM_GN_SWITCH(0, 64, K);
        case 1: BMPCFG_GM_GN_SWITCH(1, 64, K);
        case 2: BMPCFG_GM_GN_SWITCH(2, 64, K);
        case 3: BMPCFG_GM_GN_SWITCH(3, 64, K);
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
    unsigned long M;
    unsigned long mtile;
    unsigned long ntile;
    int invalid_cfg;
    int64_t *compute_cycles;
    int debug_enabled;
    unsigned long emit_cfg_count;
    unsigned long load_w_count;
    unsigned long load_a_count;
    unsigned long compute_count;
    unsigned long store_count;
} bmpmm_lowp_template_ctx_t;

static inline int bmpmm_lowp_is_mytest(const char *app_tag)
{
    return app_tag &&
           app_tag[0] == 'm' &&
           app_tag[1] == 'y' &&
           app_tag[2] == 't' &&
           app_tag[3] == 'e' &&
           app_tag[4] == 's' &&
           app_tag[5] == 't' &&
           app_tag[6] == '\0';
}

static inline void bmpmm_lowp_emit_cfg(const bmpmm_template_cfg_t *cfg, unsigned long k_cfg, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->emit_cfg_count++;
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] emit_cfg_begin p=%lu k=%lu mt=%lu nt=%lu gm=%lu gn=%lu\n",
               ctx->app_tag, cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn);
    if (!bmpcfg_emit_prec(cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn))
    {
        printf("[%s] ERROR: unsupported bmpcfg tuple p=%lu k=%lu mt=%lu nt=%lu gm=%lu gn=%lu\n",
               ctx->app_tag, cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn);
        ctx->invalid_cfg = 1;
        return;
    }
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] emit_cfg_done p=%lu k=%lu\n", ctx->app_tag, cfg->prec, k_cfg);
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
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    int16_t *c = (int16_t *)C;
    if (bmpmm_lowp_is_mytest(ctx->app_tag))
    {
        const unsigned long m_tiles = bmpmm_ceil_div_ul(cfg->M, cfg->mtile);
        const unsigned long tile_elems = cfg->mtile * cfg->ntile;
        const unsigned long tile_idx = n_tile_idx * m_tiles + m_tile_idx;
        return c + tile_idx * tile_elems;
    }
    return c + (n_tile_idx * cfg->ntile) * cfg->M + (m_tile_idx * cfg->mtile);
}

static inline void bmpmm_lowp_load_w(const void *ptr, unsigned long w_slot, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->load_w_count++;
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] load_w_begin slot=%lu ptr=0x%lx\n", ctx->app_tag, w_slot, (unsigned long)ptr);
    asm volatile("bmple 0(%0), w\n\t" : : "r"(ptr) : "memory");
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] load_w_done slot=%lu\n", ctx->app_tag, w_slot);
}

static inline void bmpmm_lowp_load_a(const void *ptr, unsigned long a_slot, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->load_a_count++;
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] load_a_begin slot=%lu ptr=0x%lx\n", ctx->app_tag, a_slot, (unsigned long)ptr);
    asm volatile("bmple 0(%0), a\n\t" : : "r"(ptr) : "memory");
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] load_a_done slot=%lu\n", ctx->app_tag, a_slot);
}

static inline void bmpmm_lowp_compute(void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->compute_count++;
    int64_t start = get_cycle_count();
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] compute_begin iter=%lu\n", ctx->app_tag, dbg_idx);
    asm volatile("bmpmm\n\t" : : : "memory");
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] compute_done iter=%lu\n", ctx->app_tag, dbg_idx);
    if (ctx->compute_cycles)
        *ctx->compute_cycles += get_cycle_count() - start;
}

static inline void bmpmm_lowp_store_c(void *ptr, unsigned long a_slot, unsigned long w_slot, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    int16_t *base = (int16_t *)ptr;
    const unsigned long m_blocks = bmpmm_lowp_align_up_ul(ctx->mtile, 8UL) / 8UL;
    const unsigned long n_blocks = bmpmm_lowp_align_up_ul(ctx->ntile, 16UL) / 16UL;
    unsigned long dbg_idx = ctx->store_count++;
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] store_begin a=%lu w=%lu ptr=0x%lx\n", ctx->app_tag, a_slot, w_slot, (unsigned long)ptr);
    for (unsigned long m_block = 0; m_block < m_blocks; ++m_block)
    {
        for (unsigned long n_block = 0; n_block < n_blocks; ++n_block)
        {
            int16_t *block_ptr;
            if (bmpmm_lowp_is_mytest(ctx->app_tag))
                block_ptr = base + n_block * 16UL * ctx->mtile + m_block * 8UL;
            else
                block_ptr = base + m_block * 8UL + (n_block * 16UL) * ctx->M;
            asm volatile("bmpse 0(%0)\n\t" : : "r"(block_ptr) : "memory");
        }
    }
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] store_done a=%lu w=%lu\n", ctx->app_tag, a_slot, w_slot);
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
        .M = M,
        .mtile = cfg.mtile,
        .ntile = cfg.ntile,
        .invalid_cfg = 0,
        .compute_cycles = compute_cycles,
        .debug_enabled = bmpmm_lowp_is_mytest(app_tag),
        .emit_cfg_count = 0,
        .load_w_count = 0,
        .load_a_count = 0,
        .compute_count = 0,
        .store_count = 0,
    };
    if (ctx.debug_enabled)
        printf("[%s][DBG] launch M=%lu K=%lu N=%lu mt=%lu nt=%lu kt=%lu gm=%lu gn=%lu p=%lu planes=%lu\n",
               app_tag, M, K, N, cfg.mtile, cfg.ntile, cfg.ktile, cfg.gm, cfg.gn, cfg.prec, ctx.planes);
    int ok = bmpmm_template_execute(&cfg, &ops, a, b, c, &ctx);
    if (ctx.debug_enabled)
        printf("[%s][DBG] template_done ok=%d invalid_cfg=%d emit=%lu loadw=%lu loada=%lu compute=%lu store=%lu\n",
               app_tag, ok, ctx.invalid_cfg, ctx.emit_cfg_count, ctx.load_w_count, ctx.load_a_count,
               ctx.compute_count, ctx.store_count);
    if (!ok || ctx.invalid_cfg)
        return 0;

    return 1;
}

#endif
