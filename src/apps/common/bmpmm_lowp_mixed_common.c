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

#include "bmpcfg_dispatch.h"

static inline unsigned long bmpmm_lowp_align_up_ul(unsigned long x, unsigned long a)
{
    return ((x + a - 1) / a) * a;
}

static inline unsigned long bmpmm_planes_from_prec(unsigned long prec)
{
    return (prec == 3UL) ? 4UL : ((prec == 1UL || prec == 2UL) ? 2UL : 1UL);
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

static inline int bmpmm_lowp_is_verify_app(const char *app_tag)
{
    return (app_tag &&
            app_tag[0] == 'b' &&
            app_tag[1] == 'm' &&
            app_tag[2] == 'p' &&
            app_tag[3] == 'u' &&
            app_tag[4] == '_' &&
            app_tag[5] == 'v' &&
            app_tag[6] == 'e' &&
            app_tag[7] == 'r' &&
            app_tag[8] == 'i' &&
            app_tag[9] == 'f' &&
            app_tag[10] == 'y' &&
            app_tag[11] == '\0') ||
           (app_tag &&
            app_tag[0] == 'm' &&
            app_tag[1] == 'y' &&
            app_tag[2] == 't' &&
            app_tag[3] == 'e' &&
            app_tag[4] == 's' &&
            app_tag[5] == 't' &&
            app_tag[6] == '\0');
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
    const unsigned long n_groups_per_tile = bmpmm_ceil_div_ul(cfg->ntile, 8UL);
    const unsigned long tile_words = bmpmm_ceil_div_ul(cfg->K, 8UL) * ctx->planes * n_groups_per_tile;
    const unsigned long k_blk0 = k0 / 8UL;
    return b + (n_tile_idx * tile_words + k_blk0 * ctx->planes * n_groups_per_tile) * 8UL;
}

static inline void *bmpmm_lowp_addr_c(void *C, const bmpmm_template_cfg_t *cfg,
                                      unsigned long m_tile_idx, unsigned long n_tile_idx, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    int16_t *c = (int16_t *)C;
    if (bmpmm_lowp_is_verify_app(ctx->app_tag))
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
            if (bmpmm_lowp_is_verify_app(ctx->app_tag))
            {
                const unsigned long block_elems = 8UL * 16UL;
                block_ptr = base + (n_block * m_blocks + m_block) * block_elems;
            }
            else
                block_ptr = base + m_block * 8UL + (n_block * 16UL) * ctx->M;
            asm volatile("bmpse 0(%0)\n\t" : : "r"(block_ptr) : "memory");
        }
    }
    if (ctx->debug_enabled && dbg_idx < 8)
        printf("[%s][DBG] store_done a=%lu w=%lu\n", ctx->app_tag, a_slot, w_slot);
}

int bmpmm_lowp_mixed_matmul_with_cfg(const char *app_tag,
                                                   int16_t *c, const int8_t *a, const int8_t *b,
                                                   unsigned long M, unsigned long K, unsigned long N,
                                                   const bmpmm_exec_cfg_t *exec_cfg,
                                                   int64_t *compute_cycles)
{
    if (!exec_cfg)
        return 0;

    if (!bmpmm_exec_cfg_is_legal(exec_cfg))
    {
        if (bmpmm_lowp_is_verify_app(app_tag))
        {
            printf("[%s] ERROR: illegal exec cfg mt=%lu nt=%lu kt=%lu gm=%lu gn=%lu g=%lu p=%lu\n",
                   app_tag, exec_cfg->mtile, exec_cfg->ntile, exec_cfg->ktile,
                   exec_cfg->gm, exec_cfg->gn, bmpmm_exec_cfg_group_g(exec_cfg), exec_cfg->prec);
        }
        return 0;
    }

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
        .debug_enabled = bmpmm_lowp_is_verify_app(app_tag),
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
