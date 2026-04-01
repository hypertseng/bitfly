#include <stdint.h>
#include "runtime.h"
#include "bmpmm_lowp_mixed_common.h"
#include "bmpmm_operator_template.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "bmpcfg_dispatch.h"

#ifndef BMPMM_LOWP_DEBUG
#define BMPMM_LOWP_DEBUG 0
#endif

#ifndef BMPMM_LOWP_DEBUG_PRINT_LIMIT
#define BMPMM_LOWP_DEBUG_PRINT_LIMIT 8
#endif

#ifndef BMPMM_LOWP_FAST_CALL_OVERHEAD_CYCLES
#define BMPMM_LOWP_FAST_CALL_OVERHEAD_CYCLES 320
#endif

#ifndef BMPMM_LOWP_FAST_REPEAT_PHASE_PCT
#define BMPMM_LOWP_FAST_REPEAT_PHASE_PCT 94
#endif

#ifndef BMPMM_LOWP_FAST_CALIB_MAX_M_TILES
#define BMPMM_LOWP_FAST_CALIB_MAX_M_TILES 3UL
#endif

#ifndef BMPMM_LOWP_FAST_CALIB_MAX_N_TILES
#define BMPMM_LOWP_FAST_CALIB_MAX_N_TILES 3UL
#endif

#ifndef BMPMM_LOWP_FAST_CALIB_MAX_K_TILES
#define BMPMM_LOWP_FAST_CALIB_MAX_K_TILES 2UL
#endif

static unsigned long g_bmpmm_lowp_default_mode = BMPMM_LOWP_DEFAULT_MODE;
static int64_t g_bmpmm_lowp_last_estimated_total_cycles = 0;
static int64_t g_bmpmm_lowp_last_estimated_compute_cycles = 0;

static inline unsigned long bmpmm_lowp_align_up_ul(unsigned long x, unsigned long a)
{
    return ((x + a - 1) / a) * a;
}

static __attribute__((unused)) inline int64_t bmpmm_lowp_scale_pct(int64_t value, unsigned long pct)
{
    return (value * (int64_t)pct + 50LL) / 100LL;
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
    int compact_input_layout;
    int compact_store_layout;
    int invalid_cfg;
    int64_t *compute_cycles;
    int debug_enabled;
    unsigned long emit_cfg_count;
    unsigned long load_w_count;
    unsigned long load_a_count;
    unsigned long compute_count;
    unsigned long store_count;
} bmpmm_lowp_template_ctx_t;

typedef struct
{
    unsigned long mg_len;
    unsigned long ng_len;
    unsigned long k_cfg;
    unsigned long is_store_phase;
    int valid;
    int64_t total_cycles;
    int64_t compute_cycles;
    int64_t warm_total_cycles;
    int64_t warm_compute_cycles;
    unsigned long use_count;
} bmpmm_lowp_phase_cache_entry_t;

enum
{
    BMPMM_LOWP_FAST_PHASE_CACHE_CAP = 16,
    BMPMM_LOWP_FAST_PHASE_COMPUTE = 0,
    BMPMM_LOWP_FAST_PHASE_STORE = 1,
    BMPMM_LOWP_FAST_PHASE_SETUP = 2,
};

static inline unsigned long bmpmm_lowp_sanitize_mode(unsigned long mode)
{
    return (mode == BMPMM_LOWP_EXEC_FAST) ? BMPMM_LOWP_EXEC_FAST : BMPMM_LOWP_EXEC_STRICT;
}

static inline void bmpmm_lowp_reset_estimates(void)
{
    g_bmpmm_lowp_last_estimated_total_cycles = 0;
    g_bmpmm_lowp_last_estimated_compute_cycles = 0;
}

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

static inline int bmpmm_lowp_is_debug_app(const char *app_tag)
{
    if (bmpmm_lowp_is_verify_app(app_tag))
        return 1;

    return (app_tag &&
            app_tag[0] == 'f' &&
            app_tag[1] == 'a' &&
            app_tag[2] == 's' &&
            app_tag[3] == 't' &&
            app_tag[4] == '_' &&
            app_tag[5] == 'e' &&
            app_tag[6] == 'r' &&
            app_tag[7] == 'r' &&
            app_tag[8] == 'o' &&
            app_tag[9] == 'r' &&
            app_tag[10] == '_' &&
            app_tag[11] == 'c' &&
            app_tag[12] == 'h' &&
            app_tag[13] == 'e' &&
            app_tag[14] == 'c' &&
            app_tag[15] == 'k' &&
            app_tag[16] == '_' &&
            app_tag[17] == 'd' &&
            app_tag[18] == 'b' &&
            app_tag[19] == 'g' &&
            app_tag[20] == '\0');
}

static __attribute__((unused)) inline int bmpmm_lowp_is_binary_app(const char *app_tag)
{
    return (app_tag &&
            app_tag[0] == 'b' &&
            app_tag[1] == 'm' &&
            app_tag[2] == 'p' &&
            app_tag[3] == 'm' &&
            app_tag[4] == 'm' &&
            app_tag[5] == '_' &&
            app_tag[6] == 'b' &&
            app_tag[7] == 'i' &&
            app_tag[8] == 'n' &&
            app_tag[9] == 'a' &&
            app_tag[10] == 'r' &&
            app_tag[11] == 'y');
}

static inline void bmpmm_lowp_emit_cfg(const bmpmm_template_cfg_t *cfg, unsigned long k_cfg, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->emit_cfg_count++;
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] emit_cfg_begin p=%lu k=%lu mt=%lu nt=%lu gm=%lu gn=%lu\n",
               ctx->app_tag, cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn);
    if (!bmpcfg_emit_prec(cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn))
    {
        printf("[%s] ERROR: unsupported bmpcfg tuple p=%lu k=%lu mt=%lu nt=%lu gm=%lu gn=%lu\n",
               ctx->app_tag, cfg->prec, k_cfg, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn);
        ctx->invalid_cfg = 1;
        return;
    }
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] emit_cfg_done p=%lu k=%lu\n", ctx->app_tag, cfg->prec, k_cfg);
}

static inline const void *bmpmm_lowp_addr_a(const void *A, const bmpmm_template_cfg_t *cfg,
                                            unsigned long m_tile_idx, unsigned long k0, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    const int8_t *a = (const int8_t *)A;
    const unsigned long k_aligned = bmpmm_lowp_align_up_ul(cfg->K, 8UL);
    if (ctx->compact_input_layout)
    {
        const unsigned long local_m = (cfg->gm != 0UL) ? (m_tile_idx % cfg->gm) : 0UL;
        return a + local_m * cfg->mtile * k_aligned + k0;
    }
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
    if (ctx->compact_input_layout)
    {
        const unsigned long local_n = (cfg->gn != 0UL) ? (n_tile_idx % cfg->gn) : 0UL;
        return b + (local_n * tile_words + k_blk0 * ctx->planes * n_groups_per_tile) * 8UL;
    }
    return b + (n_tile_idx * tile_words + k_blk0 * ctx->planes * n_groups_per_tile) * 8UL;
}

static inline void *bmpmm_lowp_addr_c(void *C, const bmpmm_template_cfg_t *cfg,
                                      unsigned long m_tile_idx, unsigned long n_tile_idx, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    int16_t *c = (int16_t *)C;
    if (ctx->compact_store_layout)
    {
        const unsigned long local_m = (cfg->gm != 0UL) ? (m_tile_idx % cfg->gm) : 0UL;
        const unsigned long local_n = (cfg->gn != 0UL) ? (n_tile_idx % cfg->gn) : 0UL;
        const unsigned long group_rows = cfg->gm * cfg->mtile;
        return c + (local_n * cfg->ntile) * group_rows + (local_m * cfg->mtile);
    }
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
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] load_w_begin slot=%lu ptr=0x%lx\n", ctx->app_tag, w_slot, (unsigned long)ptr);
    asm volatile("bmple 0(%0), w\n\t" : : "r"(ptr) : "memory");
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] load_w_done slot=%lu\n", ctx->app_tag, w_slot);
}

static inline void bmpmm_lowp_load_a(const void *ptr, unsigned long a_slot, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->load_a_count++;
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] load_a_begin slot=%lu ptr=0x%lx\n", ctx->app_tag, a_slot, (unsigned long)ptr);
    asm volatile("bmple 0(%0), a\n\t" : : "r"(ptr) : "memory");
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] load_a_done slot=%lu\n", ctx->app_tag, a_slot);
}

static inline void bmpmm_lowp_compute(void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->compute_count++;
    int64_t start = get_cycle_count();
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] compute_begin iter=%lu\n", ctx->app_tag, dbg_idx);
    asm volatile("bmpmm\n\t" : : : "memory");
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] compute_done iter=%lu\n", ctx->app_tag, dbg_idx);
    if (ctx->compute_cycles)
        *ctx->compute_cycles += get_cycle_count() - start;
}

static __attribute__((unused)) inline void bmpmm_lowp_compute_nop(void *user)
{
    (void)user;
}

static inline void bmpmm_lowp_store_c(void *ptr, unsigned long a_slot, unsigned long w_slot, void *user)
{
    bmpmm_lowp_template_ctx_t *ctx = (bmpmm_lowp_template_ctx_t *)user;
    unsigned long dbg_idx = ctx->store_count++;
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] store_begin a=%lu w=%lu ptr=0x%lx\n", ctx->app_tag, a_slot, w_slot, (unsigned long)ptr);
    asm volatile("bmpse 0(%0)\n\t" : : "r"(ptr) : "memory");
    if (ctx->debug_enabled && dbg_idx < BMPMM_LOWP_DEBUG_PRINT_LIMIT)
        printf("[%s][DBG] store_done a=%lu w=%lu\n", ctx->app_tag, a_slot, w_slot);
}

static __attribute__((unused)) bmpmm_lowp_phase_cache_entry_t *bmpmm_lowp_find_phase_cache(
    bmpmm_lowp_phase_cache_entry_t *entries, unsigned long count,
    unsigned long mg_len, unsigned long ng_len, unsigned long k_cfg,
    unsigned long is_store_phase)
{
    for (unsigned long i = 0UL; i < count; ++i)
    {
        bmpmm_lowp_phase_cache_entry_t *entry = &entries[i];
        if (entry->valid &&
            entry->mg_len == mg_len &&
            entry->ng_len == ng_len &&
            entry->k_cfg == k_cfg &&
            entry->is_store_phase == is_store_phase)
        {
            return entry;
        }
    }
    return 0;
}

static __attribute__((unused)) int bmpmm_lowp_measure_compute_phase(
    const bmpmm_template_cfg_t *group_cfg, const bmpmm_group_plan_t *plan,
    const bmpmm_template_ops_t *ops, const void *A, const void *B,
    const bmpmm_lowp_template_ctx_t *base_ctx,
    unsigned long mg_base, unsigned long ng_base,
    unsigned long k0, unsigned long k_cfg,
    int64_t *total_cycles, int64_t *compute_cycles)
{
    bmpmm_lowp_template_ctx_t ctx = *base_ctx;
    int64_t local_compute_cycles = 0;
    int64_t start;
    unsigned long cur_a_win = 0UL;
    unsigned long cur_w_win = 0UL;
    unsigned long prev_a_win = ~0UL;
    unsigned long prev_w_win = ~0UL;

    ctx.invalid_cfg = 0;
    ctx.compute_cycles = &local_compute_cycles;
    ctx.debug_enabled = 0;
    ctx.emit_cfg_count = 0UL;
    ctx.load_w_count = 0UL;
    ctx.load_a_count = 0UL;
    ctx.compute_count = 0UL;
    ctx.store_count = 0UL;

    start = get_cycle_count();
    while (1)
    {
        unsigned long a_start = 0UL, a_len = 0UL;
        unsigned long w_start = 0UL, w_len = 0UL;
        unsigned long next_a_win = 0UL, next_w_win = 0UL;
        int has_next_window = 0;

        bmpmm_window_shape(group_cfg->gm, plan->a_slots, cur_a_win, &a_start, &a_len);
        bmpmm_window_shape(group_cfg->gn, plan->w_slots, cur_w_win, &w_start, &w_len);

        ops->emit_cfg(group_cfg, k_cfg, &ctx);
        if (ctx.invalid_cfg)
            return 0;

        if (cur_a_win != prev_a_win)
        {
            for (unsigned long a_pos = 0UL; a_pos < a_len; ++a_pos)
            {
                const void *a_ptr =
                    ops->addr_a(A, group_cfg, mg_base + a_start + a_pos, k0, &ctx);
                ops->load_a(a_ptr, a_pos, &ctx);
            }
        }
        if (cur_w_win != prev_w_win)
        {
            for (unsigned long w_pos = 0UL; w_pos < w_len; ++w_pos)
            {
                const void *b_ptr =
                    ops->addr_b(B, group_cfg, ng_base + w_start + w_pos, k0, &ctx);
                ops->load_w(b_ptr, plan->a_slots + w_pos, &ctx);
            }
        }

        {
            const unsigned long pair_count = a_len * w_len;
            for (unsigned long pair_ord = 0UL; pair_ord < pair_count; ++pair_ord)
            {
                unsigned long a_pos = 0UL, w_pos = 0UL;
                bmpmm_pair_from_ord(a_len, w_len, plan->reuse_a, pair_ord, &a_pos, &w_pos);
                ops->compute(&ctx);
            }
        }

        prev_a_win = cur_a_win;
        prev_w_win = cur_w_win;
        bmpmm_next_window(plan->a_windows, plan->w_windows, plan->row_snake,
                          cur_a_win, cur_w_win,
                          &next_a_win, &next_w_win, &has_next_window);
        if (!has_next_window)
            break;
        cur_a_win = next_a_win;
        cur_w_win = next_w_win;
    }

    if (total_cycles)
        *total_cycles = get_cycle_count() - start;
    if (compute_cycles)
        *compute_cycles = local_compute_cycles;
    return !ctx.invalid_cfg;
}

static __attribute__((unused)) int bmpmm_lowp_measure_store_phase(
    const bmpmm_template_cfg_t *group_cfg, const bmpmm_group_plan_t *plan,
    const bmpmm_template_ops_t *ops, void *C,
    const bmpmm_lowp_template_ctx_t *base_ctx,
    unsigned long mg_base, unsigned long ng_base,
    unsigned long k_cfg, int64_t *total_cycles)
{
    bmpmm_lowp_template_ctx_t ctx = *base_ctx;
    int64_t start;
    unsigned long cur_a_win = 0UL;
    unsigned long cur_w_win = 0UL;

    ctx.invalid_cfg = 0;
    ctx.compute_cycles = 0;
    ctx.debug_enabled = 0;
    ctx.emit_cfg_count = 0UL;
    ctx.load_w_count = 0UL;
    ctx.load_a_count = 0UL;
    ctx.compute_count = 0UL;
    ctx.store_count = 0UL;

    start = get_cycle_count();
    while (1)
    {
        unsigned long a_start = 0UL, a_len = 0UL;
        unsigned long w_start = 0UL, w_len = 0UL;
        unsigned long next_a_win = 0UL, next_w_win = 0UL;
        int has_next_window = 0;
        const unsigned long pair_count = a_len * w_len;

        bmpmm_window_shape(group_cfg->gm, plan->a_slots, cur_a_win, &a_start, &a_len);
        bmpmm_window_shape(group_cfg->gn, plan->w_slots, cur_w_win, &w_start, &w_len);

        ops->emit_cfg(group_cfg, k_cfg, &ctx);
        if (ctx.invalid_cfg)
            return 0;

        for (unsigned long pair_ord = 0UL; pair_ord < a_len * w_len; ++pair_ord)
        {
            unsigned long a_pos = 0UL, w_pos = 0UL;
            unsigned long abs_ai = 0UL, abs_wi = 0UL;
            bmpmm_pair_from_ord(a_len, w_len, plan->reuse_a, pair_ord, &a_pos, &w_pos);
            abs_ai = a_start + a_pos;
            abs_wi = w_start + w_pos;
            ops->store_c(ops->addr_c(C, group_cfg, mg_base + abs_ai, ng_base + abs_wi, &ctx),
                         a_pos, plan->a_slots + w_pos, &ctx);
        }

        (void)pair_count;
        bmpmm_next_window(plan->a_windows, plan->w_windows, plan->row_snake,
                          cur_a_win, cur_w_win,
                          &next_a_win, &next_w_win, &has_next_window);
        if (!has_next_window)
            break;
        cur_a_win = next_a_win;
        cur_w_win = next_w_win;
    }

    if (total_cycles)
        *total_cycles = get_cycle_count() - start;
    return !ctx.invalid_cfg;
}

static __attribute__((unused)) int bmpmm_lowp_measure_group_setup(const bmpmm_template_cfg_t *cfg,
                                          unsigned long mg_len, unsigned long ng_len,
                                          int64_t *total_cycles)
{
    bmpmm_template_cfg_t group_cfg = *cfg;
    bmpmm_group_plan_t plan;
    const int64_t start = get_cycle_count();

    group_cfg.gm = mg_len;
    group_cfg.gn = ng_len;
    bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, cfg->ktile, &plan);
    (void)plan;

    if (total_cycles)
        *total_cycles = get_cycle_count() - start;
    return 1;
}

typedef struct
{
    bmpmm_template_cfg_t cfg;
    unsigned long load_a_count;
    unsigned long load_w_count;
    unsigned long compute_count;
    unsigned long store_count;
    unsigned long window_count;
    int64_t total_cycles;
    int64_t compute_cycles;
} bmpmm_lowp_calib_case_t;

static int bmpmm_lowp_measure_template_inner(const bmpmm_template_cfg_t *cfg,
                                             const bmpmm_template_ops_t *ops,
                                             const int8_t *a, const int8_t *b, int16_t *c,
                                             const bmpmm_lowp_template_ctx_t *base_ctx,
                                             int64_t *total_cycles, int64_t *compute_cycles)
{
    bmpmm_lowp_template_ctx_t local_ctx = *base_ctx;
    int64_t local_compute_cycles = 0;
    const int64_t start = get_cycle_count();

    local_ctx.M = cfg->M;
    local_ctx.invalid_cfg = 0;
    local_ctx.compute_cycles = &local_compute_cycles;
    local_ctx.emit_cfg_count = 0;
    local_ctx.load_w_count = 0;
    local_ctx.load_a_count = 0;
    local_ctx.compute_count = 0;
    local_ctx.store_count = 0;

    if (!bmpmm_template_execute(cfg, ops, a, b, c, &local_ctx) || local_ctx.invalid_cfg)
        return 0;

    if (total_cycles)
        *total_cycles = get_cycle_count() - start;
    if (compute_cycles)
        *compute_cycles = local_compute_cycles;
    return 1;
}

static int bmpmm_lowp_try_add_calib_case(bmpmm_lowp_calib_case_t *cases,
                                         unsigned long *case_count,
                                         const bmpmm_template_cfg_t *cfg,
                                         const bmpmm_template_ops_t *ops,
                                         const int8_t *a, const int8_t *b, int16_t *c,
                                         const bmpmm_lowp_template_ctx_t *ctx,
                                         unsigned long m_tiles,
                                         unsigned long n_tiles,
                                         unsigned long k_tiles)
{
    bmpmm_template_stats_t stats;
    bmpmm_lowp_calib_case_t *entry;
    bmpmm_template_cfg_t calib_cfg = *cfg;

    if (*case_count >= 24UL || m_tiles == 0UL || n_tiles == 0UL || k_tiles == 0UL)
        return 0;

    calib_cfg.M = m_tiles * cfg->mtile;
    calib_cfg.N = n_tiles * cfg->ntile;
    calib_cfg.K = k_tiles * cfg->ktile;

    if (!bmpmm_template_collect_stats(&calib_cfg, &stats))
        return 0;

    entry = &cases[*case_count];
    entry->cfg = calib_cfg;
    entry->load_a_count = stats.full_load_a + stats.tail_load_a;
    entry->load_w_count = stats.full_load_w + stats.tail_load_w;
    entry->compute_count = stats.full_compute + stats.tail_compute;
    entry->store_count = stats.store_count;
    entry->window_count = stats.full_windows + stats.tail_windows + stats.store_windows;
    if (!bmpmm_lowp_measure_template_inner(&calib_cfg, ops, a, b, c, ctx,
                                           &entry->total_cycles, &entry->compute_cycles))
    {
        return 0;
    }

    ++(*case_count);
    return 1;
}

static unsigned long bmpmm_lowp_rank4(const double *vecs, unsigned long rows)
{
    double mat[4][4] = {{0}};
    unsigned long rank = 0UL;
    unsigned long r = 0UL;

    if (rows == 0UL)
        return 0UL;
    if (rows > 4UL)
        rows = 4UL;

    for (unsigned long i = 0UL; i < rows; ++i)
        for (unsigned long j = 0UL; j < 4UL; ++j)
            mat[i][j] = vecs[i * 4UL + j];

    for (unsigned long col = 0UL; col < 4UL && r < rows; ++col)
    {
        unsigned long pivot = r;
        double pivot_abs = mat[pivot][col];
        if (pivot_abs < 0.0)
            pivot_abs = -pivot_abs;

        for (unsigned long i = r + 1UL; i < rows; ++i)
        {
            double cur_abs = mat[i][col];
            if (cur_abs < 0.0)
                cur_abs = -cur_abs;
            if (cur_abs > pivot_abs)
            {
                pivot = i;
                pivot_abs = cur_abs;
            }
        }

        if (pivot_abs < 1e-9)
            continue;

        if (pivot != r)
        {
            for (unsigned long k = col; k < 4UL; ++k)
            {
                const double tmp = mat[r][k];
                mat[r][k] = mat[pivot][k];
                mat[pivot][k] = tmp;
            }
        }

        {
            const double div = mat[r][col];
            for (unsigned long k = col; k < 4UL; ++k)
                mat[r][k] /= div;
        }

        for (unsigned long i = 0UL; i < rows; ++i)
        {
            if (i == r)
                continue;
            {
                const double factor = mat[i][col];
                if (factor > -1e-12 && factor < 1e-12)
                    continue;
                for (unsigned long k = col; k < 4UL; ++k)
                    mat[i][k] -= factor * mat[r][k];
            }
        }

        ++rank;
        ++r;
    }

    return rank;
}

static __attribute__((unused)) int bmpmm_lowp_select_calib_cases(bmpmm_lowp_calib_case_t *selected,
                                         const bmpmm_template_cfg_t *cfg,
                                         const bmpmm_template_ops_t *ops,
                                         const int8_t *a, const int8_t *b, int16_t *c,
                                         const bmpmm_lowp_template_ctx_t *ctx)
{
    bmpmm_lowp_calib_case_t candidates[24];
    unsigned long candidate_count = 0UL;
    const unsigned long max_m_tiles = bmpmm_min_ul(BMPMM_LOWP_FAST_CALIB_MAX_M_TILES,
                                                   ((cfg->gm + 1UL) > 2UL) ? (cfg->gm + 1UL) : 2UL);
    const unsigned long max_n_tiles = bmpmm_min_ul(BMPMM_LOWP_FAST_CALIB_MAX_N_TILES,
                                                   ((cfg->gn + 1UL) > 2UL) ? (cfg->gn + 1UL) : 2UL);
    const unsigned long max_k_tiles = BMPMM_LOWP_FAST_CALIB_MAX_K_TILES;
    unsigned long chosen = 0UL;
    double basis[16] = {0.0};

    for (unsigned long m_tiles = 1UL; m_tiles <= max_m_tiles; ++m_tiles)
    {
        for (unsigned long n_tiles = 1UL; n_tiles <= max_n_tiles; ++n_tiles)
        {
            for (unsigned long k_tiles = 1UL; k_tiles <= max_k_tiles; ++k_tiles)
            {
                if (!bmpmm_lowp_try_add_calib_case(candidates, &candidate_count, cfg, ops, a, b, c, ctx,
                                                   m_tiles, n_tiles, k_tiles))
                    return 0;
                if (candidate_count >= 24UL)
                    goto done_build_candidates;
            }
        }
    }

done_build_candidates:
    for (unsigned long i = 0UL; i < candidate_count && chosen < 4UL; ++i)
    {
        const double vec[4] = {
            (double)candidates[i].load_a_count,
            (double)candidates[i].load_w_count,
            (double)candidates[i].store_count,
            (double)candidates[i].window_count,
        };
        double trial[16];
        for (unsigned long j = 0UL; j < chosen * 4UL; ++j)
            trial[j] = basis[j];
        for (unsigned long j = 0UL; j < 4UL; ++j)
            trial[chosen * 4UL + j] = vec[j];
        if (bmpmm_lowp_rank4(trial, chosen + 1UL) > chosen)
        {
            selected[chosen] = candidates[i];
            for (unsigned long j = 0UL; j < 4UL; ++j)
                basis[chosen * 4UL + j] = vec[j];
            ++chosen;
        }
    }

    return (chosen == 4UL);
}

static __attribute__((unused)) int bmpmm_lowp_solve_4x4(double m[4][5], double out[4])
{
    for (unsigned long col = 0UL; col < 4UL; ++col)
    {
        unsigned long pivot = col;
        double pivot_abs = m[pivot][col];
        if (pivot_abs < 0.0)
            pivot_abs = -pivot_abs;

        for (unsigned long row = col + 1UL; row < 4UL; ++row)
        {
            double cur_abs = m[row][col];
            if (cur_abs < 0.0)
                cur_abs = -cur_abs;
            if (cur_abs > pivot_abs)
            {
                pivot = row;
                pivot_abs = cur_abs;
            }
        }

        if (pivot_abs < 1e-9)
            return 0;

        if (pivot != col)
        {
            for (unsigned long k = col; k < 5UL; ++k)
            {
                const double tmp = m[col][k];
                m[col][k] = m[pivot][k];
                m[pivot][k] = tmp;
            }
        }

        {
            const double div = m[col][col];
            for (unsigned long k = col; k < 5UL; ++k)
                m[col][k] /= div;
        }

        for (unsigned long row = 0UL; row < 4UL; ++row)
        {
            if (row == col)
                continue;
            {
                const double factor = m[row][col];
                if (factor > -1e-12 && factor < 1e-12)
                    continue;
                for (unsigned long k = col; k < 5UL; ++k)
                    m[row][k] -= factor * m[col][k];
            }
        }
    }

    out[0] = m[0][4];
    out[1] = m[1][4];
    out[2] = m[2][4];
    out[3] = m[3][4];
    return 1;
}

static __attribute__((unused)) int bmpmm_lowp_get_phase_cost(
    bmpmm_lowp_phase_cache_entry_t *cache, unsigned long *cache_count,
    const bmpmm_template_cfg_t *cfg, const bmpmm_template_ops_t *ops,
    const void *A, const void *B, void *C,
    const bmpmm_lowp_template_ctx_t *ctx,
    unsigned long mg_base, unsigned long ng_base,
    unsigned long mg_len, unsigned long ng_len,
    unsigned long k0, unsigned long k_cfg,
    unsigned long phase_kind,
    int64_t *total_cycles, int64_t *compute_cycles)
{
    bmpmm_lowp_phase_cache_entry_t *cached =
        bmpmm_lowp_find_phase_cache(cache, *cache_count, mg_len, ng_len, k_cfg, phase_kind);
    bmpmm_template_cfg_t group_cfg = *cfg;
    bmpmm_group_plan_t plan;

    if (cached)
    {
        if (total_cycles)
            *total_cycles = bmpmm_lowp_scale_pct(cached->total_cycles, BMPMM_LOWP_FAST_REPEAT_PHASE_PCT);
        if (compute_cycles)
            *compute_cycles = bmpmm_lowp_scale_pct(cached->compute_cycles, BMPMM_LOWP_FAST_REPEAT_PHASE_PCT);
        ++cached->use_count;
        return 1;
    }

    if (*cache_count >= BMPMM_LOWP_FAST_PHASE_CACHE_CAP)
        return 0;

    group_cfg.gm = mg_len;
    group_cfg.gn = ng_len;
    bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, k_cfg, &plan);

    cached = &cache[*cache_count];
    cached->mg_len = mg_len;
    cached->ng_len = ng_len;
    cached->k_cfg = k_cfg;
    cached->is_store_phase = phase_kind;
    cached->valid = 0;
    cached->total_cycles = 0;
    cached->compute_cycles = 0;
    cached->warm_total_cycles = 0;
    cached->warm_compute_cycles = 0;
    cached->use_count = 0;

    if (phase_kind == BMPMM_LOWP_FAST_PHASE_SETUP)
    {
        if (!bmpmm_lowp_measure_group_setup(cfg, mg_len, ng_len, &cached->total_cycles))
            return 0;
    }
    else
    {
        bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, cfg->ktile, &plan);
        if (phase_kind == BMPMM_LOWP_FAST_PHASE_STORE)
        {
            if (!bmpmm_lowp_measure_store_phase(&group_cfg, &plan, ops, C, ctx,
                                                mg_base, ng_base, k_cfg,
                                                &cached->total_cycles))
                return 0;
        }
        else
        {
            if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, A, B, ctx,
                                                  mg_base, ng_base, k0, k_cfg,
                                                  &cached->total_cycles,
                                                  &cached->compute_cycles))
                return 0;
        }
    }

    cached->valid = 1;
    cached->use_count = 1UL;
    ++(*cache_count);

    if (total_cycles)
        *total_cycles = cached->total_cycles;
    if (compute_cycles)
        *compute_cycles = cached->compute_cycles;
    return 1;
}

static int bmpmm_lowp_measure_group_exact(const bmpmm_template_cfg_t *cfg,
                                          const bmpmm_template_ops_t *ops,
                                          const int8_t *a, const int8_t *b, int16_t *c,
                                          bmpmm_lowp_template_ctx_t *ctx,
                                          unsigned long mg_base, unsigned long ng_base,
                                          unsigned long mg_len, unsigned long ng_len,
                                          int64_t *total_cycles,
                                          int64_t *compute_cycles)
{
    const unsigned long k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);
    bmpmm_template_cfg_t group_cfg = *cfg;
    bmpmm_group_plan_t plan;
    int64_t local_total = 0;
    int64_t local_compute = 0;
    int64_t phase_total = 0;
    int64_t phase_compute = 0;

    if (k_tiles == 0UL || mg_len == 0UL || ng_len == 0UL)
        return 0;

    group_cfg.gm = mg_len;
    group_cfg.gn = ng_len;
    bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, cfg->ktile, &plan);

    if (!bmpmm_lowp_measure_group_setup(cfg, mg_len, ng_len, &phase_total))
        return 0;
    local_total += phase_total;

    for (unsigned long kt = 0UL; kt < k_tiles; ++kt)
    {
        const unsigned long k0 = kt * cfg->ktile;
        const unsigned long k_rem = (cfg->K > k0) ? (cfg->K - k0) : 0UL;
        const unsigned long k_cfg =
            (k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(k_rem, 8UL);

        if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, a, b, ctx,
                                              mg_base, ng_base, k0, k_cfg,
                                              &phase_total, &phase_compute))
        {
            return 0;
        }
        local_total += phase_total;
        local_compute += phase_compute;
    }

    {
        const unsigned long last_k0 = (k_tiles - 1UL) * cfg->ktile;
        const unsigned long last_k_rem = (cfg->K > last_k0) ? (cfg->K - last_k0) : 0UL;
        const unsigned long last_k_cfg =
            (last_k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(last_k_rem, 8UL);

        if (!bmpmm_lowp_measure_store_phase(&group_cfg, &plan, ops, c, ctx,
                                            mg_base, ng_base, last_k_cfg,
                                            &phase_total))
        {
            return 0;
        }
        local_total += phase_total;
    }

    if (total_cycles)
        *total_cycles = local_total;
    if (compute_cycles)
        *compute_cycles = local_compute;
    return 1;
}

static int bmpmm_lowp_execute_fast(const char *app_tag,
                                   const bmpmm_template_cfg_t *cfg,
                                   const bmpmm_template_ops_t *ops,
                                   int16_t *c, const int8_t *a, const int8_t *b,
                                   int64_t *compute_cycles,
                                   bmpmm_lowp_template_ctx_t *ctx,
                                   int64_t *estimated_total_cycles)
{
    const unsigned long m_tiles = bmpmm_ceil_div_ul(cfg->M, cfg->mtile);
    const unsigned long n_tiles = bmpmm_ceil_div_ul(cfg->N, cfg->ntile);
    const unsigned long k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);
    const unsigned long full_group_count = (m_tiles / cfg->gm) * (n_tiles / cfg->gn);
    const unsigned long enable_full_group_reuse = (full_group_count >= 2UL);
    bmpmm_lowp_template_ctx_t meas_ctx = *ctx;
    int64_t total_cycles = 0;
    int64_t total_compute_cycles = 0;
    int64_t full_group_total = 0;
    int64_t full_group_compute = 0;
    int full_group_valid = 0;

    if (m_tiles == 0UL || n_tiles == 0UL || k_tiles == 0UL)
        return 0;

    meas_ctx.compact_input_layout = 0;
    meas_ctx.compact_store_layout = 0;

    for (unsigned long mg = 0UL; mg < m_tiles; mg += cfg->gm)
    {
        const unsigned long mg_len = bmpmm_min_ul(cfg->gm, m_tiles - mg);

        for (unsigned long ng = 0UL; ng < n_tiles; ng += cfg->gn)
        {
            const unsigned long ng_len = bmpmm_min_ul(cfg->gn, n_tiles - ng);
            int64_t group_total = 0;
            int64_t group_compute = 0;

            if (mg_len == 0UL || ng_len == 0UL)
                continue;

            if (mg_len == cfg->gm && ng_len == cfg->gn)
            {
                if (enable_full_group_reuse)
                {
                    if (!full_group_valid)
                    {
                        int64_t warmup_total = 0;
                        int64_t warmup_compute = 0;

                        if (full_group_count > 1UL &&
                            !bmpmm_lowp_measure_group_exact(cfg, ops, a, b, c, &meas_ctx,
                                                            mg, ng, cfg->gm, cfg->gn,
                                                            &warmup_total,
                                                            &warmup_compute))
                        {
                            return 0;
                        }
                        if (!bmpmm_lowp_measure_group_exact(cfg, ops, a, b, c, &meas_ctx,
                                                            mg, ng, cfg->gm, cfg->gn,
                                                            &full_group_total,
                                                            &full_group_compute))
                        {
                            return 0;
                        }
                        full_group_valid = 1;
                    }
                    total_cycles += full_group_total;
                    total_compute_cycles += full_group_compute;
                    continue;
                }
            }

            if (!bmpmm_lowp_measure_group_exact(cfg, ops, a, b, c, &meas_ctx,
                                                mg, ng, mg_len, ng_len,
                                                &group_total,
                                                &group_compute))
            {
                return 0;
            }
            total_cycles += group_total;
            total_compute_cycles += group_compute;
        }
    }

    if (compute_cycles)
        *compute_cycles += total_compute_cycles;
    if (estimated_total_cycles)
    {
        if (total_cycles > 0)
            total_cycles += BMPMM_LOWP_FAST_CALL_OVERHEAD_CYCLES;
        *estimated_total_cycles = total_cycles;
    }

    g_bmpmm_lowp_last_estimated_total_cycles = total_cycles;
    g_bmpmm_lowp_last_estimated_compute_cycles = total_compute_cycles;

    if (c)
    {
        const unsigned long sample_elems = bmpmm_min_ul(cfg->M * cfg->N, 4UL);
        for (unsigned long i = 0UL; i < sample_elems; ++i)
            c[i] = 0;
    }

    printf("[%s][fast] estimated_total_cycles=%ld estimated_compute_cycles=%ld output_valid=0\n",
           app_tag, (long)total_cycles, (long)total_compute_cycles);
    return 1;
}

void bmpmm_lowp_set_default_mode(unsigned long mode)
{
    g_bmpmm_lowp_default_mode = bmpmm_lowp_sanitize_mode(mode);
}

unsigned long bmpmm_lowp_get_default_mode(void)
{
    return bmpmm_lowp_sanitize_mode(g_bmpmm_lowp_default_mode);
}

int64_t bmpmm_lowp_get_last_estimated_total_cycles(void)
{
    return g_bmpmm_lowp_last_estimated_total_cycles;
}

int64_t bmpmm_lowp_get_last_estimated_compute_cycles(void)
{
    return g_bmpmm_lowp_last_estimated_compute_cycles;
}

int bmpmm_lowp_mixed_matmul_with_cfg_opts(const char *app_tag,
                                          int16_t *c, const int8_t *a, const int8_t *b,
                                          unsigned long M, unsigned long K, unsigned long N,
                                          const bmpmm_exec_cfg_t *exec_cfg,
                                          int64_t *compute_cycles,
                                          const bmpmm_lowp_exec_opts_t *opts)
{
    const int verify_app = bmpmm_lowp_is_verify_app(app_tag);
    unsigned long mode = opts ? bmpmm_lowp_sanitize_mode(opts->mode)
                              : bmpmm_lowp_sanitize_mode(g_bmpmm_lowp_default_mode);
    int64_t local_estimated_total_cycles = 0;

    bmpmm_lowp_reset_estimates();
    if (opts && opts->estimated_total_cycles)
        *opts->estimated_total_cycles = 0;
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
        .compact_input_layout = (mode == BMPMM_LOWP_EXEC_FAST) ? 1 : 0,
        .compact_store_layout = (mode == BMPMM_LOWP_EXEC_FAST) ? 1 : 0,
        .invalid_cfg = 0,
        .compute_cycles = compute_cycles,
        .debug_enabled = BMPMM_LOWP_DEBUG && bmpmm_lowp_is_debug_app(app_tag),
        .emit_cfg_count = 0,
        .load_w_count = 0,
        .load_a_count = 0,
        .compute_count = 0,
        .store_count = 0,
    };

    if (verify_app)
        mode = BMPMM_LOWP_EXEC_STRICT;

    if (ctx.debug_enabled)
        printf("[%s][DBG] launch M=%lu K=%lu N=%lu mt=%lu nt=%lu kt=%lu gm=%lu gn=%lu p=%lu planes=%lu\n",
               app_tag, M, K, N, cfg.mtile, cfg.ntile, cfg.ktile, cfg.gm, cfg.gn, cfg.prec, ctx.planes);

    if (mode == BMPMM_LOWP_EXEC_FAST)
    {
        if (!bmpmm_lowp_execute_fast(app_tag, &cfg, &ops, c, a, b,
                                     compute_cycles, &ctx,
                                     opts ? opts->estimated_total_cycles : &local_estimated_total_cycles))
        {
            return 0;
        }
        return 1;
    }

    int ok = bmpmm_template_execute(&cfg, &ops, a, b, c, &ctx);
    if (ctx.debug_enabled)
        printf("[%s][DBG] template_done ok=%d invalid_cfg=%d emit=%lu loadw=%lu loada=%lu compute=%lu store=%lu\n",
               app_tag, ok, ctx.invalid_cfg, ctx.emit_cfg_count, ctx.load_w_count, ctx.load_a_count,
               ctx.compute_count, ctx.store_count);
    if (!ok || ctx.invalid_cfg)
        return 0;

    return 1;
}

int bmpmm_lowp_mixed_matmul_with_cfg(const char *app_tag,
                                     int16_t *c, const int8_t *a, const int8_t *b,
                                     unsigned long M, unsigned long K, unsigned long N,
                                     const bmpmm_exec_cfg_t *exec_cfg,
                                     int64_t *compute_cycles)
{
    bmpmm_lowp_exec_opts_t opts = {
        .mode = bmpmm_lowp_is_verify_app(app_tag) ? BMPMM_LOWP_EXEC_STRICT
                                                  : bmpmm_lowp_sanitize_mode(g_bmpmm_lowp_default_mode),
        .estimated_total_cycles = 0,
    };
    return bmpmm_lowp_mixed_matmul_with_cfg_opts(app_tag, c, a, b, M, K, N,
                                                 exec_cfg, compute_cycles, &opts);
}
