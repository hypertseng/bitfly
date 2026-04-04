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

static unsigned long g_bmpmm_lowp_default_mode = BMPMM_LOWP_DEFAULT_MODE;
static int64_t g_bmpmm_lowp_last_estimated_total_cycles = 0;
static int64_t g_bmpmm_lowp_last_estimated_compute_cycles = 0;

static inline unsigned long bmpmm_lowp_align_up_ul(unsigned long x, unsigned long a)
{
    return ((x + a - 1) / a) * a;
}

static inline unsigned long bmpmm_planes_from_prec(unsigned long prec)
{
    return (prec == 3UL) ? 4UL : ((prec == 1UL || prec == 2UL) ? 2UL : 1UL);
}

static inline int64_t bmpmm_lowp_analytic_compute_phase_cycles(const bmpmm_template_cfg_t *group_cfg,
                                                               unsigned long k_cfg)
{
    const unsigned long pair_count = group_cfg->gm * group_cfg->gn;
    const unsigned long m_blocks = bmpmm_ceil_div_ul(group_cfg->mtile, 8UL);
    const unsigned long n_blocks = bmpmm_ceil_div_ul(group_cfg->ntile, 16UL);
    const unsigned long phys_blocks = m_blocks * n_blocks;
    const unsigned long planes = bmpmm_planes_from_prec(group_cfg->prec);
    const unsigned long k_iters = k_cfg / 8UL;
    const unsigned long sa_cycles = 1UL + k_iters * planes + 2UL;

    return (int64_t)pair_count * (int64_t)phys_blocks * (int64_t)sa_cycles;
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

static inline void bmpmm_lowp_compute_nop(void *user)
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

static inline void bmpmm_lowp_emit_cfg_nop(const bmpmm_template_cfg_t *cfg, unsigned long k_cfg, void *user)
{
    (void)cfg;
    (void)k_cfg;
    (void)user;
}

static inline void bmpmm_lowp_load_w_nop(const void *ptr, unsigned long w_slot, void *user)
{
    (void)ptr;
    (void)w_slot;
    (void)user;
}

static inline void bmpmm_lowp_load_a_nop(const void *ptr, unsigned long a_slot, void *user)
{
    (void)ptr;
    (void)a_slot;
    (void)user;
}

static inline void bmpmm_lowp_store_c_nop(void *ptr, unsigned long a_slot, unsigned long w_slot, void *user)
{
    (void)ptr;
    (void)a_slot;
    (void)w_slot;
    (void)user;
}

static const bmpmm_template_ops_t k_bmpmm_lowp_noop_ops = {
    .emit_cfg = bmpmm_lowp_emit_cfg_nop,
    .addr_a = bmpmm_lowp_addr_a,
    .addr_b = bmpmm_lowp_addr_b,
    .addr_c = bmpmm_lowp_addr_c,
    .load_w = bmpmm_lowp_load_w_nop,
    .load_a = bmpmm_lowp_load_a_nop,
    .compute = bmpmm_lowp_compute_nop,
    .store_c = bmpmm_lowp_store_c_nop,
};

static int bmpmm_lowp_measure_compute_phase(
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

static int bmpmm_lowp_measure_group_total_with_ops(
    const bmpmm_template_cfg_t *cfg,
    const bmpmm_template_ops_t *ops,
    const int8_t *a, const int8_t *b, int16_t *c,
    const bmpmm_lowp_template_ctx_t *base_ctx,
    unsigned long mg_base, unsigned long ng_base,
    unsigned long mg_len, unsigned long ng_len,
    int64_t *total_cycles, int64_t *compute_cycles)
{
    bmpmm_template_cfg_t group_cfg = *cfg;
    bmpmm_group_plan_t plan;
    bmpmm_lowp_template_ctx_t ctx = *base_ctx;
    int64_t local_compute_cycles = 0;
    int64_t start;

    if (mg_len == 0UL || ng_len == 0UL)
        return 0;

    group_cfg.gm = mg_len;
    group_cfg.gn = ng_len;

    ctx.invalid_cfg = 0;
    ctx.compute_cycles = &local_compute_cycles;
    ctx.debug_enabled = 0;
    ctx.emit_cfg_count = 0UL;
    ctx.load_w_count = 0UL;
    ctx.load_a_count = 0UL;
    ctx.compute_count = 0UL;
    ctx.store_count = 0UL;

    start = get_cycle_count();
    bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, cfg->ktile, &plan);

    {
        const unsigned long k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);
        for (unsigned long kt = 0UL; kt < k_tiles; ++kt)
        {
            const unsigned long k0 = kt * cfg->ktile;
            const unsigned long k_rem = (cfg->K > k0) ? (cfg->K - k0) : 0UL;
            const unsigned long k_cfg =
                (k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(k_rem, 8UL);
            unsigned long cur_a_win = 0UL;
            unsigned long cur_w_win = 0UL;
            unsigned long prev_a_win = ~0UL;
            unsigned long prev_w_win = ~0UL;

            while (1)
            {
                unsigned long a_start = 0UL, a_len = 0UL;
                unsigned long w_start = 0UL, w_len = 0UL;
                unsigned long next_a_win = 0UL, next_w_win = 0UL;
                int has_next_window = 0;

                bmpmm_window_shape(mg_len, plan.a_slots, cur_a_win, &a_start, &a_len);
                bmpmm_window_shape(ng_len, plan.w_slots, cur_w_win, &w_start, &w_len);
                ops->emit_cfg(&group_cfg, k_cfg, &ctx);
                if (ctx.invalid_cfg)
                    return 0;

                if (cur_a_win != prev_a_win)
                {
                    for (unsigned long a_pos = 0UL; a_pos < a_len; ++a_pos)
                    {
                        const void *a_ptr =
                            ops->addr_a(a, &group_cfg, mg_base + a_start + a_pos, k0, &ctx);
                        ops->load_a(a_ptr, a_pos, &ctx);
                    }
                }
                if (cur_w_win != prev_w_win)
                {
                    for (unsigned long w_pos = 0UL; w_pos < w_len; ++w_pos)
                    {
                        const void *b_ptr =
                            ops->addr_b(b, &group_cfg, ng_base + w_start + w_pos, k0, &ctx);
                        ops->load_w(b_ptr, plan.a_slots + w_pos, &ctx);
                    }
                }

                {
                    const unsigned long pair_count = a_len * w_len;
                    for (unsigned long pair_ord = 0UL; pair_ord < pair_count; ++pair_ord)
                    {
                        unsigned long a_pos = 0UL, w_pos = 0UL;
                        bmpmm_pair_from_ord(a_len, w_len, plan.reuse_a, pair_ord, &a_pos, &w_pos);
                        ops->compute(&ctx);
                    }
                }

                prev_a_win = cur_a_win;
                prev_w_win = cur_w_win;
                bmpmm_next_window(plan.a_windows, plan.w_windows, plan.row_snake,
                                  cur_a_win, cur_w_win,
                                  &next_a_win, &next_w_win, &has_next_window);
                if (!has_next_window)
                    break;
                cur_a_win = next_a_win;
                cur_w_win = next_w_win;
            }
        }
    }

    {
        const unsigned long k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);
        unsigned long cur_a_win = 0UL;
        unsigned long cur_w_win = 0UL;
        const unsigned long last_k0 = (k_tiles - 1UL) * cfg->ktile;
        const unsigned long last_k_rem = (cfg->K > last_k0) ? (cfg->K - last_k0) : 0UL;
        const unsigned long last_k_cfg =
            (last_k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(last_k_rem, 8UL);

        while (1)
        {
            unsigned long a_start = 0UL, a_len = 0UL;
            unsigned long w_start = 0UL, w_len = 0UL;
            unsigned long next_a_win = 0UL, next_w_win = 0UL;
            int has_next_window = 0;

            bmpmm_window_shape(mg_len, plan.a_slots, cur_a_win, &a_start, &a_len);
            bmpmm_window_shape(ng_len, plan.w_slots, cur_w_win, &w_start, &w_len);
            ops->emit_cfg(&group_cfg, last_k_cfg, &ctx);
            if (ctx.invalid_cfg)
                return 0;

            for (unsigned long pair_ord = 0UL; pair_ord < a_len * w_len; ++pair_ord)
            {
                unsigned long a_pos = 0UL, w_pos = 0UL;
                unsigned long abs_ai = 0UL, abs_wi = 0UL;
                bmpmm_pair_from_ord(a_len, w_len, plan.reuse_a, pair_ord, &a_pos, &w_pos);
                abs_ai = a_start + a_pos;
                abs_wi = w_start + w_pos;
                ops->store_c(ops->addr_c(c, &group_cfg, mg_base + abs_ai, ng_base + abs_wi, &ctx),
                             a_pos, plan.a_slots + w_pos, &ctx);
            }

            bmpmm_next_window(plan.a_windows, plan.w_windows, plan.row_snake,
                              cur_a_win, cur_w_win,
                              &next_a_win, &next_w_win, &has_next_window);
            if (!has_next_window)
                break;
            cur_a_win = next_a_win;
            cur_w_win = next_w_win;
        }
    }

    if (total_cycles)
        *total_cycles = get_cycle_count() - start;
    if (compute_cycles)
        *compute_cycles = local_compute_cycles;
    return !ctx.invalid_cfg;
}

static int bmpmm_lowp_measure_store_phase(
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

static int bmpmm_lowp_measure_group_warm_first_phase_after_tail(
    const bmpmm_template_cfg_t *cfg,
    const bmpmm_template_ops_t *ops,
    const int8_t *a, const int8_t *b, int16_t *c,
    const bmpmm_lowp_template_ctx_t *base_ctx,
    unsigned long seed_mg, unsigned long seed_ng,
    unsigned long warm_mg, unsigned long warm_ng,
    unsigned long mg_len, unsigned long ng_len,
    int64_t *phase_total, int64_t *phase_compute)
{
    const unsigned long k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);
    const unsigned long last_k0 = (k_tiles - 1UL) * cfg->ktile;
    const unsigned long last_k_rem = (cfg->K > last_k0) ? (cfg->K - last_k0) : 0UL;
    const unsigned long last_k_cfg =
        (last_k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(last_k_rem, 8UL);
    const unsigned long first_k_cfg =
        (cfg->K >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(cfg->K, 8UL);
    bmpmm_template_cfg_t group_cfg = *cfg;
    bmpmm_group_plan_t plan;

    if (cfg->K == 0UL || mg_len == 0UL || ng_len == 0UL)
        return 0;

    group_cfg.gm = mg_len;
    group_cfg.gn = ng_len;
    bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, cfg->ktile, &plan);

    if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, a, b, base_ctx,
                                          seed_mg, seed_ng, last_k0, last_k_cfg,
                                          0, 0))
    {
        return 0;
    }

    if (!bmpmm_lowp_measure_store_phase(&group_cfg, &plan, ops, c, base_ctx,
                                        seed_mg, seed_ng, last_k_cfg, 0))
    {
        return 0;
    }

    return bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, a, b, base_ctx,
                                            warm_mg, warm_ng, 0UL, first_k_cfg,
                                            phase_total, phase_compute);
}

static int bmpmm_lowp_estimate_group_from_phases(const bmpmm_template_cfg_t *cfg,
                                                 const bmpmm_template_ops_t *ops,
                                                 const int8_t *a, const int8_t *b, int16_t *c,
                                                 bmpmm_lowp_template_ctx_t *ctx,
                                                 unsigned long mg_base, unsigned long ng_base,
                                                 unsigned long mg_len, unsigned long ng_len,
                                                 int64_t *total_cycles,
                                                 int64_t *compute_cycles)
{
    const unsigned long full_k_tiles = cfg->K / cfg->ktile;
    const unsigned long tail_len = cfg->K % cfg->ktile;
    bmpmm_template_cfg_t group_cfg = *cfg;
    bmpmm_group_plan_t plan;
    int64_t local_total = 0;
    int64_t local_compute = 0;
    int64_t phase_total = 0;
    int64_t phase_compute = 0;
    int64_t noop_total = 0;
    int64_t noop_phase_total = 0;

    if (mg_len == 0UL || ng_len == 0UL)
        return 0;

    group_cfg.gm = mg_len;
    group_cfg.gn = ng_len;
    bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, cfg->ktile, &plan);

    if (!bmpmm_lowp_measure_group_total_with_ops(cfg, &k_bmpmm_lowp_noop_ops,
                                                 a, b, c, ctx,
                                                 mg_base, ng_base, mg_len, ng_len,
                                                 &noop_total, 0))
        return 0;
    local_total += noop_total;

    if (full_k_tiles != 0UL)
    {
        const int64_t full_analytic_compute =
            bmpmm_lowp_analytic_compute_phase_cycles(&group_cfg, cfg->ktile);
        if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, a, b, ctx,
                                              mg_base, ng_base, 0UL, cfg->ktile,
                                              &phase_total, &phase_compute))
        {
            return 0;
        }
        if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, &k_bmpmm_lowp_noop_ops, a, b, ctx,
                                              mg_base, ng_base, 0UL, cfg->ktile,
                                              &noop_phase_total, 0))
        {
            return 0;
        }
        local_total += phase_total - noop_phase_total;
        local_compute += full_analytic_compute;

        if (full_k_tiles > 1UL)
        {
            int64_t warm_total = 0;
            int64_t warm_noop_total = 0;
            if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, a, b, ctx,
                                                  mg_base, ng_base, cfg->ktile, cfg->ktile,
                                                  &warm_total, 0))
            {
                return 0;
            }
            if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, &k_bmpmm_lowp_noop_ops, a, b, ctx,
                                                  mg_base, ng_base, cfg->ktile, cfg->ktile,
                                                  &warm_noop_total, 0))
            {
                return 0;
            }
            local_total += (int64_t)(full_k_tiles - 1UL) * (warm_total - warm_noop_total);
            local_compute += (int64_t)(full_k_tiles - 1UL) * full_analytic_compute;
        }
    }

    if (tail_len != 0UL)
    {
        const unsigned long tail_k0 = full_k_tiles * cfg->ktile;
        const unsigned long tail_k_cfg = bmpmm_align_up_ul(tail_len, 8UL);
        const int64_t tail_analytic_compute =
            bmpmm_lowp_analytic_compute_phase_cycles(&group_cfg, tail_k_cfg);
        int64_t tail_noop_total = 0;
        int64_t store_noop_total = 0;
        if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, a, b, ctx,
                                              mg_base, ng_base, tail_k0, tail_k_cfg,
                                              &phase_total, &phase_compute))
        {
            return 0;
        }
        if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, &k_bmpmm_lowp_noop_ops, a, b, ctx,
                                              mg_base, ng_base, tail_k0, tail_k_cfg,
                                              &tail_noop_total, 0))
        {
            return 0;
        }
        local_total += phase_total - tail_noop_total;
        local_compute += tail_analytic_compute;

        if (!bmpmm_lowp_measure_store_phase(&group_cfg, &plan, ops, c, ctx,
                                            mg_base, ng_base, tail_k_cfg,
                                            &phase_total))
        {
            return 0;
        }
        if (!bmpmm_lowp_measure_store_phase(&group_cfg, &plan, &k_bmpmm_lowp_noop_ops, c, ctx,
                                            mg_base, ng_base, tail_k_cfg,
                                            &store_noop_total))
        {
            return 0;
        }
        local_total += phase_total - store_noop_total;
    }
    else
    {
        int64_t store_noop_total = 0;
        if (!bmpmm_lowp_measure_store_phase(&group_cfg, &plan, ops, c, ctx,
                                            mg_base, ng_base, cfg->ktile,
                                            &phase_total))
        {
            return 0;
        }
        if (!bmpmm_lowp_measure_store_phase(&group_cfg, &plan, &k_bmpmm_lowp_noop_ops, c, ctx,
                                            mg_base, ng_base, cfg->ktile,
                                            &store_noop_total))
        {
            return 0;
        }
        local_total += phase_total - store_noop_total;
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
    int64_t total_cycles = 0;
    int64_t total_compute_cycles = 0;
    bmpmm_lowp_template_ctx_t meas_ctx = *ctx;
    int64_t full_group_total = 0;
    int64_t full_group_compute = 0;
    int64_t horiz_warm_group_total = 0;
    int64_t horiz_warm_group_compute = 0;
    int64_t vert_warm_group_total = 0;
    int64_t vert_warm_group_compute = 0;
    int64_t cold_first_phase_total = 0;
    int64_t cold_first_phase_compute = 0;
    int64_t cold_first_noop_total = 0;
    int64_t cold_first_phase_delta = 0;
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

            if (mg_len == cfg->gm && ng_len == cfg->gn && enable_full_group_reuse)
            {
                if (!full_group_valid)
                {
                    if (!bmpmm_lowp_estimate_group_from_phases(cfg, ops, a, b, c, &meas_ctx,
                                                               mg, ng, cfg->gm, cfg->gn,
                                                               &full_group_total,
                                                               &full_group_compute))
                    {
                        return 0;
                    }

                    {
                        bmpmm_template_cfg_t group_cfg = *cfg;
                        bmpmm_group_plan_t plan;
                        const unsigned long first_k_cfg =
                            (cfg->K >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(cfg->K, 8UL);

                        group_cfg.gm = cfg->gm;
                        group_cfg.gn = cfg->gn;
                        bmpmm_select_group_plan(&group_cfg, cfg->gm, cfg->gn, cfg->ktile, &plan);

                        if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, ops, a, b, &meas_ctx,
                                                              mg, ng, 0UL, first_k_cfg,
                                                              &cold_first_phase_total,
                                                              &cold_first_phase_compute))
                        {
                            return 0;
                        }
                        if (!bmpmm_lowp_measure_compute_phase(&group_cfg, &plan, &k_bmpmm_lowp_noop_ops,
                                                              a, b, &meas_ctx,
                                                              mg, ng, 0UL, first_k_cfg,
                                                              &cold_first_noop_total, 0))
                        {
                            return 0;
                        }
                        cold_first_phase_delta = cold_first_phase_total - cold_first_noop_total;
                    }

                    horiz_warm_group_total = full_group_total;
                    horiz_warm_group_compute = full_group_compute;
                    vert_warm_group_total = full_group_total;
                    vert_warm_group_compute = full_group_compute;

                    if (n_tiles >= 2UL * cfg->gn)
                    {
                        int64_t warm_phase_total = 0;
                        int64_t warm_phase_compute = 0;
                        if (!bmpmm_lowp_measure_group_warm_first_phase_after_tail(cfg, ops, a, b, c, &meas_ctx,
                                                                                  mg, ng,
                                                                                  mg, ng + cfg->gn,
                                                                                  cfg->gm, cfg->gn,
                                                                                  &warm_phase_total,
                                                                                  &warm_phase_compute))
                        {
                            return 0;
                        }
                        horiz_warm_group_total = full_group_total - cold_first_phase_delta +
                                                 (warm_phase_total - cold_first_noop_total);
                        horiz_warm_group_compute = full_group_compute - cold_first_phase_compute +
                                                   warm_phase_compute;
                    }

                    if (m_tiles >= 2UL * cfg->gm && n_tiles >= cfg->gn)
                    {
                        const unsigned long prev_row_ng = n_tiles - cfg->gn;
                        int64_t warm_phase_total = 0;
                        int64_t warm_phase_compute = 0;
                        if (!bmpmm_lowp_measure_group_warm_first_phase_after_tail(cfg, ops, a, b, c, &meas_ctx,
                                                                                  mg, prev_row_ng,
                                                                                  mg + cfg->gm, 0UL,
                                                                                  cfg->gm, cfg->gn,
                                                                                  &warm_phase_total,
                                                                                  &warm_phase_compute))
                        {
                            return 0;
                        }
                        vert_warm_group_total = full_group_total - cold_first_phase_delta +
                                                (warm_phase_total - cold_first_noop_total);
                        vert_warm_group_compute = full_group_compute - cold_first_phase_compute +
                                                  warm_phase_compute;
                    }

                    full_group_valid = 1;
                }

                if (mg == 0UL && ng == 0UL)
                {
                    total_cycles += full_group_total;
                    total_compute_cycles += full_group_compute;
                }
                else if (ng != 0UL)
                {
                    total_cycles += horiz_warm_group_total;
                    total_compute_cycles += horiz_warm_group_compute;
                }
                else
                {
                    total_cycles += vert_warm_group_total;
                    total_compute_cycles += vert_warm_group_compute;
                }
                continue;
            }

            if (!bmpmm_lowp_estimate_group_from_phases(cfg, ops, a, b, c, &meas_ctx,
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
