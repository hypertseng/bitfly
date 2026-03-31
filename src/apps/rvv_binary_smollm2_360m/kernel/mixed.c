#include "mixed.h"
#include "runtime.h"
#include "../../common/bmpmm_operator_template.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

static unsigned long g_binary_default_mode = RVV_BINARY_DEFAULT_MODE;
static int64_t g_binary_last_estimated_total_cycles = 0;
static int64_t g_binary_last_estimated_compute_cycles = 0;

#define BMPCFG_CASE(PREC, K, M, N, GM, GN)     case K:                                         asm volatile("bmpcfg " STR(PREC) ", " STR(K) ", " STR(M) ", " STR(N) ", " STR(GM) ", " STR(GN) "\n\t");         return 1

#define BMPCFG_K_SWITCH(PREC, M, N, GM, GN, K)     switch (K)                                      {                                                   BMPCFG_CASE(PREC, 8, M, N, GM, GN);             BMPCFG_CASE(PREC, 16, M, N, GM, GN);            BMPCFG_CASE(PREC, 24, M, N, GM, GN);            BMPCFG_CASE(PREC, 32, M, N, GM, GN);            BMPCFG_CASE(PREC, 40, M, N, GM, GN);            BMPCFG_CASE(PREC, 48, M, N, GM, GN);            BMPCFG_CASE(PREC, 56, M, N, GM, GN);            BMPCFG_CASE(PREC, 64, M, N, GM, GN);            BMPCFG_CASE(PREC, 72, M, N, GM, GN);            BMPCFG_CASE(PREC, 80, M, N, GM, GN);            BMPCFG_CASE(PREC, 88, M, N, GM, GN);            BMPCFG_CASE(PREC, 96, M, N, GM, GN);            BMPCFG_CASE(PREC, 104, M, N, GM, GN);           BMPCFG_CASE(PREC, 112, M, N, GM, GN);           BMPCFG_CASE(PREC, 120, M, N, GM, GN);           BMPCFG_CASE(PREC, 128, M, N, GM, GN);       default:                                            return 0;                                   }

static inline unsigned long ceil_div_ul(unsigned long a, unsigned long b)
{
    return (a + b - 1) / b;
}

static inline unsigned long min_ul(unsigned long a, unsigned long b)
{
    return (a < b) ? a : b;
}

static inline unsigned long align_down_ul(unsigned long x, unsigned long a)
{
    return (x / a) * a;
}

static inline unsigned long binary_k_capacity_cfg(unsigned long m_tile)
{
    const unsigned long mlane = (m_tile >= 4) ? (m_tile / 4) : 1;
    return 1024 / mlane;
}

static inline unsigned long binary_k_capacity_wgt(unsigned long n_tile,
                                                  unsigned long prec)
{
    if (n_tile == 0)
        return 0;
    const unsigned long planes = (prec == 3) ? 4 : ((prec == 1 || prec == 2) ? 2 : 1);
    return 8192 / (n_tile * planes);
}

static inline unsigned long binary_k_capacity(unsigned long n_tile,
                                              unsigned long prec,
                                              unsigned long m_tile)
{
    unsigned long cap = min_ul(binary_k_capacity_cfg(m_tile), binary_k_capacity_wgt(n_tile, prec));
    return align_down_ul(cap, 8);
}

// Legacy auto-planner code was removed.
// The active execution path is `binary_mixed_matmul_with_cfg(...)`,
// which must use the caller-provided gm/gn directly so the runtime
// configuration matches the selected app tuple exactly.

static inline int bmpcfg_binary_emit(unsigned long int prec, unsigned long int K,
                                     unsigned long int mtile, unsigned long int ntile,
                                     unsigned long int gm, unsigned long int gn)
{
    if (prec != 0)
        return 0;

    if (mtile != 8 || ntile != 64)
        return 0;

    switch (gm)
    {
    case 1:
        switch (gn)
        {
        case 1:
            BMPCFG_K_SWITCH(0, 8, 64, 1, 1, K);
        case 2:
            BMPCFG_K_SWITCH(0, 8, 64, 1, 2, K);
        case 3:
            BMPCFG_K_SWITCH(0, 8, 64, 1, 3, K);
        case 4:
            BMPCFG_K_SWITCH(0, 8, 64, 1, 4, K);
        default:
            return 0;
        }
    case 2:
        switch (gn)
        {
        case 1:
            BMPCFG_K_SWITCH(0, 8, 64, 2, 1, K);
        case 2:
            BMPCFG_K_SWITCH(0, 8, 64, 2, 2, K);
        case 3:
            BMPCFG_K_SWITCH(0, 8, 64, 2, 3, K);
        case 4:
            BMPCFG_K_SWITCH(0, 8, 64, 2, 4, K);
        default:
            return 0;
        }
    case 3:
        switch (gn)
        {
        case 1:
            BMPCFG_K_SWITCH(0, 8, 64, 3, 1, K);
        case 2:
            BMPCFG_K_SWITCH(0, 8, 64, 3, 2, K);
        case 3:
            BMPCFG_K_SWITCH(0, 8, 64, 3, 3, K);
        case 4:
            BMPCFG_K_SWITCH(0, 8, 64, 3, 4, K);
        default:
            return 0;
        }
    case 4:
        switch (gn)
        {
        case 1:
            BMPCFG_K_SWITCH(0, 8, 64, 4, 1, K);
        case 2:
            BMPCFG_K_SWITCH(0, 8, 64, 4, 2, K);
        case 3:
            BMPCFG_K_SWITCH(0, 8, 64, 4, 3, K);
        case 4:
            BMPCFG_K_SWITCH(0, 8, 64, 4, 4, K);
        default:
            return 0;
        }
    default:
        return 0;
    }
}

static inline void bmpcfg_binary_kmn(unsigned long int K, unsigned long int mtile,
                                     unsigned long int ntile, unsigned long int gm,
                                     unsigned long int gn, unsigned long int prec)
{
    if (!bmpcfg_binary_emit(prec, K, mtile, ntile, gm, gn))
        printf("[bmpmm_binary] ERROR: unsupported bmpcfg tuple p=%lu k=%lu mt=%lu nt=%lu gm=%lu gn=%lu\n",
               prec, K, mtile, ntile, gm, gn);
}


typedef struct
{
    unsigned long k_cap_runtime;
    unsigned long mtile_runtime;
    unsigned long ntile_runtime;
    int invalid_cfg;
    int64_t *compute_cycles;
} binary_template_ctx_t;

typedef struct
{
    int64_t emit_cfg_cycles;
    int64_t load_a_cycles;
    int64_t load_w_cycles;
    int64_t compute_cycles;
    int64_t store_cycles;
} binary_fast_costs_t;

static inline unsigned long binary_sanitize_mode(unsigned long mode)
{
    return (mode == RVV_BINARY_EXEC_FAST) ? RVV_BINARY_EXEC_FAST : RVV_BINARY_EXEC_STRICT;
}

static inline void binary_reset_estimates(void)
{
    g_binary_last_estimated_total_cycles = 0;
    g_binary_last_estimated_compute_cycles = 0;
}

static void binary_emit_cfg(const bmpmm_template_cfg_t *cfg, unsigned long k_cfg, void *user)
{
    binary_template_ctx_t *ctx = (binary_template_ctx_t *)user;
    unsigned long k_eff = min_ul(k_cfg, ctx->k_cap_runtime);
    k_eff = align_down_ul(k_eff, 8);
    if (k_eff < 8)
    {
        ctx->invalid_cfg = 1;
        return;
    }
    if (cfg->mtile != ctx->mtile_runtime || cfg->ntile != ctx->ntile_runtime)
    {
        ctx->invalid_cfg = 1;
        return;
    }
    bmpcfg_binary_kmn(k_eff, cfg->mtile, cfg->ntile, cfg->gm, cfg->gn, cfg->prec);
}

static const void *binary_addr_a(const void *A, const bmpmm_template_cfg_t *cfg,
                                 unsigned long m_tile_idx, unsigned long k0, void *user)
{
    (void)user;
    const int8_t *a = (const int8_t *)A;
    const unsigned long k_aligned = ((cfg->K + 7) / 8) * 8;
    return a + m_tile_idx * cfg->mtile * k_aligned + k0;
}

static const void *binary_addr_b(const void *B, const bmpmm_template_cfg_t *cfg,
                                 unsigned long n_tile_idx, unsigned long k0, void *user)
{
    (void)user;
    const int8_t *b = (const int8_t *)B;
    const unsigned long n_groups_total = ceil_div_ul(cfg->N, 8);
    const unsigned long n_group0 = (n_tile_idx * cfg->ntile) / 8;
    const unsigned long k_blk0 = k0 / 8;
    return b + (k_blk0 * n_groups_total + n_group0) * 8;
}

static void *binary_addr_c(void *C, const bmpmm_template_cfg_t *cfg,
                           unsigned long m_tile_idx, unsigned long n_tile_idx, void *user)
{
    (void)user;
    int16_t *c = (int16_t *)C;
    return c + (n_tile_idx * cfg->ntile) * cfg->M + (m_tile_idx * cfg->mtile);
}

static void binary_load_w(const void *ptr, unsigned long w_slot, void *user)
{
    (void)user;
    (void)w_slot;
    asm volatile("bmple 0(%0), w\n\t" : : "r"(ptr) : "memory");
}

static void binary_load_a(const void *ptr, unsigned long a_slot, void *user)
{
    (void)user;
    (void)a_slot;
    asm volatile("bmple 0(%0), a\n\t" : : "r"(ptr) : "memory");
}

static void binary_compute(void *user)
{
    binary_template_ctx_t *ctx = (binary_template_ctx_t *)user;
    int64_t start = get_cycle_count();
    asm volatile("bmpmm\n\t" : : : "memory");
    if (ctx->compute_cycles)
        *ctx->compute_cycles += get_cycle_count() - start;
}

static void binary_store_c(void *ptr, unsigned long a_slot, unsigned long w_slot, void *user)
{
    (void)user;
    (void)a_slot;
    (void)w_slot;
    asm volatile("bmpse 0(%0)\n\t" : : "r"(ptr) : "memory");
}

static inline int binary_build_sample_group(const bmpmm_template_cfg_t *cfg,
                                            bmpmm_template_cfg_t *group_cfg,
                                            bmpmm_group_plan_t *plan)
{
    const unsigned long m_tiles = bmpmm_ceil_div_ul(cfg->M, cfg->mtile);
    const unsigned long n_tiles = bmpmm_ceil_div_ul(cfg->N, cfg->ntile);
    const unsigned long mg_len = min_ul(cfg->gm, m_tiles);
    const unsigned long ng_len = min_ul(cfg->gn, n_tiles);

    if (mg_len == 0UL || ng_len == 0UL)
        return 0;

    *group_cfg = *cfg;
    group_cfg->gm = mg_len;
    group_cfg->gn = ng_len;
    bmpmm_select_group_plan(group_cfg, mg_len, ng_len, cfg->ktile, plan);
    return 1;
}

static int binary_calibrate_fast_costs(const bmpmm_template_cfg_t *cfg,
                                       const bmpmm_template_ops_t *ops,
                                       const void *A, const void *B, void *C,
                                       const binary_template_ctx_t *base_ctx,
                                       unsigned long k0, unsigned long k_cfg,
                                       int measure_store,
                                       binary_fast_costs_t *costs)
{
    bmpmm_template_cfg_t group_cfg;
    bmpmm_group_plan_t plan;
    binary_template_ctx_t ctx;
    unsigned long a_start = 0UL, a_len = 0UL;
    unsigned long w_start = 0UL, w_len = 0UL;
    const void *a_ptr;
    const void *b_ptr;
    void *c_ptr;
    int64_t start;

    if (!costs || !binary_build_sample_group(cfg, &group_cfg, &plan))
        return 0;

    bmpmm_window_shape(group_cfg.gm, plan.a_slots, 0UL, &a_start, &a_len);
    bmpmm_window_shape(group_cfg.gn, plan.w_slots, 0UL, &w_start, &w_len);
    if (a_len == 0UL || w_len == 0UL)
        return 0;

    ctx = *base_ctx;
    ctx.invalid_cfg = 0;
    ctx.compute_cycles = 0;

    a_ptr = ops->addr_a(A, &group_cfg, a_start, k0, &ctx);
    b_ptr = ops->addr_b(B, &group_cfg, w_start, k0, &ctx);
    c_ptr = ops->addr_c(C, &group_cfg, a_start, w_start, &ctx);

    ops->emit_cfg(&group_cfg, k_cfg, &ctx);
    if (ctx.invalid_cfg)
        return 0;
    start = get_cycle_count();
    ops->emit_cfg(&group_cfg, k_cfg, &ctx);
    costs->emit_cfg_cycles = get_cycle_count() - start;
    if (ctx.invalid_cfg)
        return 0;

    ops->emit_cfg(&group_cfg, k_cfg, &ctx);
    start = get_cycle_count();
    ops->load_a(a_ptr, 0UL, &ctx);
    costs->load_a_cycles = get_cycle_count() - start;
    if (ctx.invalid_cfg)
        return 0;

    ops->emit_cfg(&group_cfg, k_cfg, &ctx);
    start = get_cycle_count();
    ops->load_w(b_ptr, plan.a_slots, &ctx);
    costs->load_w_cycles = get_cycle_count() - start;
    if (ctx.invalid_cfg)
        return 0;

    ops->emit_cfg(&group_cfg, k_cfg, &ctx);
    ops->load_a(a_ptr, 0UL, &ctx);
    ops->load_w(b_ptr, plan.a_slots, &ctx);
    start = get_cycle_count();
    ops->compute(&ctx);
    costs->compute_cycles = get_cycle_count() - start;
    if (ctx.invalid_cfg)
        return 0;

    costs->store_cycles = 0;
    if (measure_store)
    {
        ops->emit_cfg(&group_cfg, k_cfg, &ctx);
        ops->load_a(a_ptr, 0UL, &ctx);
        ops->load_w(b_ptr, plan.a_slots, &ctx);
        ops->compute(&ctx);
        start = get_cycle_count();
        ops->store_c(c_ptr, 0UL, plan.a_slots, &ctx);
        costs->store_cycles = get_cycle_count() - start;
        if (ctx.invalid_cfg)
            return 0;
    }

    return 1;
}

static int binary_execute_fast(const bmpmm_template_cfg_t *cfg,
                               const bmpmm_template_ops_t *ops,
                               int16_t *c, const int8_t *a, const int8_t *b,
                               binary_template_ctx_t *ctx,
                               int64_t *estimated_total_cycles)
{
    bmpmm_template_stats_t stats;
    binary_fast_costs_t full_costs = {0};
    binary_fast_costs_t tail_costs = {0};
    const unsigned long k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);
    const unsigned long last_k0 = (k_tiles - 1UL) * cfg->ktile;
    int64_t total_cycles = 0;
    int64_t total_compute_cycles = 0;
    const binary_fast_costs_t *store_costs = 0;

    if (!bmpmm_template_collect_stats(cfg, &stats))
        return 0;

    if (stats.full_compute != 0UL &&
        !binary_calibrate_fast_costs(cfg, ops, a, b, c, ctx,
                                     0UL, stats.full_k_cfg,
                                     stats.tail_present ? 0 : 1, &full_costs))
    {
        return 0;
    }

    if (stats.tail_compute != 0UL &&
        !binary_calibrate_fast_costs(cfg, ops, a, b, c, ctx,
                                     last_k0, stats.tail_k_cfg,
                                     1, &tail_costs))
    {
        return 0;
    }

    if (stats.full_compute != 0UL)
    {
        total_cycles += (int64_t)stats.full_windows * full_costs.emit_cfg_cycles;
        total_cycles += (int64_t)stats.full_load_a * full_costs.load_a_cycles;
        total_cycles += (int64_t)stats.full_load_w * full_costs.load_w_cycles;
        total_cycles += (int64_t)stats.full_compute * full_costs.compute_cycles;
        total_compute_cycles += (int64_t)stats.full_compute * full_costs.compute_cycles;
    }

    if (stats.tail_compute != 0UL)
    {
        total_cycles += (int64_t)stats.tail_windows * tail_costs.emit_cfg_cycles;
        total_cycles += (int64_t)stats.tail_load_a * tail_costs.load_a_cycles;
        total_cycles += (int64_t)stats.tail_load_w * tail_costs.load_w_cycles;
        total_cycles += (int64_t)stats.tail_compute * tail_costs.compute_cycles;
        total_compute_cycles += (int64_t)stats.tail_compute * tail_costs.compute_cycles;
    }

    store_costs = stats.tail_present ? &tail_costs : &full_costs;
    total_cycles += (int64_t)stats.store_windows * store_costs->emit_cfg_cycles;
    total_cycles += (int64_t)stats.store_count * store_costs->store_cycles;

    g_binary_last_estimated_total_cycles = total_cycles;
    g_binary_last_estimated_compute_cycles = total_compute_cycles;
    if (estimated_total_cycles)
        *estimated_total_cycles = total_cycles;

    if (c)
    {
        const unsigned long sample_elems = min_ul(cfg->M * cfg->N, 4UL);
        for (unsigned long i = 0UL; i < sample_elems; ++i)
            c[i] = 0;
    }

    printf("[rvv_binary][fast] estimated_total_cycles=%ld estimated_compute_cycles=%ld output_valid=0\n",
           (long)total_cycles, (long)total_compute_cycles);
    return 1;
}

void binary_set_default_mode(unsigned long int mode)
{
    g_binary_default_mode = binary_sanitize_mode(mode);
}

unsigned long int binary_get_default_mode(void)
{
    return binary_sanitize_mode(g_binary_default_mode);
}

int64_t binary_get_last_estimated_total_cycles(void)
{
    return g_binary_last_estimated_total_cycles;
}

int64_t binary_get_last_estimated_compute_cycles(void)
{
    return g_binary_last_estimated_compute_cycles;
}

int binary_mixed_matmul_with_cfg_opts(int16_t *c, const int8_t *a, const int8_t *b,
                                      const unsigned long int M, const unsigned long int K,
                                      const unsigned long int N,
                                      const binary_exec_cfg_t *exec_cfg,
                                      const binary_exec_opts_t *opts)
{
    unsigned long mode = opts ? binary_sanitize_mode(opts->mode)
                              : binary_sanitize_mode(g_binary_default_mode);
    int64_t local_estimated_total_cycles = 0;

    binary_reset_estimates();
    if (opts && opts->estimated_total_cycles)
        *opts->estimated_total_cycles = 0;
    if (!exec_cfg)
        return 0;

    if (exec_cfg->mtile == 0 || exec_cfg->ntile == 0 || exec_cfg->ktile == 0 ||
        exec_cfg->gm == 0 || exec_cfg->gn == 0 || exec_cfg->prec > 3)
    {
        return 0;
    }

    if ((exec_cfg->ktile & 7UL) != 0)
        return 0;

    const unsigned long k_cap_runtime = binary_k_capacity(exec_cfg->ntile, exec_cfg->prec, exec_cfg->mtile);
    if (k_cap_runtime < 8)
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
        .emit_cfg = binary_emit_cfg,
        .addr_a = binary_addr_a,
        .addr_b = binary_addr_b,
        .addr_c = binary_addr_c,
        .load_w = binary_load_w,
        .load_a = binary_load_a,
        .compute = binary_compute,
        .store_c = binary_store_c,
    };

    binary_template_ctx_t ctx = {
        .k_cap_runtime = k_cap_runtime,
        .mtile_runtime = cfg.mtile,
        .ntile_runtime = cfg.ntile,
        .invalid_cfg = 0,
        .compute_cycles = 0,
    };

    if (mode == RVV_BINARY_EXEC_FAST)
    {
        return binary_execute_fast(&cfg, &ops, c, a, b, &ctx,
                                   opts ? opts->estimated_total_cycles : &local_estimated_total_cycles);
    }

    if (!bmpmm_template_execute(&cfg, &ops, a, b, c, &ctx) || ctx.invalid_cfg)
        return 0;

    return 1;
}

int binary_mixed_matmul_with_cfg(int16_t *c, const int8_t *a, const int8_t *b,
                                 const unsigned long int M, const unsigned long int K,
                                 const unsigned long int N,
                                 const binary_exec_cfg_t *exec_cfg)
{
    binary_exec_opts_t opts = {
        .mode = binary_sanitize_mode(g_binary_default_mode),
        .estimated_total_cycles = 0,
    };
    return binary_mixed_matmul_with_cfg_opts(c, a, b, M, K, N, exec_cfg, &opts);
}
