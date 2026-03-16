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

#define BINARY_PLAN_CACHE_SIZE 16

#define BMPCFG_CASE(PREC, K, M, N, GM, GN) \
    case K:                                 \
        asm volatile("bmpcfg " STR(PREC) ", " STR(K) ", " STR(M) ", " STR(N) ", " STR(GM) ", " STR(GN) "\n\t"); \
        return 1

#define BMPCFG_K_SWITCH(PREC, M, N, GM, GN, K) \
    switch (K)                                  \
    {                                           \
        BMPCFG_CASE(PREC, 8, M, N, GM, GN);     \
        BMPCFG_CASE(PREC, 16, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 24, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 32, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 40, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 48, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 56, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 64, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 72, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 80, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 88, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 96, M, N, GM, GN);    \
        BMPCFG_CASE(PREC, 104, M, N, GM, GN);   \
        BMPCFG_CASE(PREC, 112, M, N, GM, GN);   \
        BMPCFG_CASE(PREC, 120, M, N, GM, GN);   \
        BMPCFG_CASE(PREC, 128, M, N, GM, GN);   \
    default:                                    \
        return 0;                               \
    }

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

// New architecture-derived K capacity model:
// 1) Config/activation side: k <= 1024 / mlane, mlane = m_tile / 4, m_tile is 16 here.
// 2) Weight side:         k <= 8192 / (n_tile * planes), n_tile is 32 here, planes=1 for binary.
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

static inline unsigned long align_even_ge2(unsigned long x)
{
    if (x < 2)
        x = 2;
    if (x & 1UL)
        x += 1;
    return x;
}

static inline unsigned long compute_virtual_m_tile_binary(unsigned long k_dim)
{
    const unsigned long planes = 1;
    unsigned long d = ceil_div_ul(k_dim, 8);
    unsigned long num = 2 * planes * d;
    unsigned long den = d * (planes - 1) + 3 * planes;
    unsigned long m_v = ceil_div_ul(num, den);
    if (m_v < 2)
        m_v = 2;
    return m_v;
}

// Internal plan type kept private to this translation unit.
typedef struct
{
    unsigned long int M;
    unsigned long int K;
    unsigned long int N;
    unsigned long int d_block;
    unsigned long int k_block;
    unsigned long int m_v;
    unsigned long int n_v;
    unsigned long int m_passes;
    unsigned long int k_passes;
    unsigned long int t_compute;
    unsigned long int t_load;
    unsigned long int t_store;
    unsigned long int t_gap;
    unsigned long int rm_group_g;
    int full_store_hidden;
    int valid;
} binary_mixed_plan_t;

typedef struct
{
    unsigned long M;
    unsigned long K;
    unsigned long N;
    binary_mixed_plan_t plan;
} binary_plan_cache_entry_t;

static binary_plan_cache_entry_t g_binary_plan_cache[BINARY_PLAN_CACHE_SIZE];
static unsigned long g_binary_plan_cache_count = 0;

static inline unsigned long model_t_compute_binary(unsigned long m_v,
                                                   unsigned long d_block)
{
    return (m_v * (d_block + 3)) / 2;
}

static inline unsigned long model_t_load_binary(unsigned long m_v,
                                                unsigned long d_block)
{
    return ((m_v + 2) * d_block) / 2;
}

static inline unsigned long model_t_store_binary(unsigned long m_v,
                                                 unsigned long n_v)
{
    return ceil_div_ul(m_v * n_v * 16, 128);
}

static inline unsigned long runtime_supported_rm_group_g(void)
{
    // Current RTL still shows g=2 deadlock in long runs; keep runtime grouping
    // conservative so the full application flow remains executable.
    return 1;
}

static inline int binary_plan_matches_shape(const binary_plan_cache_entry_t *entry,
                                            unsigned long M,
                                            unsigned long K,
                                            unsigned long N)
{
    return entry->plan.valid && entry->M == M && entry->K == K && entry->N == N;
}

static int binary_plan_cache_lookup(unsigned long M,
                                    unsigned long K,
                                    unsigned long N,
                                    binary_mixed_plan_t *plan)
{
    for (unsigned long i = 0; i < g_binary_plan_cache_count; ++i)
    {
        if (binary_plan_matches_shape(&g_binary_plan_cache[i], M, K, N))
        {
            if (plan)
                *plan = g_binary_plan_cache[i].plan;
            return 1;
        }
    }
    return 0;
}

static void binary_plan_cache_insert(unsigned long M,
                                     unsigned long K,
                                     unsigned long N,
                                     const binary_mixed_plan_t *plan)
{
    unsigned long slot = 0;

    if (g_binary_plan_cache_count < BINARY_PLAN_CACHE_SIZE)
    {
        slot = g_binary_plan_cache_count++;
    }
    else
    {
        slot = (M ^ K ^ N) % BINARY_PLAN_CACHE_SIZE;
    }

    g_binary_plan_cache[slot].M = M;
    g_binary_plan_cache[slot].K = K;
    g_binary_plan_cache[slot].N = N;
    g_binary_plan_cache[slot].plan = *plan;
}

static binary_mixed_plan_t choose_binary_plan(unsigned long M,
                                              unsigned long K,
                                              unsigned long N)
{
    const unsigned long n_v = 16;
    const unsigned long d_rem = ceil_div_ul(K, 8);
    const unsigned long k_cap = binary_k_capacity(32, 1, 16);
    const unsigned long d_cap = min_ul(d_rem, k_cap / 8);

    binary_mixed_plan_t best_plan = {0};
    const unsigned long g_cap = runtime_supported_rm_group_g();
    unsigned long best_overflow = ~0UL;
    unsigned long best_work = 0;
    unsigned long best_cycles = 1;

    for (unsigned long d = d_cap; d >= 1; --d)
    {
        unsigned long m_cap = 64 / d;
        unsigned long m_seed = align_even_ge2(compute_virtual_m_tile_binary(d * 8));
        unsigned long m_limit = min_ul(M, m_cap);

        if (m_cap < 2)
            continue;
        if (m_limit < 2)
            continue;

        for (unsigned long m_v = 2; m_v <= m_limit; m_v += 2)
        {
            unsigned long tc = model_t_compute_binary(m_v, d);
            unsigned long tl = model_t_load_binary(m_v, d);
            unsigned long ts = model_t_store_binary(m_v, n_v);
            unsigned long gap = (tc >= tl) ? (tc - tl) : 0;
            unsigned long overflow = (gap >= ts) ? 0 : (ts - gap);
            unsigned long work = m_v * n_v * d * 8;
            int prefer = 0;

            if (!best_plan.valid)
            {
                prefer = 1;
            }
            else if (overflow < best_overflow)
            {
                prefer = 1;
            }
            else if (overflow == best_overflow)
            {
                unsigned long lhs = work * best_cycles;
                unsigned long rhs = best_work * tc;
                if (lhs > rhs)
                {
                    prefer = 1;
                }
                else if (lhs == rhs)
                {
                    unsigned long dist_cur = (m_v > m_seed) ? (m_v - m_seed) : (m_seed - m_v);
                    unsigned long dist_best = (best_plan.m_v > m_seed) ? (best_plan.m_v - m_seed) : (m_seed - best_plan.m_v);
                    if (dist_cur < dist_best ||
                        (dist_cur == dist_best && d > best_plan.d_block))
                    {
                        prefer = 1;
                    }
                }
            }

            if (prefer)
            {
                best_overflow = overflow;
                best_work = work;
                best_cycles = (tc == 0) ? 1 : tc;

                best_plan.M = M;
                best_plan.K = K;
                best_plan.N = N;
                best_plan.d_block = d;
                best_plan.k_block = d * 8;
                best_plan.m_v = m_v;
                best_plan.n_v = n_v;
                best_plan.m_passes = ceil_div_ul(M, m_v);
                best_plan.k_passes = ceil_div_ul(K, d * 8);
                best_plan.t_compute = tc;
                best_plan.t_load = tl;
                best_plan.t_store = ts;
                best_plan.t_gap = gap;
                // Use grouped reuse only when hardware advertises multi-context support.
                best_plan.rm_group_g = (g_cap >= 2 && best_plan.m_passes >= 2) ? 2 : 1;
                best_plan.full_store_hidden = (overflow == 0);
                best_plan.valid = 1;
            }
        }

        if (d == 1)
            break;
    }

    return best_plan;
}

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

int64_t mixed_compute_time = 0;

typedef struct
{
    unsigned long k_cap_runtime;
    unsigned long mtile_runtime;
    unsigned long ntile_runtime;
    int invalid_cfg;
    int64_t *compute_cycles;
} binary_template_ctx_t;

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

int binary_mixed_matmul_with_cfg(int16_t *c, const int8_t *a, const int8_t *b,
                                 const unsigned long int M, const unsigned long int K,
                                 const unsigned long int N,
                                 const binary_exec_cfg_t *exec_cfg)
{
    int64_t mixed_compute_time_local = 0;
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
        .compute_cycles = &mixed_compute_time_local,
    };

    if (!bmpmm_template_execute(&cfg, &ops, a, b, c, &ctx) || ctx.invalid_cfg)
        return 0;

    mixed_compute_time += mixed_compute_time_local;
    return 1;
}
