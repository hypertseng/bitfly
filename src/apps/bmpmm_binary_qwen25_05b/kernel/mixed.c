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

    if (!bmpmm_template_execute(&cfg, &ops, a, b, c, &ctx) || ctx.invalid_cfg)
        return 0;

    return 1;
}
