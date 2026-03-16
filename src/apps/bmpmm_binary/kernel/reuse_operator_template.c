#include <stdint.h>
#include <stddef.h>

__attribute__((weak)) void qloma_bmpcfg_dispatch(int prec, int ktile, int mtile,
                                                 int ntile, int gm, int gn)
{
    (void)prec;
    (void)ktile;
    (void)mtile;
    (void)ntile;
    (void)gm;
    (void)gn;
}

__attribute__((weak)) void qloma_bmple_load_a(uintptr_t ptr, int slot)
{
    (void)ptr;
    (void)slot;
}

__attribute__((weak)) void qloma_bmple_load_w(uintptr_t ptr, int slot)
{
    (void)ptr;
    (void)slot;
}

__attribute__((weak)) void qloma_bmpmm_compute(void)
{
}

__attribute__((weak)) void qloma_bmpse_store(uintptr_t ptr, int a_slot, int w_slot)
{
    (void)ptr;
    (void)a_slot;
    (void)w_slot;
}

/*
 * Reuse-aware GEMM operator template.
 *
 * Key meanings:
 * - g = gm * gn reuse-space size
 * - gm: activation reuse group count (rows-side grouping)
 * - gn: weight reuse group count (cols-side grouping)
 * - mtile, ntile: logical tile size configured by bmpcfg
 *
 * Assumptions aligned with your architecture discussion:
 * - A/W data live in VRF banks and are reused there.
 * - URS stores output partial sums in output-stationary mode.
 * - URS lifecycle is hardware-managed (software does not control URS directly).
 */

typedef struct
{
    int g;
    int gm;
    int gn;
    int mtile;
    int ntile;
    int ktile;
    int prec;
} reuse_cfg_t;

static inline int ceil_div_i(int a, int b)
{
    return (a + b - 1) / b;
}

static inline int min_i(int a, int b)
{
    return a < b ? a : b;
}

/*
 * Hooks for platform-specific address mapping.
 * Replace these with your real tensor layout mapping functions.
 */
static inline uintptr_t addr_A(const void *A, int m0, int k0)
{
    (void)A;
    (void)m0;
    (void)k0;
    return 0;
}

static inline uintptr_t addr_B(const void *B, int k0, int n0)
{
    (void)B;
    (void)k0;
    (void)n0;
    return 0;
}

static inline uintptr_t addr_C(void *C, int m0, int n0)
{
    (void)C;
    (void)m0;
    (void)n0;
    return 0;
}

/*
 * ISA wrappers: keep one place to adapt when assembler format changes.
 */
static inline void isa_bmpcfg(int prec, int ktile, int mtile, int ntile)
{
    qloma_bmpcfg_dispatch(prec, ktile, mtile, ntile, 1, 1);
}

static inline void isa_bmple(uintptr_t a_ptr, uintptr_t b_ptr, int a_slot, int w_slot)
{
    qloma_bmple_load_a(a_ptr, a_slot);
    qloma_bmple_load_w(b_ptr, w_slot);
}

static inline void isa_bmpmm(void)
{
    qloma_bmpmm_compute();
}

static inline void isa_bmpse(uintptr_t c_ptr)
{
    qloma_bmpse_store(c_ptr, 0, 0);
}

/*
 * Execute one GEMM with explicit (gm, gn, mtile, ntile) schedule.
 *
 * Software controls:
 * - loop order and grouping (gm/gn)
 * - instruction emission order
 *
 * Hardware controls automatically:
 * - URS allocate/init/accumulate/writeback timing
 * - VRF bank-pair placement and conflict handling
 */
void gemm_reuse_template(
    const void *A,
    const void *B,
    void *C,
    int M,
    int N,
    int K,
    const reuse_cfg_t *cfg)
{
    const int mt = cfg->mtile;
    const int nt = cfg->ntile;
    const int kt = cfg->ktile;
    const int gm = cfg->gm;
    const int gn = cfg->gn;

    const int m_tiles = ceil_div_i(M, mt);
    const int n_tiles = ceil_div_i(N, nt);
    const int k_tiles = ceil_div_i(K, kt);

    /* One-time tile geometry configuration (can also be hoisted by kernel launcher). */
    isa_bmpcfg(cfg->prec, kt, mt, nt);

    for (int kti = 0; kti < k_tiles; ++kti)
    {
        const int k0 = kti * kt;
        const int k_len = min_i(kt, K - k0);
        (void)k_len;

        for (int mg = 0; mg < m_tiles; mg += gm)
        {
            const int mg_len = min_i(gm, m_tiles - mg);

            for (int ng = 0; ng < n_tiles; ng += gn)
            {
                const int ng_len = min_i(gn, n_tiles - ng);

                /*
                 * 1) Load/prepare grouped A/W blocks into VRF slots.
                 * 2) Compute all pair combinations in this group.
                 */
                for (int ai = 0; ai < mg_len; ++ai)
                {
                    const int m_idx = mg + ai;
                    const int m0 = m_idx * mt;

                    for (int wi = 0; wi < ng_len; ++wi)
                    {
                        const int n_idx = ng + wi;
                        const int n0 = n_idx * nt;

                        uintptr_t a_ptr = addr_A(A, m0, k0);
                        uintptr_t b_ptr = addr_B(B, k0, n0);

                        /* Load/map one (A slot, W slot) working set to VRF bank pairs. */
                        isa_bmple(a_ptr, b_ptr, ai, wi);

                        /* Trigger compute; output stays in URS and accumulates automatically. */
                        isa_bmpmm();
                    }
                }
            }
        }
    }

    /*
     * Store stage per output tile.
     * Hardware decides when URS is complete and flushes final accumulated results.
     */
    for (int mt_i = 0; mt_i < m_tiles; ++mt_i)
    {
        for (int nt_i = 0; nt_i < n_tiles; ++nt_i)
        {
            uintptr_t c_ptr = addr_C(C, mt_i * mt, nt_i * nt);
            isa_bmpse(c_ptr);
        }
    }
}
