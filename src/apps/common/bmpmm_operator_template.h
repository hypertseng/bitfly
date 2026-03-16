#ifndef BMPMM_OPERATOR_TEMPLATE_H
#define BMPMM_OPERATOR_TEMPLATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        unsigned long M;
        unsigned long K;
        unsigned long N;
        unsigned long mtile;
        unsigned long ntile;
        unsigned long ktile;
        unsigned long gm;
        unsigned long gn;
        unsigned long prec;
    } bmpmm_template_cfg_t;

    typedef struct
    {
        void (*emit_cfg)(const bmpmm_template_cfg_t *cfg, unsigned long k_cfg, void *user);
        const void *(*addr_a)(const void *A, const bmpmm_template_cfg_t *cfg,
                              unsigned long m_tile_idx, unsigned long k0, void *user);
        const void *(*addr_b)(const void *B, const bmpmm_template_cfg_t *cfg,
                              unsigned long n_tile_idx, unsigned long k0, void *user);
        void *(*addr_c)(void *C, const bmpmm_template_cfg_t *cfg,
                        unsigned long m_tile_idx, unsigned long n_tile_idx, void *user);
        void (*load_w)(const void *ptr, unsigned long w_slot, void *user);
        void (*load_a)(const void *ptr, unsigned long a_slot, void *user);
        void (*compute)(void *user);
        void (*store_c)(void *ptr, unsigned long a_slot, unsigned long w_slot, void *user);
    } bmpmm_template_ops_t;

    static inline unsigned long bmpmm_ceil_div_ul(unsigned long a, unsigned long b)
    {
        return (a + b - 1) / b;
    }

    static inline unsigned long bmpmm_min_ul(unsigned long a, unsigned long b)
    {
        return (a < b) ? a : b;
    }

    static inline unsigned long bmpmm_align_up_ul(unsigned long x, unsigned long a)
    {
        return ((x + a - 1) / a) * a;
    }

    static inline int bmpmm_template_execute(
        const bmpmm_template_cfg_t *cfg,
        const bmpmm_template_ops_t *ops,
        const void *A,
        const void *B,
        void *C,
        void *user)
    {
        if (!cfg || !ops || !ops->emit_cfg || !ops->addr_a || !ops->addr_b || !ops->addr_c ||
            !ops->load_w || !ops->load_a || !ops->compute || !ops->store_c)
        {
            return 0;
        }

        if (cfg->mtile == 0 || cfg->ntile == 0 || cfg->ktile == 0 || cfg->gm == 0 || cfg->gn == 0)
        {
            return 0;
        }

        const unsigned long m_tiles = bmpmm_ceil_div_ul(cfg->M, cfg->mtile);
        const unsigned long n_tiles = bmpmm_ceil_div_ul(cfg->N, cfg->ntile);
        const unsigned long k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);

        for (unsigned long mg = 0; mg < m_tiles; mg += cfg->gm)
        {
            const unsigned long mg_len = bmpmm_min_ul(cfg->gm, m_tiles - mg);

            for (unsigned long ng = 0; ng < n_tiles; ng += cfg->gn)
            {
                const unsigned long ng_len = bmpmm_min_ul(cfg->gn, n_tiles - ng);

                for (unsigned long kt = 0; kt < k_tiles; ++kt)
                {
                    const unsigned long k0 = kt * cfg->ktile;
                    const unsigned long k_rem = (cfg->K > k0) ? (cfg->K - k0) : 0;
                    const unsigned long k_cfg = (k_rem >= cfg->ktile)
                                                    ? cfg->ktile
                                                    : bmpmm_align_up_ul(k_rem, 8);

                    ops->emit_cfg(cfg, k_cfg, user);

                    for (unsigned long wi = 0; wi < ng_len; ++wi)
                    {
                        const unsigned long n_idx = ng + wi;
                        const void *b_ptr = ops->addr_b(B, cfg, n_idx, k0, user);
                        ops->load_w(b_ptr, wi, user);

                        for (unsigned long ai = 0; ai < mg_len; ++ai)
                        {
                            const unsigned long m_idx = mg + ai;
                            const void *a_ptr = ops->addr_a(A, cfg, m_idx, k0, user);
                            ops->load_a(a_ptr, ai, user);
                            ops->compute(user);
                        }
                    }
                }

                for (unsigned long ai = 0; ai < mg_len; ++ai)
                {
                    const unsigned long m_idx = mg + ai;
                    for (unsigned long wi = 0; wi < ng_len; ++wi)
                    {
                        const unsigned long n_idx = ng + wi;
                        void *c_ptr = ops->addr_c(C, cfg, m_idx, n_idx, user);
                        ops->store_c(c_ptr, ai, wi, user);
                    }
                }
            }
        }

        return 1;
    }

#ifdef __cplusplus
}
#endif

#endif
