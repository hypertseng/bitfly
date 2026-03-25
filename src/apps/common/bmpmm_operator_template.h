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

    static inline unsigned long bmpmm_weight_bits_from_prec(unsigned long prec)
    {
        switch (prec)
        {
        case 0UL:
            return 1UL;
        case 1UL:
        case 2UL:
            return 2UL;
        case 3UL:
            return 4UL;
        default:
            return 0UL;
        }
    }

    static inline void bmpmm_pair_index_to_coords(unsigned long pair_idx,
                                                  unsigned long mg_len,
                                                  unsigned long ng_len,
                                                  int reuse_a,
                                                  unsigned long *ai,
                                                  unsigned long *wi)
    {
        if (reuse_a)
        {
            *ai = pair_idx / ng_len;
            *wi = pair_idx % ng_len;
        }
        else
        {
            *wi = pair_idx / mg_len;
            *ai = pair_idx % mg_len;
        }
    }

    static inline int bmpmm_template_execute(
        const bmpmm_template_cfg_t *cfg,
        const bmpmm_template_ops_t *ops,
        const void *A,
        const void *B,
        void *C,
        void *user)
    {
        unsigned long m_tiles;
        unsigned long n_tiles;
        unsigned long k_tiles;
        unsigned long weight_bits;
        unsigned long act_bits;
        unsigned long wgt_bits;
        int reuse_a;

        if (!cfg || !ops || !ops->emit_cfg || !ops->addr_a || !ops->addr_b || !ops->addr_c ||
            !ops->load_w || !ops->load_a || !ops->compute || !ops->store_c)
        {
            return 0;
        }

        if (cfg->mtile == 0 || cfg->ntile == 0 || cfg->ktile == 0 || cfg->gm == 0 || cfg->gn == 0)
        {
            return 0;
        }

        m_tiles = bmpmm_ceil_div_ul(cfg->M, cfg->mtile);
        n_tiles = bmpmm_ceil_div_ul(cfg->N, cfg->ntile);
        k_tiles = bmpmm_ceil_div_ul(cfg->K, cfg->ktile);
        weight_bits = bmpmm_weight_bits_from_prec(cfg->prec);
        if (weight_bits == 0)
            return 0;

        act_bits = cfg->mtile * cfg->ktile * 8UL;
        wgt_bits = cfg->ntile * cfg->ktile * weight_bits;
        reuse_a = (act_bits >= wgt_bits) ? 1 : 0;

        for (unsigned long mg = 0; mg < m_tiles; mg += cfg->gm)
        {
            unsigned long mg_len = bmpmm_min_ul(cfg->gm, m_tiles - mg);

            for (unsigned long ng = 0; ng < n_tiles; ng += cfg->gn)
            {
                unsigned long ng_len = bmpmm_min_ul(cfg->gn, n_tiles - ng);
                unsigned long pair_count = mg_len * ng_len;
                unsigned long cur_pair_ai = 0;
                unsigned long cur_pair_wi = 0;
                unsigned long k0 = 0;
                unsigned long k_rem = 0;
                unsigned long k_cfg = 0;
                const void *a_ptr = 0;
                const void *b_ptr = 0;

                if (pair_count == 0)
                    continue;

                bmpmm_pair_index_to_coords(0, mg_len, ng_len, reuse_a, &cur_pair_ai, &cur_pair_wi);
                k0 = 0;
                k_rem = cfg->K;
                k_cfg = (k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(k_rem, 8UL);
                a_ptr = ops->addr_a(A, cfg, mg + cur_pair_ai, k0, user);
                b_ptr = ops->addr_b(B, cfg, ng + cur_pair_wi, k0, user);
                ops->emit_cfg(cfg, k_cfg, user);
                ops->load_w(b_ptr, cur_pair_wi, user);
                ops->load_a(a_ptr, cur_pair_ai, user);

                for (unsigned long pair_idx = 0; pair_idx < pair_count; ++pair_idx)
                {
                    unsigned long ai = 0;
                    unsigned long wi = 0;

                    bmpmm_pair_index_to_coords(pair_idx, mg_len, ng_len, reuse_a, &ai, &wi);

                    for (unsigned long kt = 0; kt < k_tiles; ++kt)
                    {
                        if (kt + 1 < k_tiles)
                        {
                            unsigned long next_kt = kt + 1;
                            unsigned long next_k0 = next_kt * cfg->ktile;
                            unsigned long next_k_rem = (cfg->K > next_k0) ? (cfg->K - next_k0) : 0;
                            unsigned long next_k_cfg = (next_k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(next_k_rem, 8UL);
                            const void *next_a_ptr = ops->addr_a(A, cfg, mg + ai, next_k0, user);
                            const void *next_b_ptr = ops->addr_b(B, cfg, ng + wi, next_k0, user);

                            ops->compute(user);
                            ops->emit_cfg(cfg, next_k_cfg, user);
                            ops->load_w(next_b_ptr, wi, user);
                            ops->load_a(next_a_ptr, ai, user);
                        }
                        else
                        {
                            ops->compute(user);
                        }
                    }

                    ops->store_c(ops->addr_c(C, cfg, mg + ai, ng + wi, user), ai, wi, user);

                    if (pair_idx + 1 < pair_count)
                    {
                        unsigned long next_ai = 0;
                        unsigned long next_wi = 0;
                        const void *next_pair_a_ptr;
                        const void *next_pair_b_ptr;

                        bmpmm_pair_index_to_coords(pair_idx + 1, mg_len, ng_len, reuse_a, &next_ai, &next_wi);
                        next_pair_a_ptr = ops->addr_a(A, cfg, mg + next_ai, 0, user);
                        next_pair_b_ptr = ops->addr_b(B, cfg, ng + next_wi, 0, user);
                        ops->emit_cfg(cfg, cfg->ktile, user);
                        ops->load_w(next_pair_b_ptr, next_wi, user);
                        ops->load_a(next_pair_a_ptr, next_ai, user);
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
