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

    typedef struct
    {
        unsigned long a_slots;
        unsigned long w_slots;
        unsigned long a_windows;
        unsigned long w_windows;
        unsigned long row_snake;
        unsigned long reuse_a;
    } bmpmm_group_plan_t;

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

    static inline void bmpmm_window_shape(unsigned long total_len, unsigned long block_len,
                                          unsigned long win_idx, unsigned long *start, unsigned long *len)
    {
        unsigned long local_start = win_idx * block_len;
        unsigned long local_len = 0;
        if (local_start < total_len)
            local_len = bmpmm_min_ul(block_len, total_len - local_start);
        if (start)
            *start = local_start;
        if (len)
            *len = local_len;
    }

    static inline void bmpmm_next_window(unsigned long a_windows, unsigned long w_windows,
                                         unsigned long row_snake,
                                         unsigned long cur_a, unsigned long cur_w,
                                         unsigned long *next_a, unsigned long *next_w, int *valid)
    {
        unsigned long na = cur_a;
        unsigned long nw = cur_w;
        int ok = 0;

        if (row_snake)
        {
            if ((cur_a & 1UL) == 0UL)
            {
                if (cur_w + 1UL < w_windows)
                {
                    nw = cur_w + 1UL;
                    ok = 1;
                }
                else if (cur_a + 1UL < a_windows)
                {
                    na = cur_a + 1UL;
                    nw = (w_windows == 0UL) ? 0UL : (w_windows - 1UL);
                    ok = 1;
                }
            }
            else
            {
                if (cur_w > 0UL)
                {
                    nw = cur_w - 1UL;
                    ok = 1;
                }
                else if (cur_a + 1UL < a_windows)
                {
                    na = cur_a + 1UL;
                    nw = 0UL;
                    ok = 1;
                }
            }
        }
        else
        {
            if ((cur_w & 1UL) == 0UL)
            {
                if (cur_a + 1UL < a_windows)
                {
                    na = cur_a + 1UL;
                    ok = 1;
                }
                else if (cur_w + 1UL < w_windows)
                {
                    na = (a_windows == 0UL) ? 0UL : (a_windows - 1UL);
                    nw = cur_w + 1UL;
                    ok = 1;
                }
            }
            else
            {
                if (cur_a > 0UL)
                {
                    na = cur_a - 1UL;
                    ok = 1;
                }
                else if (cur_w + 1UL < w_windows)
                {
                    na = 0UL;
                    nw = cur_w + 1UL;
                    ok = 1;
                }
            }
        }

        if (next_a)
            *next_a = na;
        if (next_w)
            *next_w = nw;
        if (valid)
            *valid = ok;
    }

    static inline void bmpmm_pair_from_ord(unsigned long a_len, unsigned long w_len,
                                           unsigned long reuse_a, unsigned long pair_ord,
                                           unsigned long *a_pos, unsigned long *w_pos)
    {
        unsigned long ord = 0UL;
        if (reuse_a)
        {
            for (unsigned long ap = 0UL; ap < a_len; ++ap)
            {
                if ((ap & 1UL) == 0UL)
                {
                    for (unsigned long wp = 0UL; wp < w_len; ++wp, ++ord)
                    {
                        if (ord == pair_ord)
                        {
                            *a_pos = ap;
                            *w_pos = wp;
                            return;
                        }
                    }
                }
                else
                {
                    for (unsigned long wrev = 0UL; wrev < w_len; ++wrev, ++ord)
                    {
                        unsigned long wp = w_len - 1UL - wrev;
                        if (ord == pair_ord)
                        {
                            *a_pos = ap;
                            *w_pos = wp;
                            return;
                        }
                    }
                }
            }
        }
        else
        {
            for (unsigned long wp = 0UL; wp < w_len; ++wp)
            {
                if ((wp & 1UL) == 0UL)
                {
                    for (unsigned long ap = 0UL; ap < a_len; ++ap, ++ord)
                    {
                        if (ord == pair_ord)
                        {
                            *a_pos = ap;
                            *w_pos = wp;
                            return;
                        }
                    }
                }
                else
                {
                    for (unsigned long arev = 0UL; arev < a_len; ++arev, ++ord)
                    {
                        unsigned long ap = a_len - 1UL - arev;
                        if (ord == pair_ord)
                        {
                            *a_pos = ap;
                            *w_pos = wp;
                            return;
                        }
                    }
                }
            }
        }

        *a_pos = 0UL;
        *w_pos = 0UL;
    }

    static inline void bmpmm_select_group_plan(const bmpmm_template_cfg_t *cfg,
                                               unsigned long mg_len, unsigned long ng_len,
                                               unsigned long k_cfg,
                                               bmpmm_group_plan_t *plan)
    {
        unsigned long weight_bits;
        unsigned long total_slots;
        unsigned long best_load_bytes = ~0UL;
        long best_pref = -1L;
        long best_same_a = -1L;
        long best_same_w = -1L;
        unsigned long best_a_slots = 1UL;
        unsigned long best_w_slots = 1UL;
        unsigned long best_row_snake = 0UL;
        unsigned long reuse_a;

        if (!plan)
            return;

        weight_bits = bmpmm_weight_bits_from_prec(cfg->prec);
        reuse_a = ((cfg->mtile * k_cfg * 8UL) >= (cfg->ntile * k_cfg * weight_bits)) ? 1UL : 0UL;
        total_slots = bmpmm_min_ul(4UL, mg_len + ng_len);

        for (unsigned long a_slots = 1UL; a_slots <= bmpmm_min_ul(mg_len, total_slots - 1UL); ++a_slots)
        {
            unsigned long w_slots = total_slots - a_slots;
            unsigned long a_windows;
            unsigned long w_windows;

            if (w_slots == 0UL || w_slots > ng_len)
                continue;

            a_windows = bmpmm_ceil_div_ul(mg_len, a_slots);
            w_windows = bmpmm_ceil_div_ul(ng_len, w_slots);

            for (unsigned long row_snake = 0UL; row_snake <= 1UL; ++row_snake)
            {
                unsigned long cur_a = 0UL;
                unsigned long cur_w = 0UL;
                unsigned long prev_a = ~0UL;
                unsigned long prev_w = ~0UL;
                unsigned long load_a = 0UL;
                unsigned long load_w = 0UL;
                long same_a = 0L;
                long same_w = 0L;
                unsigned long prev_pair_a = ~0UL;
                unsigned long prev_pair_w = ~0UL;
                int have_prev_pair = 0;

                while (1)
                {
                    unsigned long a_start = 0UL, a_len = 0UL;
                    unsigned long w_start = 0UL, w_len = 0UL;
                    unsigned long pair_count = 0UL;
                    unsigned long next_a = 0UL, next_w = 0UL;
                    int has_next = 0;

                    bmpmm_window_shape(mg_len, a_slots, cur_a, &a_start, &a_len);
                    bmpmm_window_shape(ng_len, w_slots, cur_w, &w_start, &w_len);

                    if (cur_a != prev_a)
                        load_a += a_len;
                    if (cur_w != prev_w)
                        load_w += w_len;

                    pair_count = a_len * w_len;
                    for (unsigned long pair_ord = 0UL; pair_ord < pair_count; ++pair_ord)
                    {
                        unsigned long a_pos = 0UL, w_pos = 0UL;
                        unsigned long abs_a = 0UL, abs_w = 0UL;
                        bmpmm_pair_from_ord(a_len, w_len, reuse_a, pair_ord, &a_pos, &w_pos);
                        abs_a = a_start + a_pos;
                        abs_w = w_start + w_pos;
                        if (have_prev_pair)
                        {
                            if (abs_a == prev_pair_a)
                                ++same_a;
                            if (abs_w == prev_pair_w)
                                ++same_w;
                        }
                        prev_pair_a = abs_a;
                        prev_pair_w = abs_w;
                        have_prev_pair = 1;
                    }

                    prev_a = cur_a;
                    prev_w = cur_w;
                    bmpmm_next_window(a_windows, w_windows, row_snake, cur_a, cur_w, &next_a, &next_w, &has_next);
                    if (!has_next)
                        break;
                    cur_a = next_a;
                    cur_w = next_w;
                }

                {
                    unsigned long load_bytes =
                        load_a * cfg->mtile * k_cfg +
                        (load_w * cfg->ntile * k_cfg * weight_bits) / 8UL;
                    long preferred = reuse_a ? same_a : same_w;
                    int better = 0;

                    if (load_bytes < best_load_bytes)
                        better = 1;
                    else if (load_bytes == best_load_bytes && preferred > best_pref)
                        better = 1;
                    else if (load_bytes == best_load_bytes && preferred == best_pref && same_a > best_same_a)
                        better = 1;
                    else if (load_bytes == best_load_bytes && preferred == best_pref &&
                             same_a == best_same_a && same_w > best_same_w)
                        better = 1;
                    else if (load_bytes == best_load_bytes && preferred == best_pref &&
                             same_a == best_same_a && same_w == best_same_w &&
                             a_slots > best_a_slots)
                        better = 1;
                    else if (load_bytes == best_load_bytes && preferred == best_pref &&
                             same_a == best_same_a && same_w == best_same_w &&
                             a_slots == best_a_slots && row_snake < best_row_snake)
                        better = 1;

                    if (better)
                    {
                        best_load_bytes = load_bytes;
                        best_pref = preferred;
                        best_same_a = same_a;
                        best_same_w = same_w;
                        best_a_slots = a_slots;
                        best_w_slots = w_slots;
                        best_row_snake = row_snake;
                    }
                }
            }
        }

        plan->a_slots = best_a_slots;
        plan->w_slots = best_w_slots;
        plan->a_windows = bmpmm_ceil_div_ul(mg_len, best_a_slots);
        plan->w_windows = bmpmm_ceil_div_ul(ng_len, best_w_slots);
        plan->row_snake = best_row_snake;
        plan->reuse_a = reuse_a;
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
        bmpmm_group_plan_t plan;

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

        for (unsigned long mg = 0; mg < m_tiles; mg += cfg->gm)
        {
            unsigned long mg_len = bmpmm_min_ul(cfg->gm, m_tiles - mg);

            for (unsigned long ng = 0; ng < n_tiles; ng += cfg->gn)
            {
                unsigned long ng_len = bmpmm_min_ul(cfg->gn, n_tiles - ng);
                bmpmm_template_cfg_t group_cfg;

                if (mg_len == 0 || ng_len == 0)
                    continue;

                group_cfg = *cfg;
                group_cfg.gm = mg_len;
                group_cfg.gn = ng_len;
                bmpmm_select_group_plan(&group_cfg, mg_len, ng_len, cfg->ktile, &plan);

                for (unsigned long kt = 0; kt < k_tiles; ++kt)
                {
                    unsigned long k0 = kt * cfg->ktile;
                    unsigned long k_rem = (cfg->K > k0) ? (cfg->K - k0) : 0UL;
                    unsigned long k_cfg = (k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(k_rem, 8UL);
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
                        bmpmm_template_cfg_t window_cfg;

                        bmpmm_window_shape(mg_len, plan.a_slots, cur_a_win, &a_start, &a_len);
                        bmpmm_window_shape(ng_len, plan.w_slots, cur_w_win, &w_start, &w_len);
                        window_cfg = group_cfg;
                        ops->emit_cfg(&group_cfg, k_cfg, user);

                        if (cur_a_win != prev_a_win)
                        {
                            for (unsigned long a_pos = 0UL; a_pos < a_len; ++a_pos)
                            {
                                const void *a_ptr = ops->addr_a(A, &window_cfg, mg + a_start + a_pos, k0, user);
                                ops->load_a(a_ptr, a_pos, user);
                            }
                        }
                        if (cur_w_win != prev_w_win)
                        {
                            for (unsigned long w_pos = 0UL; w_pos < w_len; ++w_pos)
                            {
                                const void *b_ptr = ops->addr_b(B, &group_cfg, ng + w_start + w_pos, k0, user);
                                ops->load_w(b_ptr, plan.a_slots + w_pos, user);
                            }
                        }

                        {
                            unsigned long pair_count = a_len * w_len;
                            for (unsigned long pair_ord = 0UL; pair_ord < pair_count; ++pair_ord)
                            {
                                unsigned long a_pos = 0UL, w_pos = 0UL;
                                bmpmm_pair_from_ord(a_len, w_len, plan.reuse_a, pair_ord, &a_pos, &w_pos);
                                ops->compute(user);
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

                {
                    unsigned long cur_a_win = 0UL;
                    unsigned long cur_w_win = 0UL;
                    unsigned long last_k0 = (k_tiles - 1UL) * cfg->ktile;
                    unsigned long last_k_rem = (cfg->K > last_k0) ? (cfg->K - last_k0) : 0UL;
                    unsigned long last_k_cfg =
                        (last_k_rem >= cfg->ktile) ? cfg->ktile : bmpmm_align_up_ul(last_k_rem, 8UL);
                    while (1)
                    {
                        unsigned long a_start = 0UL, a_len = 0UL;
                        unsigned long w_start = 0UL, w_len = 0UL;
                        unsigned long next_a_win = 0UL, next_w_win = 0UL;
                        int has_next_window = 0;
                        unsigned long pair_count;
                        bmpmm_template_cfg_t window_cfg;

                        bmpmm_window_shape(mg_len, plan.a_slots, cur_a_win, &a_start, &a_len);
                        bmpmm_window_shape(ng_len, plan.w_slots, cur_w_win, &w_start, &w_len);
                        window_cfg = group_cfg;
                        ops->emit_cfg(&group_cfg, last_k_cfg, user);
                        pair_count = a_len * w_len;

                        for (unsigned long pair_ord = 0UL; pair_ord < pair_count; ++pair_ord)
                        {
                            unsigned long a_pos = 0UL, w_pos = 0UL;
                            unsigned long abs_ai = 0UL, abs_wi = 0UL;
                            bmpmm_pair_from_ord(a_len, w_len, plan.reuse_a, pair_ord, &a_pos, &w_pos);
                            abs_ai = a_start + a_pos;
                            abs_wi = w_start + w_pos;
                            ops->store_c(ops->addr_c(C, &window_cfg, mg + abs_ai, ng + abs_wi, user),
                                         a_pos, plan.a_slots + w_pos, user);
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
            }
        }

        return 1;
    }

#ifdef __cplusplus
}
#endif

#endif
