#include "runtime.h"
#include "util.h"
#include <stdint.h>
#include <string.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "../common/bmpmm_bench_common.h"
#include "../common/bmpmm_lowp_mixed_common.h"
#include "kernel/data.h"
#include "kernel/bench_cases.h"

static unsigned long min_ul(unsigned long a, unsigned long b)
{
    return (a < b) ? a : b;
}

static void unpack_packed_tiles_to_col_major(int16_t *dst, const int16_t *src,
                                             unsigned long M, unsigned long N,
                                             unsigned long mtile, unsigned long ntile)
{
    const unsigned long m_tiles = (M + mtile - 1) / mtile;
    const unsigned long n_tiles = (N + ntile - 1) / ntile;
    const unsigned long tile_elems = mtile * ntile;

    for (unsigned long n_tile = 0; n_tile < n_tiles; ++n_tile)
    {
        const unsigned long n0 = n_tile * ntile;
        const unsigned long n_valid = min_ul(ntile, N - n0);
        for (unsigned long m_tile = 0; m_tile < m_tiles; ++m_tile)
        {
            const unsigned long m0 = m_tile * mtile;
            const unsigned long m_valid = min_ul(mtile, M - m0);
            const unsigned long tile_idx = n_tile * m_tiles + m_tile;
            const int16_t *tile = src + tile_idx * tile_elems;
            for (unsigned long n_local = 0; n_local < n_valid; ++n_local)
            {
                for (unsigned long m_local = 0; m_local < m_valid; ++m_local)
                {
                    dst[(n0 + n_local) * M + (m0 + m_local)] =
                        tile[n_local * mtile + m_local];
                }
            }
        }
    }
}

static int compare_col_major(const int16_t *got, const int16_t *expect,
                             unsigned long M, unsigned long N)
{
    int mismatches = 0;
    int nonzero = 0;
    const unsigned long total = M * N;
    for (unsigned long idx = 0; idx < total; ++idx)
        if (got[idx] != 0)
            ++nonzero;
    printf("[mytest] raw_nonzero=%d/%d\n", nonzero, (int)total);
    printf("[mytest] raw_head={");
    for (unsigned long idx = 0; idx < 16 && idx < total; ++idx)
        printf(idx == 15 || idx + 1 == total ? "%d" : "%d, ", got[idx]);
    printf("}\n");
    printf("[mytest] col_nz={");
    for (unsigned long n = 0; n < 8 && n < N; ++n)
    {
        int col_nz = 0;
        for (unsigned long m = 0; m < M; ++m)
            if (got[n * M + m] != 0)
                ++col_nz;
        printf(n == 7 || n + 1 == N ? "%d" : "%d, ", col_nz);
    }
    printf("}\n");
    printf("[mytest] row_nz={");
    for (unsigned long m = 0; m < 16 && m < M; ++m)
    {
        int row_nz = 0;
        for (unsigned long n = 0; n < N; ++n)
            if (got[n * M + m] != 0)
                ++row_nz;
        printf(m == 15 || m + 1 == M ? "%d" : "%d, ", row_nz);
    }
    printf("}\n");
    printf("[mytest] nz_pos={");
    int nz_dumped = 0;
    for (unsigned long n = 0; n < N && nz_dumped < 32; ++n)
    {
        for (unsigned long m = 0; m < M && nz_dumped < 32; ++m)
        {
            if (got[n * M + m] != 0)
            {
                printf(nz_dumped == 31 ? "(%d,%d)" : "(%d,%d), ", (int)m, (int)n);
                ++nz_dumped;
            }
        }
    }
    printf("}\n");
    for (unsigned long dbg_n = 0; dbg_n < 4 && dbg_n < N; ++dbg_n)
    {
        printf("[mytest] col%d_rows={", (int)dbg_n);
        int dumped = 0;
        for (unsigned long m = 0; m < M && dumped < 40; ++m)
        {
            if (got[dbg_n * M + m] != 0)
            {
                printf(dumped == 39 ? "%d" : "%d, ", (int)m);
                ++dumped;
            }
        }
        printf("}\n");
    }
    for (unsigned long n = 0; n < N; ++n)
    {
        for (unsigned long m = 0; m < M; ++m)
        {
            unsigned long idx = n * M + m;
            if (got[idx] != expect[idx])
            {
                if (mismatches < 16)
                {
                    printf("[mytest] mismatch at (m=%d,n=%d): got %d expect %d\n",
                           (int)m, (int)n, got[idx], expect[idx]);
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

int main()
{
    int failures = 0;
    printf("[mytest] low-bit mixed-precision correctness check\n");

    for (int i = 0; i < BMPMM_BENCH_CASE_COUNT; ++i)
    {
        const bmpmm_bench_case_t *sc = &kBenchCases[i];
        BenchKernelData data = get_bench_kernel_data(i);

        printf("\n------------------------------------------------------------\n");
        printf("[mytest] case%d name=%s shape=(%lu,%lu,%lu), cfg=(mt=%lu,nt=%lu,kt=%lu,gm=%lu,gn=%lu,p=%lu)\n",
               i + 1, sc->layer, sc->M, sc->N, sc->K,
               sc->cfg.mtile, sc->cfg.ntile, sc->cfg.ktile,
               sc->cfg.gm, sc->cfg.gn, sc->cfg.prec);

        start_timer();
        int ok = bmpmm_lowp_mixed_matmul_with_cfg("mytest",
                                                  data.result_hp,
                                                  data.activation_lp,
                                                  data.weight_lp,
                                                  sc->M,
                                                  sc->K,
                                                  sc->N,
                                                  &sc->cfg,
                                                  0);
        stop_timer();
        if (!ok)
        {
            printf("[mytest] ERROR: execution failed for %s\n", sc->layer);
            ++failures;
            continue;
        }

        for (unsigned long dbg_col = 20; dbg_col < 24; ++dbg_col)
        {
            printf("[mytest][DBG] packed col%lu={", dbg_col);
            for (unsigned long dbg_row = 0; dbg_row < sc->cfg.mtile; ++dbg_row)
            {
                unsigned long packed_idx = dbg_col * sc->cfg.mtile + dbg_row;
                printf(dbg_row + 1 == sc->cfg.mtile ? "%d" : "%d, ", data.result_hp[packed_idx]);
            }
            printf("} gold={");
            for (unsigned long dbg_row = 0; dbg_row < sc->cfg.mtile; ++dbg_row)
            {
                unsigned long gold_idx = dbg_col * sc->M + dbg_row;
                printf(dbg_row + 1 == sc->cfg.mtile ? "%d" : "%d, ", data.result_torch[gold_idx]);
            }
            printf("}\n");
        }

        printf("[mytest][DBG] unpack_begin %s\n", sc->layer);
        unpack_packed_tiles_to_col_major(data.result_lp,
                                         data.result_hp,
                                         sc->M,
                                         sc->N,
                                         sc->cfg.mtile,
                                         sc->cfg.ntile);
        printf("[mytest][DBG] unpack_done %s\n", sc->layer);

        printf("[mytest][DBG] compare_begin %s\n", sc->layer);
        int mismatches = compare_col_major(data.result_lp, data.result_torch, sc->M, sc->N);
        printf("[mytest][DBG] compare_done %s mismatches=%d\n", sc->layer, mismatches);
        int64_t runtime = get_timer();
        printf("[mytest] runtime=%ld cycles\n", (long)runtime);
        printf("[mytest] sample={{%d, %d, %d, %d}}\n",
               data.result_lp[0], data.result_lp[1], data.result_lp[2], data.result_lp[3]);

        if (mismatches == 0)
        {
            printf("[mytest] PASS %s\n", sc->layer);
        }
        else
        {
            printf("[mytest] FAIL %s mismatches=%d\n", sc->layer, mismatches);
            ++failures;
        }
    }

    if (failures == 0)
    {
        printf("\n[mytest] ALL CASES PASSED\n");
        return 0;
    }

    printf("\n[mytest] TOTAL FAILURES=%d\n", failures);
    return 1;
}
