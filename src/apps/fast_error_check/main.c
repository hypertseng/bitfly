#include "runtime.h"
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

#define FAST_ERR_DEFAULT_M 16UL
#define FAST_ERR_DEFAULT_K 64UL
#define FAST_ERR_DEFAULT_N 16UL
#define FAST_ERR_MAX_M 40UL
#define FAST_ERR_MAX_K 128UL
#define FAST_ERR_MAX_N 40UL
#define FAST_ERR_LIMIT_PCT 5.0
#define FAST_ERR_NR_LANES 4UL

typedef struct
{
    unsigned long mode;
    int64_t *estimated_total_cycles;
} rvv_int2_vector_exec_opts_t;

typedef struct
{
    unsigned long mode;
    int64_t *estimated_total_cycles;
} rvv_int4_vector_exec_opts_t;

typedef struct
{
    unsigned long mode;
    int64_t *estimated_total_cycles;
} rvv_binary_vector_exec_opts_t;

#define RVV_INT2_VECTOR_EXEC_STRICT 0UL
#define RVV_INT2_VECTOR_EXEC_FAST 1UL
#define RVV_INT4_VECTOR_EXEC_STRICT 0UL
#define RVV_INT4_VECTOR_EXEC_FAST 1UL
#define RVV_BINARY_VECTOR_EXEC_STRICT 0UL
#define RVV_BINARY_VECTOR_EXEC_FAST 1UL

extern int64_t rvv_int2_vector_compute_time;
extern int64_t rvv_int4_vector_compute_time;
extern int64_t rvv_binary_vector_compute_time;

void rvv_int2_vector_int2_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                           unsigned long int M, unsigned long int K, unsigned long int N,
                                           const rvv_int2_vector_exec_opts_t *opts);
int64_t rvv_int2_vector_get_last_estimated_total_cycles(void);
int64_t rvv_int2_vector_get_last_estimated_compute_cycles(void);

void rvv_int4_vector_int4_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                           unsigned long int M, unsigned long int K, unsigned long int N,
                                           const rvv_int4_vector_exec_opts_t *opts);
int64_t rvv_int4_vector_get_last_estimated_total_cycles(void);
int64_t rvv_int4_vector_get_last_estimated_compute_cycles(void);

void rvv_binary_vector_int8_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                             unsigned long int M, unsigned long int K, unsigned long int N,
                                             const rvv_binary_vector_exec_opts_t *opts);
int64_t rvv_binary_vector_get_last_estimated_total_cycles(void);
int64_t rvv_binary_vector_get_last_estimated_compute_cycles(void);

static inline unsigned long ceil_div_ul(unsigned long a, unsigned long b)
{
    return (a + b - 1UL) / b;
}

static inline unsigned long min_ul(unsigned long a, unsigned long b)
{
    return (a < b) ? a : b;
}

static inline unsigned long max_ul(unsigned long a, unsigned long b)
{
    return (a > b) ? a : b;
}

static double rel_err_pct(int64_t estimate, int64_t exact)
{
    if (exact == 0)
        return 0.0;
    double diff = (double)(estimate - exact);
    if (diff < 0.0)
        diff = -diff;
    return 100.0 * diff / (double)exact;
}

typedef struct
{
    const char *name;
    unsigned long M;
    unsigned long K;
    unsigned long N;
} fast_err_shape_t;

static const fast_err_shape_t kBmpmmInt2Shapes[] = {
    {"baseline", 16UL, 64UL, 16UL},
    {"tail_mn", 24UL, 64UL, 24UL},
    {"large_mkn", 40UL, 128UL, 40UL},
};

#define FAST_ERR_BMPMM_INT2_SHAPE_COUNT ((int)(sizeof(kBmpmmInt2Shapes) / sizeof(kBmpmmInt2Shapes[0])))

static uint64_t pack_row_chunk(const int8_t *row, unsigned long k_chunk, unsigned long k_dim)
{
    uint8_t bytes[8] = {0};
    for (unsigned long i = 0; i < 8UL; ++i)
    {
        const unsigned long k = k_chunk * 8UL + i;
        if (k < k_dim)
            bytes[i] = (uint8_t)row[k];
    }

    uint64_t word = 0;
    for (unsigned long i = 0; i < 8UL; ++i)
        word |= ((uint64_t)bytes[i]) << (i * 8UL);
    return word;
}

static void pack_activations_bmpu(uint64_t *dst_words, const int8_t *src, unsigned long m_dim,
                                  unsigned long k_dim, unsigned long mtile)
{
    const unsigned long d = ceil_div_ul(k_dim, 8UL);
    unsigned long out = 0UL;
    for (unsigned long tile_m = 0UL; tile_m < m_dim; tile_m += mtile)
    {
        const unsigned long tile_rows = min_ul(mtile, m_dim - tile_m);
        const unsigned long m_blocks = max_ul(1UL, ceil_div_ul(tile_rows, 8UL));
        for (unsigned long m_block = 0UL; m_block < m_blocks; ++m_block)
        {
            const unsigned long base = tile_m + m_block * 8UL;
            for (unsigned long k_chunk = 0UL; k_chunk < d; ++k_chunk)
            {
                for (unsigned long lane = 0UL; lane < FAST_ERR_NR_LANES; ++lane)
                {
                    const unsigned long row = base + 2UL * lane;
                    dst_words[out++] = (row < m_dim) ? pack_row_chunk(src + row * k_dim, k_chunk, k_dim) : 0ULL;
                }
                for (unsigned long lane = 0UL; lane < FAST_ERR_NR_LANES; ++lane)
                {
                    const unsigned long row = base + 2UL * lane + 1UL;
                    dst_words[out++] = (row < m_dim) ? pack_row_chunk(src + row * k_dim, k_chunk, k_dim) : 0ULL;
                }
            }
        }
    }
}

static uint64_t pack_8x8_bit_block(const uint8_t block[8][8])
{
    uint64_t word = 0ULL;
    for (unsigned long n_col = 0UL; n_col < 8UL; ++n_col)
    {
        uint64_t byte_val = 0ULL;
        for (unsigned long k_row = 0UL; k_row < 8UL; ++k_row)
            byte_val |= ((uint64_t)(block[k_row][n_col] & 0x1U)) << (7UL - k_row);
        word |= (byte_val & 0xFFULL) << (n_col * 8UL);
    }
    return word;
}

static uint64_t pack_weight_word_int2(const int8_t *weight, unsigned long k_dim, unsigned long n_dim,
                                      unsigned long plane, unsigned long k_blk, unsigned long n0)
{
    uint8_t block[8][8] = {{0}};
    const unsigned long k_base = k_blk * 8UL;
    for (unsigned long kr = 0UL; kr < 8UL; ++kr)
    {
        for (unsigned long nc = 0UL; nc < 8UL; ++nc)
        {
            const unsigned long kg = k_base + kr;
            const unsigned long ng = n0 + nc;
            if (kg < k_dim && ng < n_dim)
            {
                const unsigned int raw = ((unsigned int)((int)weight[kg * n_dim + ng])) & 0x3U;
                block[kr][nc] = (uint8_t)((raw >> plane) & 0x1U);
            }
        }
    }
    return pack_8x8_bit_block(block);
}

static void pack_weights_bmpu_int2(uint64_t *dst_words, const int8_t *weight, unsigned long k_dim,
                                   unsigned long n_dim, unsigned long ntile)
{
    const unsigned long d = ceil_div_ul(k_dim, 8UL);
    unsigned long out = 0UL;
    for (unsigned long tile_n = 0UL; tile_n < n_dim; tile_n += ntile)
    {
        const unsigned long tile_cols = min_ul(ntile, n_dim - tile_n);
        const unsigned long n_blocks = max_ul(1UL, ceil_div_ul(tile_cols, 16UL));
        for (unsigned long n_block = 0UL; n_block < n_blocks; ++n_block)
        {
            const unsigned long n_base = tile_n + n_block * 16UL;
            for (unsigned long k_blk = 0UL; k_blk < d; ++k_blk)
            {
                for (unsigned long plane = 0UL; plane < 2UL; ++plane)
                {
                    dst_words[out++] = pack_weight_word_int2(weight, k_dim, n_dim, plane, k_blk, n_base);
                    dst_words[out++] = pack_weight_word_int2(weight, k_dim, n_dim, plane, k_blk, n_base + 8UL);
                }
            }
        }
    }
}

static void fill_activation(int8_t *dst, unsigned long m_dim, unsigned long k_dim)
{
    for (unsigned long m = 0UL; m < m_dim; ++m)
        for (unsigned long k = 0UL; k < k_dim; ++k)
            dst[m * k_dim + k] = (int8_t)(((m * 13UL + k * 7UL + 5UL) % 15UL) - 7L);
}

static void fill_int2_weight(int8_t *dst, unsigned long k_dim, unsigned long n_dim)
{
    static const int8_t lut[4] = {-2, -1, 0, 1};
    for (unsigned long k = 0UL; k < k_dim; ++k)
        for (unsigned long n = 0UL; n < n_dim; ++n)
            dst[k * n_dim + n] = lut[(k * 11UL + n * 3UL + 1UL) & 0x3UL];
}

static void fill_int4_weight(int8_t *dst, unsigned long k_dim, unsigned long n_dim)
{
    for (unsigned long k = 0UL; k < k_dim; ++k)
        for (unsigned long n = 0UL; n < n_dim; ++n)
            dst[k * n_dim + n] = (int8_t)(((k * 11UL + n * 5UL + 3UL) & 0xFUL) - 8L);
}

static void fill_binary_weight(int8_t *dst, unsigned long k_dim, unsigned long n_dim)
{
    for (unsigned long k = 0UL; k < k_dim; ++k)
        for (unsigned long n = 0UL; n < n_dim; ++n)
            dst[k * n_dim + n] = (int8_t)((k * 3UL + n * 5UL + 1UL) & 0x1UL);
}

static int run_bmpmm_int2_case(const fast_err_shape_t *shape)
{
    static int8_t activation[FAST_ERR_MAX_M * FAST_ERR_MAX_K] __attribute__((aligned(32)));
    static int8_t weight_int2[FAST_ERR_MAX_K * FAST_ERR_MAX_N] __attribute__((aligned(32)));
    static uint64_t activation_lp_words[1024] __attribute__((aligned(32)));
    static uint64_t weight_lp_words[512] __attribute__((aligned(32)));
    static int16_t result_strict[FAST_ERR_MAX_M * FAST_ERR_MAX_N] __attribute__((aligned(32)));
    static int16_t result_fast[FAST_ERR_MAX_M * FAST_ERR_MAX_N] __attribute__((aligned(32)));
    const bmpmm_exec_cfg_t cfg = {8UL, 16UL, 64UL, 2UL, 1UL, 2UL};
    const unsigned long M = shape->M;
    const unsigned long K = shape->K;
    const unsigned long N = shape->N;
    int64_t strict_total;
    int64_t strict_total_warm;
    int64_t strict_total_same_buf;
    int64_t strict_compute = 0;
    int64_t strict_compute_warm = 0;
    int64_t strict_compute_same_buf = 0;
    int64_t fast_total = 0;
    int64_t fast_compute = 0;
    bmpmm_lowp_exec_opts_t strict_opts = {BMPMM_LOWP_EXEC_STRICT, 0};
    bmpmm_lowp_exec_opts_t fast_opts = {BMPMM_LOWP_EXEC_FAST, &fast_total};
    const double err_same_buf_limit = FAST_ERR_LIMIT_PCT;
    double err_same_buf = 0.0;
    int pass;

    fill_activation(activation, M, K);
    fill_int2_weight(weight_int2, K, N);
    pack_activations_bmpu(activation_lp_words, activation, M, K, cfg.mtile);
    pack_weights_bmpu_int2(weight_lp_words, weight_int2, K, N, cfg.ntile);
    memset(result_strict, 0, sizeof(result_strict));
    memset(result_fast, 0, sizeof(result_fast));

    printf("[fast_error_check][bmpmm_int2][%s] shape=(%lu,%lu,%lu) strict_begin\n",
           shape->name, M, N, K);
    start_timer();
    (void)bmpmm_lowp_mixed_matmul_with_cfg_opts("fast_error_check",
                                                result_strict,
                                                activation,
                                                (const int8_t *)weight_lp_words,
                                                M,
                                                K,
                                                N,
                                                &cfg,
                                                &strict_compute,
                                                &strict_opts);
    stop_timer();
    strict_total = get_timer();
    printf("[fast_error_check][bmpmm_int2][%s] strict_done total=%ld compute=%ld\n",
           shape->name, (long)strict_total, (long)strict_compute);

    start_timer();
    (void)bmpmm_lowp_mixed_matmul_with_cfg_opts("fast_error_check",
                                                result_strict,
                                                activation,
                                                (const int8_t *)weight_lp_words,
                                                M,
                                                K,
                                                N,
                                                &cfg,
                                                &strict_compute_warm,
                                                &strict_opts);
    stop_timer();
    strict_total_warm = get_timer();
    printf("[fast_error_check][bmpmm_int2][%s] strict_warm_done total=%ld compute=%ld\n",
           shape->name, (long)strict_total_warm, (long)strict_compute_warm);

    start_timer();
    (void)bmpmm_lowp_mixed_matmul_with_cfg_opts("fast_error_check",
                                                result_fast,
                                                activation,
                                                (const int8_t *)weight_lp_words,
                                                M,
                                                K,
                                                N,
                                                &cfg,
                                                &strict_compute_same_buf,
                                                &strict_opts);
    stop_timer();
    strict_total_same_buf = get_timer();
    printf("[fast_error_check][bmpmm_int2][%s] strict_same_buf_done total=%ld compute=%ld\n",
           shape->name, (long)strict_total_same_buf, (long)strict_compute_same_buf);

    printf("[fast_error_check][bmpmm_int2][%s] fast_begin\n", shape->name);
    (void)bmpmm_lowp_mixed_matmul_with_cfg_opts("fast_error_check",
                                                result_fast,
                                                activation,
                                                (const int8_t *)weight_lp_words,
                                                M,
                                                K,
                                                N,
                                                &cfg,
                                                &fast_compute,
                                                &fast_opts);
    printf("[fast_error_check][bmpmm_int2][%s] fast_done total=%ld compute=%ld\n",
           shape->name, (long)fast_total, (long)fast_compute);

    err_same_buf = rel_err_pct(fast_total, strict_total_same_buf);
    pass = (err_same_buf <= err_same_buf_limit);
    printf("[fast_error_check][bmpmm_int2][%s] strict_total=%ld strict_compute=%ld strict_warm_total=%ld strict_same_buf_total=%ld fast_total=%ld fast_compute=%ld err_same_buf=%.2f%%\n",
           shape->name, (long)strict_total, (long)strict_compute, (long)strict_total_warm, (long)strict_total_same_buf,
           (long)fast_total, (long)fast_compute, err_same_buf);
    printf("[fast_error_check][bmpmm_int2][%s] err_vs_cold=%.2f%% err_vs_warm=%.2f%% err_vs_same_buf=%.2f%% verdict=%s limit=%.2f%%\n",
           shape->name, rel_err_pct(fast_total, strict_total),
           rel_err_pct(fast_total, strict_total_warm), err_same_buf,
           pass ? "PASS" : "FAIL", err_same_buf_limit);
    return pass;
}

static void run_bmpmm_int2_test(void)
{
    int all_pass = 1;

    for (int i = 0; i < FAST_ERR_BMPMM_INT2_SHAPE_COUNT; ++i)
    {
        if (!run_bmpmm_int2_case(&kBmpmmInt2Shapes[i]))
            all_pass = 0;
    }

    printf("[fast_error_check][bmpmm_int2] summary tested=%d verdict=%s limit=%.2f%%\n",
           FAST_ERR_BMPMM_INT2_SHAPE_COUNT, all_pass ? "PASS" : "FAIL", FAST_ERR_LIMIT_PCT);
}

static void run_rvv_int2_test(void)
{
    static int8_t activation[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_K] __attribute__((aligned(32)));
    static int8_t weight[FAST_ERR_DEFAULT_K * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    static int16_t result_strict[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    static int16_t result_fast[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    int64_t strict_total;
    int64_t fast_total = 0;
    rvv_int2_vector_exec_opts_t strict_opts = {RVV_INT2_VECTOR_EXEC_STRICT, 0};
    rvv_int2_vector_exec_opts_t fast_opts = {RVV_INT2_VECTOR_EXEC_FAST, &fast_total};

    fill_activation(activation, FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K);
    fill_int2_weight(weight, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N);

    start_timer();
    rvv_int2_vector_int2_matmul_with_opts(result_strict, activation, weight,
                                          FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N,
                                          &strict_opts);
    stop_timer();
    strict_total = get_timer();

    rvv_int2_vector_int2_matmul_with_opts(result_fast, activation, weight,
                                          FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N,
                                          &fast_opts);

    printf("[fast_error_check][rvv_int2] strict_total=%ld fast_total=%ld fast_compute=%ld err_total=%.2f%%\n",
           (long)strict_total, (long)fast_total,
           (long)rvv_int2_vector_get_last_estimated_compute_cycles(),
           rel_err_pct(fast_total, strict_total));
}

static void run_rvv_int4_test(void)
{
    static int8_t activation[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_K] __attribute__((aligned(32)));
    static int8_t weight[FAST_ERR_DEFAULT_K * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    static int16_t result_strict[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    static int16_t result_fast[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    int64_t strict_total;
    int64_t fast_total = 0;
    rvv_int4_vector_exec_opts_t strict_opts = {RVV_INT4_VECTOR_EXEC_STRICT, 0};
    rvv_int4_vector_exec_opts_t fast_opts = {RVV_INT4_VECTOR_EXEC_FAST, &fast_total};

    fill_activation(activation, FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K);
    fill_int4_weight(weight, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N);

    start_timer();
    rvv_int4_vector_int4_matmul_with_opts(result_strict, activation, weight,
                                          FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N,
                                          &strict_opts);
    stop_timer();
    strict_total = get_timer();

    rvv_int4_vector_int4_matmul_with_opts(result_fast, activation, weight,
                                          FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N,
                                          &fast_opts);

    printf("[fast_error_check][rvv_int4] strict_total=%ld fast_total=%ld fast_compute=%ld err_total=%.2f%%\n",
           (long)strict_total, (long)fast_total,
           (long)rvv_int4_vector_get_last_estimated_compute_cycles(),
           rel_err_pct(fast_total, strict_total));
}

static void run_rvv_binary_test(void)
{
    static int8_t activation[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_K] __attribute__((aligned(32)));
    static int8_t weight[FAST_ERR_DEFAULT_K * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    static int16_t result_strict[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    static int16_t result_fast[FAST_ERR_DEFAULT_M * FAST_ERR_DEFAULT_N] __attribute__((aligned(32)));
    int64_t strict_total;
    int64_t strict_compute;
    int64_t fast_total = 0;
    int64_t fast_compute;
    rvv_binary_vector_exec_opts_t strict_opts = {RVV_BINARY_VECTOR_EXEC_STRICT, 0};
    rvv_binary_vector_exec_opts_t fast_opts = {RVV_BINARY_VECTOR_EXEC_FAST, &fast_total};

    fill_activation(activation, FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K);
    fill_binary_weight(weight, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N);

    start_timer();
    rvv_binary_vector_int8_matmul_with_opts(result_strict, activation, weight,
                                            FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N,
                                            &strict_opts);
    stop_timer();
    strict_total = get_timer();
    strict_compute = rvv_binary_vector_compute_time;

    rvv_binary_vector_int8_matmul_with_opts(result_fast, activation, weight,
                                            FAST_ERR_DEFAULT_M, FAST_ERR_DEFAULT_K, FAST_ERR_DEFAULT_N,
                                            &fast_opts);
    fast_compute = rvv_binary_vector_get_last_estimated_compute_cycles();

    printf("[fast_error_check][rvv_binary] strict_total=%ld strict_compute=%ld fast_total=%ld fast_compute=%ld err_total=%.2f%% err_compute=%.2f%%\n",
           (long)strict_total, (long)strict_compute, (long)fast_total, (long)fast_compute,
           rel_err_pct(fast_total, strict_total), rel_err_pct(fast_compute, strict_compute));
}

int main(void)
{
    printf("[fast_error_check] bmpmm_int2_shapes=%d limit=%.2f%%\n",
           FAST_ERR_BMPMM_INT2_SHAPE_COUNT, FAST_ERR_LIMIT_PCT);
    printf("[fast_error_check] run=bmpmm_int2_multi_shape\n");
    run_bmpmm_int2_test();
    printf("[fast_error_check] run=rvv_int2\n");
    run_rvv_int2_test();
    printf("[fast_error_check] run=rvv_int4\n");
    run_rvv_int4_test();
    printf("[fast_error_check] run=rvv_binary\n");
    run_rvv_binary_test();
    return 0;
}
