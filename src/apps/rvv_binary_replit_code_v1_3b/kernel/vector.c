#include "vector.h"
#include "runtime.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#if defined(SPIKE) || defined(ARA_LINUX)
#define RVV_BINARY_PROGRESS_FLUSH() fflush(stdout)
#else
#define RVV_BINARY_PROGRESS_FLUSH() ((void)0)
#endif

#ifndef RVV_BINARY_PROGRESS_ENABLE
#define RVV_BINARY_PROGRESS_ENABLE 1
#endif

#if RVV_BINARY_PROGRESS_ENABLE
#define RVV_BINARY_PROGRESS_LOG(...)        \
    do                                     \
    {                                      \
        printf(__VA_ARGS__);               \
        RVV_BINARY_PROGRESS_FLUSH();       \
    } while (0)
#else
#define RVV_BINARY_PROGRESS_LOG(...) ((void)0)
#endif

#ifdef RVV_BINARY_FAST_DEBUG
#define RVV_BINARY_FAST_DBG(...) printf(__VA_ARGS__)
#else
#define RVV_BINARY_FAST_DBG(...) ((void)0)
#endif

#ifdef RVV_BINARY_FAST_DEBUG_DEEP
#define RVV_BINARY_FAST_LOOP_DBG(k, K, N)                                                    \
    do                                                                                       \
    {                                                                                        \
        if (((k) & 511UL) == 0UL)                                                            \
            RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] kernel k=%lu/%lu strideN=%lu\n",     \
                                (unsigned long)(k), (unsigned long)(K), (unsigned long)(N)); \
    } while (0)
#define RVV_BINARY_FAST_STEP_DBG(tag, k, K)                                                 \
    do                                                                                      \
    {                                                                                       \
        if ((k) <= 2UL || (k) + 2UL >= (K))                                                 \
            RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] " tag " k=%lu/%lu\n",               \
                                (unsigned long)(k), (unsigned long)(K));                    \
    } while (0)
#define RVV_BINARY_FAST_PRELOAD_DBG(tag, ptr)                                               \
    RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] " tag " ptr=%p\n", (const void *)(ptr))
#else
#define RVV_BINARY_FAST_LOOP_DBG(k, K, N) ((void)0)
#define RVV_BINARY_FAST_STEP_DBG(tag, k, K) ((void)0)
#define RVV_BINARY_FAST_PRELOAD_DBG(tag, ptr) ((void)0)
#endif

// helper macro used by the vector kernel only
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define RVV_BINARY_FAST_STABLE_ATTR __attribute__((noinline))

#ifndef RVV_BINARY_FAST_SAMPLE_N_MAX
#define RVV_BINARY_FAST_SAMPLE_N_MAX 64UL
#endif

#ifndef RVV_BINARY_FAST_SAMPLE_K0
#define RVV_BINARY_FAST_SAMPLE_K0 8UL
#endif

#ifndef RVV_BINARY_FAST_SAMPLE_K1
#define RVV_BINARY_FAST_SAMPLE_K1 64UL
#endif

#ifndef RVV_BINARY_FAST_SEED_N64_TOTAL0
#define RVV_BINARY_FAST_SEED_N64_TOTAL0 830LL
#endif

#ifndef RVV_BINARY_FAST_SEED_N64_COMPUTE0
#define RVV_BINARY_FAST_SEED_N64_COMPUTE0 492LL
#endif

#ifndef RVV_BINARY_FAST_FULL_EXACT_BLOCKS_MAX
#define RVV_BINARY_FAST_FULL_EXACT_BLOCKS_MAX 8UL
#endif

static volatile int g_rvv_binary_fast_sample_touch_sink = 0;
static uint8_t g_rvv_binary_fast_cache_valid[RVV_BINARY_FAST_SAMPLE_N_MAX + 1UL];
static int64_t g_rvv_binary_fast_cache_total0[RVV_BINARY_FAST_SAMPLE_N_MAX + 1UL];
static int64_t g_rvv_binary_fast_cache_compute0[RVV_BINARY_FAST_SAMPLE_N_MAX + 1UL];
static volatile int g_rvv_binary_fast_unpack_touch_sink = 0;
static uint8_t g_rvv_binary_fast_unpack_cache_valid[RVV_BINARY_FAST_SAMPLE_N_MAX + 1UL];
static int64_t g_rvv_binary_fast_unpack_cache_cycles0[RVV_BINARY_FAST_SAMPLE_N_MAX + 1UL];

static unsigned long g_rvv_binary_vector_default_mode = RVV_BINARY_VECTOR_DEFAULT_MODE;
static int64_t g_rvv_binary_vector_last_estimated_total_cycles = 0;
static int64_t g_rvv_binary_vector_last_estimated_compute_cycles = 0;
static int g_rvv_binary_profile_compute_cycles = 1;

static inline unsigned long rvv_binary_vector_sanitize_mode(unsigned long mode)
{
    return (mode == RVV_BINARY_VECTOR_EXEC_FAST) ? RVV_BINARY_VECTOR_EXEC_FAST : RVV_BINARY_VECTOR_EXEC_STRICT;
}

static inline void rvv_binary_vector_reset_estimates(void)
{
    g_rvv_binary_vector_last_estimated_total_cycles = 0;
    g_rvv_binary_vector_last_estimated_compute_cycles = 0;
}

static int64_t rvv_binary_scale_cycles(unsigned long int full_k,
                                       unsigned long int sample_k,
                                       int64_t sample_cycles);

#define RVV_BINARY_FAST_SAMPLE_N_BLOCKS_MAX ((RVV_BINARY_FAST_SAMPLE_N_MAX + 15UL) / 16UL)

static inline int8_t rvv_binary_expand_1bit(unsigned long raw)
{
    return raw ? (int8_t)1 : (int8_t)-1;
}

static void rvv_binary_fill_packed_weight_sample(uint64_t *dst_words,
                                                 unsigned long int sample_k,
                                                 unsigned long int sample_n)
{
    const unsigned long int k_blocks = (sample_k + 7UL) / 8UL;
    const unsigned long int n_blocks = (sample_n + 15UL) / 16UL;
    unsigned long int word_idx = 0UL;

    for (unsigned long int k_blk = 0UL; k_blk < k_blocks; ++k_blk)
    {
        for (unsigned long int n_blk = 0UL; n_blk < n_blocks; ++n_blk)
        {
            const uint64_t base = 0x6996966996696996ULL ^
                                  ((uint64_t)k_blk << 28) ^
                                  ((uint64_t)n_blk << 12);
            dst_words[word_idx++] = base;
            dst_words[word_idx++] = base ^ 0x5555AAAA5555AAAAULL;
        }
    }
}

static RVV_BINARY_FAST_STABLE_ATTR void rvv_binary_unpack_weight_sample(int8_t *dst,
                                                                        const uint64_t *src_words,
                                                                        unsigned long int sample_k,
                                                                        unsigned long int sample_n)
{
    const unsigned long int k_blocks = (sample_k + 7UL) / 8UL;
    const unsigned long int n_blocks = (sample_n + 15UL) / 16UL;
    unsigned long int word_idx = 0UL;

    for (unsigned long int k_blk = 0UL; k_blk < k_blocks; ++k_blk)
    {
        const unsigned long int k_base = k_blk * 8UL;
        for (unsigned long int n_blk = 0UL; n_blk < n_blocks; ++n_blk)
        {
            const unsigned long int n_base = n_blk * 16UL;
            uint64_t words[2UL];
            words[0] = src_words[word_idx++];
            words[1] = src_words[word_idx++];

            for (unsigned long int half = 0UL; half < 2UL; ++half)
            {
                const uint64_t word = words[half];
                for (unsigned long int col = 0UL; col < 8UL; ++col)
                {
                    const uint8_t packed_col = (uint8_t)((word >> (col * 8UL)) & 0xFFUL);
                    const unsigned long int dst_col = half * 8UL + col;
                    if (n_base + dst_col >= sample_n)
                        continue;
                    for (unsigned long int row = 0UL; row < 8UL; ++row)
                    {
                        if (k_base + row >= sample_k)
                            continue;
                        dst[(k_base + row) * sample_n + (n_base + dst_col)] =
                            rvv_binary_expand_1bit((packed_col >> (7UL - row)) & 0x1U);
                    }
                }
            }
        }
    }
}

static RVV_BINARY_FAST_STABLE_ATTR int64_t rvv_binary_measure_unpack_sample_exact(unsigned long int sample_k,
                                                                                  unsigned long int sample_n)
{
    uint64_t src_words[(RVV_BINARY_FAST_SAMPLE_K1 / 8UL) * RVV_BINARY_FAST_SAMPLE_N_BLOCKS_MAX * 2UL];
    int8_t dst_bytes[RVV_BINARY_FAST_SAMPLE_K1 * RVV_BINARY_FAST_SAMPLE_N_MAX];
    int touch = 0;

    rvv_binary_fill_packed_weight_sample(src_words, sample_k, sample_n);
    asm volatile("fence rw, rw" ::: "memory");
    const int64_t start = get_cycle_count();
    rvv_binary_unpack_weight_sample(dst_bytes, src_words, sample_k, sample_n);
    asm volatile("fence rw, rw" ::: "memory");
    const int64_t cycles = get_cycle_count() - start;

    if (sample_k != 0UL && sample_n != 0UL)
    {
        const unsigned long int total = sample_k * sample_n;
        const unsigned long int stride = (total > 32UL) ? (total / 32UL) : 1UL;
        for (unsigned long int i = 0UL; i < total; i += stride)
            touch += dst_bytes[i];
        touch += dst_bytes[total - 1UL];
    }
    g_rvv_binary_fast_unpack_touch_sink = touch;
    asm volatile("" ::: "memory");
    return cycles;
}

static int64_t rvv_binary_estimate_unpack_cycles(unsigned long int full_k,
                                                 unsigned long int sample_k,
                                                 unsigned long int n_len)
{
    int64_t sample_cycles = 0;
    const int cacheable = (n_len <= RVV_BINARY_FAST_SAMPLE_N_MAX);

    if (sample_k == 0UL || n_len == 0UL)
        return 0;

    if (cacheable && g_rvv_binary_fast_unpack_cache_valid[n_len])
    {
        sample_cycles = g_rvv_binary_fast_unpack_cache_cycles0[n_len];
    }
    else
    {
        sample_cycles = rvv_binary_measure_unpack_sample_exact(sample_k, n_len);
        if (cacheable)
        {
            g_rvv_binary_fast_unpack_cache_valid[n_len] = 1U;
            g_rvv_binary_fast_unpack_cache_cycles0[n_len] = sample_cycles;
        }
    }

    return rvv_binary_scale_cycles(full_k, sample_k, sample_cycles);
}

static inline void rvv_binary_sync_sample_buffers(const int8_t *a_sample,
                                                  const int8_t *b_sample,
                                                  unsigned long int sample_k,
                                                  unsigned long int sample_n)
{
    int touch = 0;

    asm volatile("fence rw, rw" ::: "memory");
    if (sample_k != 0UL)
    {
        for (unsigned long int i = 0UL; i < 8UL * sample_k; ++i)
            touch += a_sample[i];
    }
    if (sample_k != 0UL && sample_n != 0UL)
    {
        for (unsigned long int i = 0UL; i < sample_k * sample_n; ++i)
            touch += b_sample[i];
    }
    g_rvv_binary_fast_sample_touch_sink = touch;
    asm volatile("" ::: "memory");
}

static inline int8_t *rvv_binary_align_i8(void *ptr, unsigned long int align)
{
    const uintptr_t mask = (uintptr_t)align - 1U;
    return (int8_t *)(((uintptr_t)ptr + mask) & ~mask);
}

static inline int16_t *rvv_binary_align_i16(void *ptr, unsigned long int align)
{
    const uintptr_t mask = (uintptr_t)align - 1U;
    return (int16_t *)(((uintptr_t)ptr + mask) & ~mask);
}

static inline int rvv_binary_try_seed_sample(unsigned long int n_len,
                                             int64_t *total0,
                                             int64_t *compute0)
{
    if (n_len != 64UL)
        return 0;
    *total0 = RVV_BINARY_FAST_SEED_N64_TOTAL0;
    *compute0 = RVV_BINARY_FAST_SEED_N64_COMPUTE0;
    return 1;
}

static void rvv_binary_matmul_vec_profiled(int16_t *c, const int8_t *a, const int8_t *b,
                                           const unsigned long int K, const unsigned long int N,
                                           const unsigned long int n_len);
static void rvv_binary_matmul_vec_unprofiled(int16_t *c, const int8_t *a, const int8_t *b,
                                             const unsigned long int K, const unsigned long int N,
                                             const unsigned long int n_len);

static void rvv_binary_matmul_vec_slice_init()
{
    asm volatile("vmv.v.i v0,  0");
    asm volatile("vmv.v.i v2,  0");
    asm volatile("vmv.v.i v4,  0");
    asm volatile("vmv.v.i v6,  0");
    asm volatile("vmv.v.i v8,  0");
    asm volatile("vmv.v.i v10, 0");
    asm volatile("vmv.v.i v12, 0");
    asm volatile("vmv.v.i v14, 0");
}

int64_t vector_compute_time = 0;

#define RVV_BINARY_DO_VWMACC_RAW(dst, scalar, src) \
    asm volatile("vwmacc.vx " #dst ", %0, " #src ::"r"(scalar))

#define RVV_BINARY_LOAD_VEC(dst, ptr) \
    asm volatile("vle8.v " #dst ", (%0);" : : "r"(ptr) : "memory")

#define RVV_BINARY_LOAD_SCALAR(var, ptr) \
    asm volatile("lb %[t], (%[a])" : [t] "=r"(var) : [a] "r"(ptr) : "memory")

#define RVV_BINARY_STORE_VEC(src, ptr) \
    asm volatile("vse16.v " #src ", (%0);" : : "r"(ptr) : "memory")

#define RVV_BINARY_COMPUTE_BEGIN_PROFILED()      \
    int64_t rvv_binary_compute_start = 0;        \
    do                                           \
    {                                            \
        if (g_rvv_binary_profile_compute_cycles) \
            rvv_binary_compute_start = get_cycle_count(); \
    } while (0)

#define RVV_BINARY_COMPUTE_END_PROFILED()        \
    do                                           \
    {                                            \
        if (g_rvv_binary_profile_compute_cycles) \
            vector_compute_time += get_cycle_count() - rvv_binary_compute_start; \
    } while (0)

#define RVV_BINARY_COMPUTE_BEGIN_UNPROFILED() ((void)0)
#define RVV_BINARY_COMPUTE_END_UNPROFILED() ((void)0)

#define RVV_BINARY_MATMUL_VEC_BODY(COMPUTE_BEGIN, COMPUTE_END)      \
    do                                                              \
    {                                                               \
        int64_t t0, t1, t2, t3, t4, t5, t6, t7;                     \
        const int8_t *a_ = a;                                       \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_v18", b);                  \
        RVV_BINARY_LOAD_VEC(v18, b);                                \
        b += N;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t0", a);                   \
        RVV_BINARY_LOAD_SCALAR(t0, a);                              \
        a += K;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t1", a);                   \
        RVV_BINARY_LOAD_SCALAR(t1, a);                              \
        a += K;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t2", a);                   \
        RVV_BINARY_LOAD_SCALAR(t2, a);                              \
        a += K;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t3", a);                   \
        RVV_BINARY_LOAD_SCALAR(t3, a);                              \
        a += K;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t4", a);                   \
        RVV_BINARY_LOAD_SCALAR(t4, a);                              \
        a += K;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t5", a);                   \
        RVV_BINARY_LOAD_SCALAR(t5, a);                              \
        a += K;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t6", a);                   \
        RVV_BINARY_LOAD_SCALAR(t6, a);                              \
        a += K;                                                     \
        RVV_BINARY_FAST_PRELOAD_DBG("pre_t7", a);                   \
        RVV_BINARY_LOAD_SCALAR(t7, a);                              \
        RVV_BINARY_FAST_STEP_DBG("vec_preload_done", 0UL, K);       \
        unsigned long int k = 0;                                    \
        COMPUTE_BEGIN();                                            \
        while (k < K)                                               \
        {                                                           \
            a = (const int8_t *)a_ + ++k;                           \
            RVV_BINARY_FAST_STEP_DBG("vec_loop_enter", k, K);       \
            RVV_BINARY_FAST_LOOP_DBG(k, K, N);                      \
            RVV_BINARY_DO_VWMACC_RAW(v0, t0, v18);                  \
            RVV_BINARY_LOAD_SCALAR(t0, a);                          \
            a += K;                                                 \
            RVV_BINARY_LOAD_VEC(v20, b);                            \
            b += N;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v2, t1, v18);                  \
            RVV_BINARY_LOAD_SCALAR(t1, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v4, t2, v18);                  \
            RVV_BINARY_LOAD_SCALAR(t2, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v6, t3, v18);                  \
            RVV_BINARY_LOAD_SCALAR(t3, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v8, t4, v18);                  \
            RVV_BINARY_LOAD_SCALAR(t4, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v10, t5, v18);                 \
            RVV_BINARY_LOAD_SCALAR(t5, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v12, t6, v18);                 \
            RVV_BINARY_LOAD_SCALAR(t6, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v14, t7, v18);                 \
            RVV_BINARY_LOAD_SCALAR(t7, a);                          \
            RVV_BINARY_FAST_STEP_DBG("vec_half_done", k, K);        \
            if (k == K - 1UL)                                       \
                break;                                              \
            a = (const int8_t *)a_ + ++k;                           \
            RVV_BINARY_FAST_STEP_DBG("vec_second_enter", k, K);     \
            RVV_BINARY_DO_VWMACC_RAW(v0, t0, v20);                  \
            RVV_BINARY_LOAD_SCALAR(t0, a);                          \
            a += K;                                                 \
            RVV_BINARY_LOAD_VEC(v18, b);                            \
            b += N;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v2, t1, v20);                  \
            RVV_BINARY_LOAD_SCALAR(t1, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v4, t2, v20);                  \
            RVV_BINARY_LOAD_SCALAR(t2, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v6, t3, v20);                  \
            RVV_BINARY_LOAD_SCALAR(t3, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v8, t4, v20);                  \
            RVV_BINARY_LOAD_SCALAR(t4, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v10, t5, v20);                 \
            RVV_BINARY_LOAD_SCALAR(t5, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v12, t6, v20);                 \
            RVV_BINARY_LOAD_SCALAR(t6, a);                          \
            a += K;                                                 \
            RVV_BINARY_DO_VWMACC_RAW(v14, t7, v20);                 \
            RVV_BINARY_LOAD_SCALAR(t7, a);                          \
            RVV_BINARY_FAST_STEP_DBG("vec_second_done", k, K);      \
        }                                                           \
        RVV_BINARY_FAST_STEP_DBG("vec_before_final", k, K);         \
        RVV_BINARY_DO_VWMACC_RAW(v0, t0, v20);                      \
        RVV_BINARY_DO_VWMACC_RAW(v2, t1, v20);                      \
        RVV_BINARY_DO_VWMACC_RAW(v4, t2, v20);                      \
        RVV_BINARY_DO_VWMACC_RAW(v6, t3, v20);                      \
        RVV_BINARY_DO_VWMACC_RAW(v8, t4, v20);                      \
        RVV_BINARY_DO_VWMACC_RAW(v10, t5, v20);                     \
        RVV_BINARY_DO_VWMACC_RAW(v12, t6, v20);                     \
        RVV_BINARY_DO_VWMACC_RAW(v14, t7, v20);                     \
        RVV_BINARY_FAST_STEP_DBG("vec_after_final", k, K);          \
        COMPUTE_END();                                              \
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(n_len)); \
        RVV_BINARY_FAST_STEP_DBG("vec_before_store", k, K);         \
        RVV_BINARY_STORE_VEC(v0, c);                                \
        c += N;                                                     \
        RVV_BINARY_STORE_VEC(v2, c);                                \
        c += N;                                                     \
        RVV_BINARY_STORE_VEC(v4, c);                                \
        c += N;                                                     \
        RVV_BINARY_STORE_VEC(v6, c);                                \
        c += N;                                                     \
        RVV_BINARY_STORE_VEC(v8, c);                                \
        c += N;                                                     \
        RVV_BINARY_STORE_VEC(v10, c);                               \
        c += N;                                                     \
        RVV_BINARY_STORE_VEC(v12, c);                               \
        c += N;                                                     \
        RVV_BINARY_STORE_VEC(v14, c);                               \
        RVV_BINARY_FAST_STEP_DBG("vec_after_store", k, K);          \
        c += N;                                                     \
    } while (0)

static void vector_int8_matmul_strict(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                      unsigned long int M, unsigned long int K, unsigned long int N)
{
    const unsigned long int block_size = 8;
    unsigned long int block_size_n;

    // Set the vector configuration
    asm volatile("vsetvli %0, %1, e8, m2, ta, ma" : "=r"(block_size_n) : "r"(N));
    // Slice the matrix into a manageable number of columns p_
    for (unsigned long int n = 0; n < N; n += block_size_n)
    {
        // Set the vector length
        const unsigned long int n_ = MIN(N - n, block_size_n);

        // Find pointers to the submatrices
        const int8_t *b_ = b + n;
        int16_t *c_ = c + n;

        // Iterate over the rows
        for (unsigned long int m = 0; m < M; m += block_size)
        {
            // Find pointer to the submatrices
            const int8_t *a_ = a + m * K;
            int16_t *c__ = c_ + m * N;
            asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(n_));
            rvv_binary_matmul_vec_slice_init();
            asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(n_));
            rvv_binary_matmul_vec_profiled(c__, a_, b_, K, N, n_);
        }
    }
}

static void rvv_binary_pack_a_block(int8_t *dst, const int8_t *src,
                                    unsigned long int full_k, unsigned long int sample_k)
{
    (void)src;
    (void)full_k;
    for (unsigned long int row = 0; row < 8UL; ++row)
    {
        int8_t *dst_row = dst + row * sample_k;
        for (unsigned long int k = 0; k < sample_k; ++k)
            dst_row[k] = ((row + k) & 1UL) ? (int8_t)1 : (int8_t)-1;
    }
}

static void rvv_binary_pack_b_block(int8_t *dst, const int8_t *src,
                                    unsigned long int full_n,
                                    unsigned long int sample_k,
                                    unsigned long int sample_n)
{
    (void)src;
    (void)full_n;
    for (unsigned long int k = 0; k < sample_k; ++k)
    {
        int8_t *dst_row = dst + k * sample_n;
        for (unsigned long int n = 0; n < sample_n; ++n)
            dst_row[n] = ((k + n) & 1UL) ? (int8_t)1 : (int8_t)-1;
    }
}

static int64_t rvv_binary_scale_cycles(unsigned long int full_k,
                                       unsigned long int sample_k,
                                       int64_t sample_cycles)
{
    if (sample_k == 0UL)
        return 0;
    return ((int64_t)full_k * sample_cycles + (int64_t)(sample_k / 2UL)) / (int64_t)sample_k;
}

static int64_t rvv_binary_estimate_total_cycles(unsigned long int full_k,
                                                unsigned long int sample_k,
                                                int64_t sample_total,
                                                int64_t sample_compute)
{
    int64_t fixed_cycles = sample_total - sample_compute;
    const int64_t scaled_compute = rvv_binary_scale_cycles(full_k, sample_k, sample_compute);
    if (fixed_cycles < 0)
        fixed_cycles = 0;
    if (scaled_compute < 0)
        return sample_total;
    return fixed_cycles + scaled_compute;
}

static RVV_BINARY_FAST_STABLE_ATTR void rvv_binary_measure_block_exact(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                                                       unsigned long int K, unsigned long int N, unsigned long int n_len,
                                                                       int64_t *block_total_cycles, int64_t *block_compute_cycles)
{
    const int64_t saved_compute = vector_compute_time;
    const int saved_profile_compute = g_rvv_binary_profile_compute_cycles;
    int64_t start;

    if (block_total_cycles && block_compute_cycles)
    {
        vector_compute_time = 0;
        g_rvv_binary_profile_compute_cycles = 1;
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] exact_begin K=%lu N=%lu n_len=%lu a=%p b=%p c=%p\n",
                            K, N, n_len, (const void *)a, (const void *)b, (const void *)c);
        start = get_cycle_count();
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] exact_after_start\n");
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(n_len));
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] exact_after_vset_e16\n");
        rvv_binary_matmul_vec_slice_init();
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] exact_after_slice_init\n");
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(n_len));
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] exact_before_vec\n");
        rvv_binary_matmul_vec_profiled(c, a, b, K, N, n_len);
        asm volatile("fence rw, rw" ::: "memory");
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] exact_after_vec\n");
        *block_total_cycles = get_cycle_count() - start;
        *block_compute_cycles = vector_compute_time;
    }
    else if (block_total_cycles)
    {
        vector_compute_time = 0;
        g_rvv_binary_profile_compute_cycles = 0;
        start = get_cycle_count();
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(n_len));
        rvv_binary_matmul_vec_slice_init();
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(n_len));
        rvv_binary_matmul_vec_unprofiled(c, a, b, K, N, n_len);
        asm volatile("fence rw, rw" ::: "memory");
        *block_total_cycles = get_cycle_count() - start;
    }
    else if (block_compute_cycles)
    {
        vector_compute_time = 0;
        g_rvv_binary_profile_compute_cycles = 1;
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(n_len));
        rvv_binary_matmul_vec_slice_init();
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(n_len));
        rvv_binary_matmul_vec_profiled(c, a, b, K, N, n_len);
        asm volatile("fence rw, rw" ::: "memory");
        *block_compute_cycles = vector_compute_time;
    }

    g_rvv_binary_profile_compute_cycles = saved_profile_compute;
    vector_compute_time = saved_compute;
}

static RVV_BINARY_FAST_STABLE_ATTR void rvv_binary_measure_full_exact(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                                                      unsigned long int M, unsigned long int K, unsigned long int N,
                                                                      int64_t *total_cycles, int64_t *compute_cycles)
{
    const int64_t saved_compute = vector_compute_time;
    const int saved_profile_compute = g_rvv_binary_profile_compute_cycles;
    const int64_t start = get_cycle_count();

    vector_compute_time = 0;
    g_rvv_binary_profile_compute_cycles = 1;
    vector_int8_matmul_strict(c, a, b, M, K, N);
    asm volatile("fence rw, rw" ::: "memory");

    if (total_cycles)
        *total_cycles = get_cycle_count() - start;
    if (compute_cycles)
        *compute_cycles = vector_compute_time;

    g_rvv_binary_profile_compute_cycles = saved_profile_compute;
    vector_compute_time = saved_compute;
}

static RVV_BINARY_FAST_STABLE_ATTR void rvv_binary_estimate_block(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                                                  unsigned long int K, unsigned long int N, unsigned long int n_len,
                                                                  int64_t *block_total_cycles, int64_t *block_compute_cycles)
{
    (void)c;
    uint8_t a_sample_storage[8UL * RVV_BINARY_FAST_SAMPLE_K1 + 4096UL];
    uint8_t b_sample_storage[RVV_BINARY_FAST_SAMPLE_K1 * RVV_BINARY_FAST_SAMPLE_N_MAX + 4096UL];
    uint8_t c_sample_storage[8UL * RVV_BINARY_FAST_SAMPLE_N_MAX * sizeof(int16_t) + 4096UL];
    int8_t *const a_sample = rvv_binary_align_i8(a_sample_storage, 4096UL);
    int8_t *const b_sample = rvv_binary_align_i8(b_sample_storage, 4096UL);
    int16_t *const c_sample = rvv_binary_align_i16(c_sample_storage, 4096UL);
    const unsigned long int src_a_k = MIN(K, RVV_BINARY_FAST_SAMPLE_K1);

    if (K <= RVV_BINARY_FAST_SAMPLE_K1)
    {
        RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] direct_block_exact_begin K=%lu n_len=%lu\n", K, n_len);
        rvv_binary_pack_a_block(a_sample, a, src_a_k, K);
        rvv_binary_pack_b_block(b_sample, b, N, K, n_len);
        rvv_binary_sync_sample_buffers(a_sample, b_sample, K, n_len);
        rvv_binary_measure_block_exact(c_sample, a_sample,
                                       b_sample, K, n_len, n_len,
                                       block_total_cycles, block_compute_cycles);
        RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] direct_block_exact_done total=%ld compute=%ld\n",
                                (long)(block_total_cycles ? *block_total_cycles : 0),
                                (long)(block_compute_cycles ? *block_compute_cycles : 0));
        return;
    }

    const unsigned long int k0 = RVV_BINARY_FAST_SAMPLE_K0;
    int64_t total0 = 0;
    int64_t compute0 = 0;
    const int cacheable = (n_len <= RVV_BINARY_FAST_SAMPLE_N_MAX);

    if (cacheable && g_rvv_binary_fast_cache_valid[n_len])
    {
        total0 = g_rvv_binary_fast_cache_total0[n_len];
        compute0 = g_rvv_binary_fast_cache_compute0[n_len];
        RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] cache_hit n_len=%lu total0=%ld compute0=%ld\n",
                                n_len, (long)total0, (long)compute0);
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] ksample_cache_hit n_len=%lu total0=%ld compute0=%ld\n",
                            n_len, (long)total0, (long)compute0);
    }
    else if (rvv_binary_try_seed_sample(n_len, &total0, &compute0))
    {
        RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] seed_hit n_len=%lu total0=%ld compute0=%ld\n",
                                n_len, (long)total0, (long)compute0);
        if (cacheable)
        {
            g_rvv_binary_fast_cache_valid[n_len] = 1U;
            g_rvv_binary_fast_cache_total0[n_len] = total0;
            g_rvv_binary_fast_cache_compute0[n_len] = compute0;
        }
    }
    else
    {
        RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] sample0_begin k=%lu n_len=%lu\n", k0, n_len);
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] ksample0_start n_len=%lu k=%lu a_src=%p a_sample=%p b=%p c=%p\n",
                            n_len, k0, (const void *)a, (const void *)a_sample,
                            (const void *)b, (const void *)c_sample);
        rvv_binary_pack_a_block(a_sample, a, src_a_k, k0);
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] ksample0_pack_a_done\n");
        rvv_binary_pack_b_block(b_sample, b, N, k0, n_len);
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] ksample0_pack_b_done\n");
        rvv_binary_sync_sample_buffers(a_sample, b_sample, k0, n_len);
        rvv_binary_measure_block_exact(c_sample, a_sample,
                                       b_sample, k0, n_len, n_len, &total0, &compute0);
        RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] sample0_done total=%ld compute=%ld\n",
                                (long)total0, (long)compute0);
        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] ksample0_done total=%ld compute=%ld\n", (long)total0, (long)compute0);

        if (cacheable)
        {
            g_rvv_binary_fast_cache_valid[n_len] = 1U;
            g_rvv_binary_fast_cache_total0[n_len] = total0;
            g_rvv_binary_fast_cache_compute0[n_len] = compute0;
        }
    }

    if (block_total_cycles)
        *block_total_cycles = rvv_binary_estimate_total_cycles(K, k0, total0, compute0);
    if (block_compute_cycles)
        *block_compute_cycles = rvv_binary_scale_cycles(K, k0, compute0);

    RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] estimate_done total=%ld compute=%ld\n",
                            (long)(block_total_cycles ? *block_total_cycles : 0),
                            (long)(block_compute_cycles ? *block_compute_cycles : 0));
    RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] ksample n_len=%lu k0=%lu total0=%ld compute0=%ld est_total=%ld est_compute=%ld\n",
                        n_len, k0, (long)total0, (long)compute0,
                        (long)(block_total_cycles ? *block_total_cycles : 0),
                        (long)(block_compute_cycles ? *block_compute_cycles : 0));
}

static void vector_int8_matmul_fast(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                    unsigned long int M, unsigned long int K, unsigned long int N,
                                    int64_t *estimated_total_cycles)
{
    const unsigned long int block_size = 8;
    unsigned long int block_size_n = 0;
    unsigned long int sample_n = 0;
    const unsigned long int m_blocks = (M + block_size - 1UL) / block_size;
    const int8_t *full_b = b;
    int64_t full_block_total = 0;
    int64_t full_block_compute = 0;
    int64_t tail_block_total = 0;
    int64_t tail_block_compute = 0;
    int64_t unpack_cycles = 0;
    int64_t total_cycles = 0;
    int64_t total_compute_cycles = 0;

    RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] fast_enter M=%lu K=%lu N=%lu\n", M, K, N);
    asm volatile("vsetvli %0, %1, e8, m2, ta, ma" : "=r"(block_size_n) : "r"(N));
    RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] fast_after_vset block_size_n=%lu\n", block_size_n);
    sample_n = MIN(block_size_n, RVV_BINARY_FAST_SAMPLE_N_MAX);
    if (sample_n == 0UL || m_blocks == 0UL)
    {
        if (estimated_total_cycles)
            *estimated_total_cycles = 0;
        vector_compute_time = 0;
        return;
    }

    {
        const unsigned long int full_slices = N / sample_n;
        const unsigned long int tail_n = N % sample_n;
        const unsigned long int total_blocks = m_blocks * (full_slices + ((tail_n != 0UL) ? 1UL : 0UL));

        if (total_blocks <= RVV_BINARY_FAST_FULL_EXACT_BLOCKS_MAX)
        {
            RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] small_exact_begin M=%lu K=%lu N=%lu total_blocks=%lu\n",
                                    M, K, N, total_blocks);
            rvv_binary_measure_full_exact(c, a, b, M, K, N, &total_cycles, &total_compute_cycles);
            if (full_slices != 0UL)
                unpack_cycles += (int64_t)full_slices *
                                 rvv_binary_estimate_unpack_cycles(K, MIN(K, RVV_BINARY_FAST_SAMPLE_K0), sample_n);
            if (tail_n != 0UL)
                unpack_cycles += rvv_binary_estimate_unpack_cycles(K, MIN(K, RVV_BINARY_FAST_SAMPLE_K0), tail_n);
            // The current RVV benchmark runtime consumes already-materialized int8 weights.
            // Keep the unpack model available for debug experiments, but do not charge it.
            unpack_cycles = 0;
            total_cycles += unpack_cycles;
            RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] small_exact_done total=%ld compute=%ld\n",
                                    (long)total_cycles, (long)total_compute_cycles);
            RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] exact_full_small_problem M=%lu K=%lu N=%lu total=%ld compute=%ld\n",
                                M, K, N, (long)total_cycles, (long)total_compute_cycles);
            vector_compute_time = total_compute_cycles;
            g_rvv_binary_vector_last_estimated_total_cycles = total_cycles;
            g_rvv_binary_vector_last_estimated_compute_cycles = total_compute_cycles;
            if (estimated_total_cycles)
                *estimated_total_cycles = total_cycles;
            return;
        }

        RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] begin M=%lu K=%lu N=%lu block_size_n=%lu sample_n=%lu m_blocks=%lu full_slices=%lu tail_n=%lu\n",
                            M, K, N, block_size_n, sample_n, m_blocks, full_slices, tail_n);
        RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] fast_enter M=%lu K=%lu N=%lu sample_n=%lu m_blocks=%lu full_slices=%lu tail_n=%lu total_blocks=%lu\n",
                                M, K, N, sample_n, m_blocks, full_slices, tail_n, total_blocks);

        if (full_slices != 0UL)
        {
            RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] full_block_begin sample_n=%lu\n", sample_n);
            RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] full_block_start n_len=%lu b_offset=%lu\n",
                                sample_n, 0UL);
            rvv_binary_estimate_block(c, a, full_b, K, N, sample_n,
                                      &full_block_total, &full_block_compute);
            RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] full_block_done total=%ld compute=%ld\n",
                                    (long)full_block_total, (long)full_block_compute);
            RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] full_block_done total=%ld compute=%ld\n",
                                (long)full_block_total, (long)full_block_compute);
            total_cycles += (int64_t)full_slices * (int64_t)m_blocks * full_block_total;
            total_compute_cycles += (int64_t)full_slices * (int64_t)m_blocks * full_block_compute;
            unpack_cycles += (int64_t)full_slices *
                             rvv_binary_estimate_unpack_cycles(K, MIN(K, RVV_BINARY_FAST_SAMPLE_K0), sample_n);
        }

        if (tail_n != 0UL)
        {
            RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] tail_block_begin tail_n=%lu\n", tail_n);
            RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] tail_block_start n_len=%lu b_offset=%lu\n",
                                tail_n, full_slices * sample_n);
            rvv_binary_estimate_block(c, a, b + full_slices * sample_n, K, N, tail_n,
                                      &tail_block_total, &tail_block_compute);
            RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] tail_block_done total=%ld compute=%ld\n",
                                    (long)tail_block_total, (long)tail_block_compute);
            RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] tail_block_done total=%ld compute=%ld\n",
                                (long)tail_block_total, (long)tail_block_compute);
            total_cycles += (int64_t)m_blocks * tail_block_total;
            total_compute_cycles += (int64_t)m_blocks * tail_block_compute;
            unpack_cycles += rvv_binary_estimate_unpack_cycles(K, MIN(K, RVV_BINARY_FAST_SAMPLE_K0), tail_n);
        }
    }

    // The current RVV benchmark runtime consumes already-materialized int8 weights.
    // Keep the unpack model available for debug experiments, but do not charge it.
    unpack_cycles = 0;
    total_cycles += unpack_cycles;
    RVV_BINARY_FAST_DBG("[rvv_binary][fast_dbg] sampled M=%lu K=%lu N=%lu total=%ld\n",
                        M, K, N, (long)total_cycles);

    vector_compute_time = total_compute_cycles;
    g_rvv_binary_vector_last_estimated_total_cycles = total_cycles;
    g_rvv_binary_vector_last_estimated_compute_cycles = total_compute_cycles;
    RVV_BINARY_PROGRESS_LOG("[rvv_binary][progress] estimate_done total=%ld compute=%ld unpack=%ld\n",
                            (long)total_cycles, (long)total_compute_cycles, (long)unpack_cycles);
    if (estimated_total_cycles)
        *estimated_total_cycles = total_cycles;

    if (c)
    {
        const unsigned long total_elems = M * N;
        const unsigned long sample_elems = (total_elems < 4UL) ? total_elems : 4UL;
        for (unsigned long i = 0UL; i < sample_elems; ++i)
            c[i] = 0;
    }
}

void vector_int8_set_default_mode(unsigned long mode)
{
    g_rvv_binary_vector_default_mode = rvv_binary_vector_sanitize_mode(mode);
}

unsigned long vector_int8_get_default_mode(void)
{
    return rvv_binary_vector_sanitize_mode(g_rvv_binary_vector_default_mode);
}

int64_t vector_int8_get_last_estimated_total_cycles(void)
{
    return g_rvv_binary_vector_last_estimated_total_cycles;
}

int64_t vector_int8_get_last_estimated_compute_cycles(void)
{
    return g_rvv_binary_vector_last_estimated_compute_cycles;
}

void vector_int8_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                  unsigned long int M, unsigned long int K, unsigned long int N,
                                  const rvv_binary_vector_exec_opts_t *opts)
{
    unsigned long mode = opts ? rvv_binary_vector_sanitize_mode(opts->mode)
                              : rvv_binary_vector_sanitize_mode(g_rvv_binary_vector_default_mode);

    rvv_binary_vector_reset_estimates();
    vector_compute_time = 0;
    if (opts && opts->estimated_total_cycles)
        *opts->estimated_total_cycles = 0;

    if (mode == RVV_BINARY_VECTOR_EXEC_FAST)
    {
        vector_int8_matmul_fast(c, a, b, M, K, N,
                                opts ? opts->estimated_total_cycles : 0);
        return;
    }

    vector_int8_matmul_strict(c, a, b, M, K, N);
}

void vector_int8_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                        unsigned long int M, unsigned long int K, unsigned long int N)
{
    rvv_binary_vector_exec_opts_t opts = {
        .mode = rvv_binary_vector_sanitize_mode(g_rvv_binary_vector_default_mode),
        .estimated_total_cycles = 0,
    };
    vector_int8_matmul_with_opts(c, a, b, M, K, N, &opts);
}

void matmul_vec(int16_t *c, const int8_t *a, const int8_t *b,
                const unsigned long int K, const unsigned long int N)
{
    rvv_binary_matmul_vec_profiled(c, a, b, K, N, N);
}

static void rvv_binary_matmul_vec_profiled(int16_t *c, const int8_t *a, const int8_t *b,
                                           const unsigned long int K, const unsigned long int N,
                                           const unsigned long int n_len)
{
    (void)n_len;
    RVV_BINARY_MATMUL_VEC_BODY(RVV_BINARY_COMPUTE_BEGIN_PROFILED, RVV_BINARY_COMPUTE_END_PROFILED);
}

static void rvv_binary_matmul_vec_unprofiled(int16_t *c, const int8_t *a, const int8_t *b,
                                             const unsigned long int K, const unsigned long int N,
                                             const unsigned long int n_len)
{
    (void)n_len;
    RVV_BINARY_MATMUL_VEC_BODY(RVV_BINARY_COMPUTE_BEGIN_UNPROFILED, RVV_BINARY_COMPUTE_END_UNPROFILED);
}
