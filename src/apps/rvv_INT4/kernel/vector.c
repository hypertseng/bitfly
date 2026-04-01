#include "vector.h"
#include "runtime.h"

// helper macro used by the vector kernel only
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static unsigned long g_rvv_int4_vector_default_mode = RVV_INT4_VECTOR_DEFAULT_MODE;
static int64_t g_rvv_int4_vector_last_estimated_total_cycles = 0;
static int64_t g_rvv_int4_vector_last_estimated_compute_cycles = 0;

static inline unsigned long rvv_int4_vector_sanitize_mode(unsigned long mode)
{
    return (mode == RVV_INT4_VECTOR_EXEC_FAST) ? RVV_INT4_VECTOR_EXEC_FAST : RVV_INT4_VECTOR_EXEC_STRICT;
}

static inline void rvv_int4_vector_reset_estimates(void)
{
    g_rvv_int4_vector_last_estimated_total_cycles = 0;
    g_rvv_int4_vector_last_estimated_compute_cycles = 0;
}

static void rvv_int4_matmul_vec_profiled(int16_t *c, const int8_t *a, const int8_t *b,
                                         const unsigned long int K, const unsigned long int N);
static void rvv_int4_matmul_vec_unprofiled(int16_t *c, const int8_t *a, const int8_t *b,
                                           const unsigned long int K, const unsigned long int N);

static void rvv_int4_matmul_vec_slice_init()
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

#define RVV_INT4_DO_VWMACC_RAW(dst, scalar, src) \
    asm volatile("vwmacc.vx " #dst ", %0, " #src ::"r"(scalar))

#define RVV_INT4_DO_VWMACC_PROFILED(dst, scalar, src) \
    do                                                \
    {                                                 \
        int64_t start = get_cycle_count();            \
        RVV_INT4_DO_VWMACC_RAW(dst, scalar, src);     \
        vector_compute_time += get_cycle_count() - start; \
    } while (0)

#define RVV_INT4_MATMUL_VEC_BODY(VWMACC_OP)                      \
    do                                                           \
    {                                                            \
        int8_t t0, t1, t2, t3, t4, t5, t6, t7;                   \
        const int8_t *a_ = a;                                    \
        asm volatile("vle8.v v18, (%0);" ::"r"(b));              \
        b += N;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a)); \
        a += K;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a)); \
        a += K;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a)); \
        a += K;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a)); \
        a += K;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a)); \
        a += K;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a)); \
        a += K;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a)); \
        a += K;                                                  \
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a)); \
        unsigned long int k = 0;                                 \
        while (k < K)                                            \
        {                                                        \
            a = (const int8_t *)a_ + ++k;                        \
            asm volatile("vle8.v v20, (%0);" ::"r"(b));          \
            b += N;                                              \
            VWMACC_OP(v0, t0, v18);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v2, t1, v18);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v4, t2, v18);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v6, t3, v18);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v8, t4, v18);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v10, t5, v18);                             \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v12, t6, v18);                             \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v14, t7, v18);                             \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a)); \
            asm volatile("vle8.v v18, (%0);" ::"r"(b));          \
            b += N;                                              \
            if (k == K - 1)                                      \
                break;                                           \
            a = (const int8_t *)a_ + ++k;                        \
            VWMACC_OP(v0, t0, v20);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v2, t1, v20);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v4, t2, v20);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v6, t3, v20);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v8, t4, v20);                              \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v10, t5, v20);                             \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v12, t6, v20);                             \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a)); \
            a += K;                                              \
            VWMACC_OP(v14, t7, v20);                             \
            asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a)); \
        }                                                        \
        VWMACC_OP(v0, t0, v20);                                  \
        VWMACC_OP(v2, t1, v20);                                  \
        VWMACC_OP(v4, t2, v20);                                  \
        VWMACC_OP(v6, t3, v20);                                  \
        VWMACC_OP(v8, t4, v20);                                  \
        VWMACC_OP(v10, t5, v20);                                 \
        VWMACC_OP(v12, t6, v20);                                 \
        VWMACC_OP(v14, t7, v20);                                 \
        asm volatile("vsetivli zero, 0, e16, m2, ta, ma");       \
        asm volatile("vse16.v v0, (%0);" ::"r"(c));              \
        c += N;                                                  \
        asm volatile("vse16.v v2, (%0);" ::"r"(c));              \
        c += N;                                                  \
        asm volatile("vse16.v v4, (%0);" ::"r"(c));              \
        c += N;                                                  \
        asm volatile("vse16.v v6, (%0);" ::"r"(c));              \
        c += N;                                                  \
        asm volatile("vse16.v v8, (%0);" ::"r"(c));              \
        c += N;                                                  \
        asm volatile("vse16.v v10, (%0);" ::"r"(c));             \
        c += N;                                                  \
        asm volatile("vse16.v v12, (%0);" ::"r"(c));             \
        c += N;                                                  \
        asm volatile("vse16.v v14, (%0);" ::"r"(c));             \
        c += N;                                                  \
    } while (0)

static void vector_int4_matmul_strict(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
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
            rvv_int4_matmul_vec_slice_init();
            asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(n_));
            rvv_int4_matmul_vec_profiled(c__, a_, b_, K, N);
        }
    }
}

static void rvv_int4_measure_block(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                     unsigned long int K, unsigned long int N, unsigned long int n_len,
                                     int64_t *block_total_cycles, int64_t *block_compute_cycles)
{
    const int64_t saved_compute = vector_compute_time;
    int64_t start;

    vector_compute_time = 0;
    start = get_cycle_count();
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(n_len));
    rvv_int4_matmul_vec_slice_init();
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(n_len));
    rvv_int4_matmul_vec_unprofiled(c, a, b, K, N);
    if (block_total_cycles)
        *block_total_cycles = get_cycle_count() - start;
    if (block_compute_cycles)
        *block_compute_cycles = vector_compute_time;
    vector_compute_time = saved_compute;
}

static void vector_int4_matmul_fast(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                    unsigned long int M, unsigned long int K, unsigned long int N,
                                    int64_t *estimated_total_cycles)
{
    const unsigned long int block_size = 8;
    const unsigned long int max_sample_n = 8UL;
    unsigned long int block_size_n = 0;
    unsigned long int sample_n = 0;
    const unsigned long int m_blocks = (M + block_size - 1UL) / block_size;
    const int8_t *full_b = b;
    int64_t full_block_total = 0;
    int64_t full_block_compute = 0;
    int64_t tail_block_total = 0;
    int64_t tail_block_compute = 0;
    int64_t total_cycles = 0;
    int64_t total_compute_cycles = 0;

    asm volatile("vsetvli %0, %1, e8, m2, ta, ma" : "=r"(block_size_n) : "r"(N));
    sample_n = (block_size_n < max_sample_n) ? block_size_n : max_sample_n;
    if (block_size_n == 0UL || m_blocks == 0UL)
    {
        if (estimated_total_cycles)
            *estimated_total_cycles = 0;
        vector_compute_time = 0;
        return;
    }

    {
        const unsigned long int full_slices = N / sample_n;
        const unsigned long int tail_n = N % sample_n;

        if (full_slices != 0UL)
        {
            rvv_int4_measure_block(c, a, full_b, K, N, sample_n,
                                     &full_block_total, &full_block_compute);
            total_cycles += (int64_t)full_slices * (int64_t)m_blocks * full_block_total;
            total_compute_cycles += (int64_t)full_slices * (int64_t)m_blocks * full_block_compute;
        }

        if (tail_n != 0UL)
        {
            rvv_int4_measure_block(c, a, b + full_slices * sample_n, K, N, tail_n,
                                     &tail_block_total, &tail_block_compute);
            total_cycles += (int64_t)m_blocks * tail_block_total;
            total_compute_cycles += (int64_t)m_blocks * tail_block_compute;
        }
    }

    vector_compute_time = total_compute_cycles;
    g_rvv_int4_vector_last_estimated_total_cycles = total_cycles;
    g_rvv_int4_vector_last_estimated_compute_cycles = total_compute_cycles;
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

void vector_int4_set_default_mode(unsigned long mode)
{
    g_rvv_int4_vector_default_mode = rvv_int4_vector_sanitize_mode(mode);
}

unsigned long vector_int4_get_default_mode(void)
{
    return rvv_int4_vector_sanitize_mode(g_rvv_int4_vector_default_mode);
}

int64_t vector_int4_get_last_estimated_total_cycles(void)
{
    return g_rvv_int4_vector_last_estimated_total_cycles;
}

int64_t vector_int4_get_last_estimated_compute_cycles(void)
{
    return g_rvv_int4_vector_last_estimated_compute_cycles;
}

void vector_int4_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                  unsigned long int M, unsigned long int K, unsigned long int N,
                                  const rvv_int4_vector_exec_opts_t *opts)
{
    unsigned long mode = opts ? rvv_int4_vector_sanitize_mode(opts->mode)
                              : rvv_int4_vector_sanitize_mode(g_rvv_int4_vector_default_mode);

    rvv_int4_vector_reset_estimates();
    vector_compute_time = 0;
    if (opts && opts->estimated_total_cycles)
        *opts->estimated_total_cycles = 0;

    if (mode == RVV_INT4_VECTOR_EXEC_FAST)
    {
        vector_int4_matmul_fast(c, a, b, M, K, N,
                                opts ? opts->estimated_total_cycles : 0);
        return;
    }

    vector_int4_matmul_strict(c, a, b, M, K, N);
}

void vector_int4_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                        unsigned long int M, unsigned long int K, unsigned long int N)
{
    rvv_int4_vector_exec_opts_t opts = {
        .mode = rvv_int4_vector_sanitize_mode(g_rvv_int4_vector_default_mode),
        .estimated_total_cycles = 0,
    };
    vector_int4_matmul_with_opts(c, a, b, M, K, N, &opts);
}

static void rvv_int4_matmul_vec_profiled(int16_t *c, const int8_t *a, const int8_t *b,
                                         const unsigned long int K, const unsigned long int N)
{
    RVV_INT4_MATMUL_VEC_BODY(RVV_INT4_DO_VWMACC_PROFILED);
}

static void rvv_int4_matmul_vec_unprofiled(int16_t *c, const int8_t *a, const int8_t *b,
                                           const unsigned long int K, const unsigned long int N)
{
    RVV_INT4_MATMUL_VEC_BODY(RVV_INT4_DO_VWMACC_RAW);
}
