#include "vector.h"
#include "runtime.h"

// helper macro used by the vector kernel only
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static unsigned long g_rvv_binary_vector_default_mode = RVV_BINARY_VECTOR_DEFAULT_MODE;
static int64_t g_rvv_binary_vector_last_estimated_total_cycles = 0;
static int64_t g_rvv_binary_vector_last_estimated_compute_cycles = 0;

static inline unsigned long rvv_binary_vector_sanitize_mode(unsigned long mode)
{
    return (mode == RVV_BINARY_VECTOR_EXEC_FAST) ? RVV_BINARY_VECTOR_EXEC_FAST : RVV_BINARY_VECTOR_EXEC_STRICT;
}

static inline void rvv_binary_vector_reset_estimates(void)
{
    g_rvv_binary_vector_last_estimated_total_cycles = 0;
    g_rvv_binary_vector_last_estimated_compute_cycles = 0;
}

void matmul_vec_slice_init()
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
            matmul_vec_slice_init();
            asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(n_));
            matmul_vec(c__, a_, b_, K, N);
        }
    }
}

static void rvv_binary_measure_block(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                     unsigned long int K, unsigned long int N, unsigned long int n_len,
                                     int64_t *block_total_cycles, int64_t *block_compute_cycles)
{
    const int64_t saved_compute = vector_compute_time;
    int64_t start;

    vector_compute_time = 0;
    start = get_cycle_count();
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(n_len));
    matmul_vec_slice_init();
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(n_len));
    matmul_vec(c, a, b, K, N);
    if (block_total_cycles)
        *block_total_cycles = get_cycle_count() - start;
    if (block_compute_cycles)
        *block_compute_cycles = vector_compute_time;
    vector_compute_time = saved_compute;
}

static void vector_int8_matmul_fast(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                                    unsigned long int M, unsigned long int K, unsigned long int N,
                                    int64_t *estimated_total_cycles)
{
    const unsigned long int block_size = 8;
    unsigned long int block_size_n = 0;
    const unsigned long int m_blocks = (M + block_size - 1UL) / block_size;
    const int8_t *full_b = b;
    int64_t full_block_total = 0;
    int64_t full_block_compute = 0;
    int64_t tail_block_total = 0;
    int64_t tail_block_compute = 0;
    int64_t total_cycles = 0;
    int64_t total_compute_cycles = 0;

    asm volatile("vsetvli %0, %1, e8, m2, ta, ma" : "=r"(block_size_n) : "r"(N));
    if (block_size_n == 0UL || m_blocks == 0UL)
    {
        if (estimated_total_cycles)
            *estimated_total_cycles = 0;
        vector_compute_time = 0;
        return;
    }

    {
        const unsigned long int full_slices = N / block_size_n;
        const unsigned long int tail_n = N % block_size_n;

        if (full_slices != 0UL)
        {
            rvv_binary_measure_block(c, a, full_b, K, N, block_size_n,
                                     &full_block_total, &full_block_compute);
            total_cycles += (int64_t)full_slices * (int64_t)m_blocks * full_block_total;
            total_compute_cycles += (int64_t)full_slices * (int64_t)m_blocks * full_block_compute;
        }

        if (tail_n != 0UL)
        {
            rvv_binary_measure_block(c, a, b + full_slices * block_size_n, K, N, tail_n,
                                     &tail_block_total, &tail_block_compute);
            total_cycles += (int64_t)m_blocks * tail_block_total;
            total_compute_cycles += (int64_t)m_blocks * tail_block_compute;
        }
    }

    vector_compute_time = total_compute_cycles;
    g_rvv_binary_vector_last_estimated_total_cycles = total_cycles;
    g_rvv_binary_vector_last_estimated_compute_cycles = total_compute_cycles;
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
    // Temporary variables
    int8_t t0, t1, t2, t3, t4, t5, t6, t7;

    // Original pointer
    const int8_t *a_ = a;

    // Prefetch one row of matrix B
    asm volatile("vle8.v v18, (%0);" ::"r"(b));
    b += N;

    // Prefetch one row of scalar values
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
    a += K;
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
    a += K;
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
    a += K;
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
    a += K;
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
    a += K;
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
    a += K;
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
    a += K;
    asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));

    // Compute the multiplication
    unsigned long int k = 0;

    while (k < K)
    {
        // Calculate pointer to the matrix A
        a = (const int8_t *)a_ + ++k;

        asm volatile("vle8.v v20, (%0);" ::"r"(b));
        b += N;
        int64_t start = get_cycle_count();
        asm volatile("vwmacc.vx v0, %0, v18" ::"r"(t0));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v2, %0, v18" ::"r"(t1));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v4, %0, v18" ::"r"(t2));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v6, %0, v18" ::"r"(t3));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v8, %0, v18" ::"r"(t4));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v10, %0, v18" ::"r"(t5));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v12, %0, v18" ::"r"(t6));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v14, %0, v18" ::"r"(t7));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));

        // Load one row of B
        asm volatile("vle8.v v18, (%0);" ::"r"(b));
        b += N;

        if (k == K - 1)
            break;

        a = (const int8_t *)a_ + ++k;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t0));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t1));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t3));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v8, %0, v20" ::"r"(t4));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t5));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t6));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
        a += K;
        start = get_cycle_count();
        asm volatile("vwmacc.vx v14, %0, v20" ::"r"(t7));
        vector_compute_time += get_cycle_count() - start;
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));
    }

    // Last iteration: store results
    int64_t start = get_cycle_count();
    asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t0));
    asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t1));
    asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2));
    asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t3));
    asm volatile("vwmacc.vx v8, %0, v20" ::"r"(t4));
    asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t5));
    asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t6));
    asm volatile("vwmacc.vx v14, %0, v20" ::"r"(t7));
    vector_compute_time += get_cycle_count() - start;
    asm volatile("vsetivli zero, 0, e16, m2, ta, ma");
    asm volatile("vse16.v v0, (%0);" ::"r"(c));
    c += N;
    asm volatile("vse16.v v2, (%0);" ::"r"(c));
    c += N;
    asm volatile("vse16.v v4, (%0);" ::"r"(c));
    c += N;
    asm volatile("vse16.v v6, (%0);" ::"r"(c));
    c += N;
    asm volatile("vse16.v v8, (%0);" ::"r"(c));
    c += N;
    asm volatile("vse16.v v10, (%0);" ::"r"(c));
    c += N;
    asm volatile("vse16.v v12, (%0);" ::"r"(c));
    c += N;
    asm volatile("vse16.v v14, (%0);" ::"r"(c));
    c += N;
}
