#include "vector.h"
#include "runtime.h"

// helper macro used by the vector kernel only
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

void vector_int8_matmul(int16_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
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
