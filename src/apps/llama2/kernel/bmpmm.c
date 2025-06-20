#include "bmpmm.h"

// #define DEBUG
// 算子1: 混合精度矩阵乘法（使用自定义指令）
#define DEFINE_KERNEL(K_VAL)                                                               \
    void mixed_precision_matmul_##K_VAL(int16_t *result, const int8_t *a, const int8_t *w) \
    {                                                                                      \
        asm volatile("mpcfg " #K_VAL "\n\t");                                              \
        asm volatile("mple 0(%0), a\n\t" ::"r"(a) : "memory");                             \
        asm volatile("mple 0(%0), w\n\t" ::"r"(w) : "memory");                             \
        asm volatile("mpmm\n\t" ::);                                                       \
        asm volatile("mpse 0(%0)\n\t" ::"r"(result) : "memory");                           \
    }

// 实例化你需要的 K 值
DEFINE_KERNEL(16)
DEFINE_KERNEL(32)
DEFINE_KERNEL(64)
DEFINE_KERNEL(128)
DEFINE_KERNEL(256)
DEFINE_KERNEL(480)

mixed_kernel_func get_mixed_kernel(int K)
{
    switch (K)
    {
    case 16:
        return mixed_precision_matmul_16;
    case 32:
        return mixed_precision_matmul_32;
    case 64:
        return mixed_precision_matmul_64;
    case 128:
        return mixed_precision_matmul_128;
    case 256:
        return mixed_precision_matmul_256;
    case 480:
        return mixed_precision_matmul_480;
    }
}

void static inline get_config(unsigned long int K)
{
    switch (K)
    {
    case 8:
        asm volatile("mpcfg 8\n\t");
        break;
    case 16:
        asm volatile("mpcfg 16\n\t");
        break;
    case 24:
        asm volatile("mpcfg 24\n\t");
        break;
    case 32:
        asm volatile("mpcfg 32\n\t");
        break;
    case 40:
        asm volatile("mpcfg 40\n\t");
    case 48:
        asm volatile("mpcfg 48\n\t");
        break;
    case 56:
        asm volatile("mpcfg 56\n\t");
        break;
    case 64:
        asm volatile("mpcfg 64\n\t");
        break;
    case 72:
        asm volatile("mpcfg 72\n\t");
        break;
    case 80:
        asm volatile("mpcfg 80\n\t");
        break;
    case 88:
        asm volatile("mpcfg 88\n\t");
        break;
    case 96:
        asm volatile("mpcfg 96\n\t");
        break;
    case 104:
        asm volatile("mpcfg 104\n\t");
        break;
    case 112:
        asm volatile("mpcfg 112\n\t");
        break;
    case 120:
        asm volatile("mpcfg 120\n\t");
        break;
    case 128:
        asm volatile("mpcfg 128\n\t");
        break;
    case 136:
        asm volatile("mpcfg 136\n\t");
        break;
    case 144:
        asm volatile("mpcfg 144\n\t");
        break;
    case 152:
        asm volatile("mpcfg 152\n\t");
        break;
    case 160:
        asm volatile("mpcfg 160\n\t");
        break;
    case 168:
        asm volatile("mpcfg 168\n\t");
        break;
    case 176:
        asm volatile("mpcfg 176\n\t");
        break;
    case 184:
        asm volatile("mpcfg 184\n\t");
        break;
    case 192:
        asm volatile("mpcfg 192\n\t");
        break;
    case 200:
        asm volatile("mpcfg 200\n\t");
        break;
    case 208:
        asm volatile("mpcfg 208\n\t");
        break;
    case 216:
        asm volatile("mpcfg 216\n\t");
        break;
    case 224:
        asm volatile("mpcfg 224\n\t");
        break;
    case 232:
        asm volatile("mpcfg 232\n\t");
        break;
    case 240:
        asm volatile("mpcfg 240\n\t");
        break;
    case 248:
        asm volatile("mpcfg 248\n\t");
        break;
    case 256:
        asm volatile("mpcfg 256\n\t");
        break;
    case 264:
        asm volatile("mpcfg 264\n\t");
        break;
    case 272:
        asm volatile("mpcfg 272\n\t");
        break;
    case 280:
        asm volatile("mpcfg 280\n\t");
        break;
    case 288:
        asm volatile("mpcfg 288\n\t");
        break;
    case 296:
        asm volatile("mpcfg 296\n\t");
        break;
    case 304:
        asm volatile("mpcfg 304\n\t");
        break;
    case 312:
        asm volatile("mpcfg 312\n\t");
        break;
    case 320:
        asm volatile("mpcfg 320\n\t");
        break;
    case 328:
        asm volatile("mpcfg 328\n\t");
        break;
    case 336:
        asm volatile("mpcfg 336\n\t");
        break;
    case 344:
        asm volatile("mpcfg 344\n\t");
        break;
    case 352:
        asm volatile("mpcfg 352\n\t");
        break;
    case 360:
        asm volatile("mpcfg 360\n\t");
        break;
    case 368:
        asm volatile("mpcfg 368\n\t");
        break;
    case 376:
        asm volatile("mpcfg 376\n\t");
        break;
    case 384:
        asm volatile("mpcfg 384\n\t");
        break;
    case 392:
        asm volatile("mpcfg 392\n\t");
        break;
    case 400:
        asm volatile("mpcfg 400\n\t");
        break;
    case 408:
        asm volatile("mpcfg 408\n\t");
        break;
    case 416:
        asm volatile("mpcfg 416\n\t");
        break;
    case 424:
        asm volatile("mpcfg 424\n\t");
        break;
    case 432:
        asm volatile("mpcfg 432\n\t");
        break;
    case 440:
        asm volatile("mpcfg 440\n\t");
        break;
    case 448:
        asm volatile("mpcfg 448\n\t");
        break;
    case 456:
        asm volatile("mpcfg 456\n\t");
        break;
    case 464:
        asm volatile("mpcfg 464\n\t");
        break;
    case 472:
        asm volatile("mpcfg 472\n\t");
        break;
    case 480:
        asm volatile("mpcfg 480\n\t");
        break;
    }
}

void binary_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                         const unsigned long int M, const unsigned long int K,
                         const unsigned long int N)
{
    int tm = ceil(M / 16.0);
    int tn = ceil(N / 32.0);
    int tk = ceil(K / 480.0);
    if (tk > 1)
    {
        asm volatile("mpcfg 480\n\t");
    }
    for (int i = 0; i < tm; i++)
    {
        for (int j = 0; j < tn; j++)
        {
            int16_t *c_ = c + i * N * 16 + j * 32 * 16;
            for (int k = 0; k < tk; k++)
            {
                if (k == tk - 1)
                    get_config(K % 480);
                const int8_t *a_ = a + i * 16 * K + k * 16 * 480;
                const int8_t *b_ = b + j * 32 * K / 8 + k * 32 * 480 / 8;
                asm volatile("mple 0(%0), a\n\t" ::"r"(a_) : "memory");
                asm volatile("mple 0(%0), w\n\t" ::"r"(b_) : "memory");
                asm volatile("mpmm\n\t" ::);
            }
            asm volatile("mpse 0(%0)\n\t" ::"r"(c_) : "memory");
        }
    }
}

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

// void matmul_vec(int16_t *c, const int8_t *a, const int8_t *b,
//                 const unsigned long int K, const unsigned long int N)
// {
//     // Temporary variables
//     int8_t t0, t1, t2, t3, t4, t5, t6, t7;

//     // Original pointer
//     const int8_t *a_ = a;

//     // Prefetch one row of matrix B
//     asm volatile("vle8.v v18, (%0);" ::"r"(b));
//     b += N;

//     // Prefetch one row of scalar values
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
//     a += K;
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
//     a += K;
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
//     a += K;
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
//     a += K;
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
//     a += K;
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
//     a += K;
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
//     a += K;
//     asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));

//     // Compute the multiplication
//     unsigned long int k = 0;

//     while (k < K)
//     {
//         // Calculate pointer to the matrix A
//         a = (const int8_t *)a_ + ++k;

//         asm volatile("vle8.v v20, (%0);" ::"r"(b));
//         b += N;

//         asm volatile("vsext.vf2 v18, v18");
//         asm volatile("vmacc.vx v0, %0, v18" ::"r"(t0));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v2, %0, v18" ::"r"(t1));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v4, %0, v18" ::"r"(t2));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v6, %0, v18" ::"r"(t3));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v8, %0, v18" ::"r"(t4));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v10, %0, v18" ::"r"(t5));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v12, %0, v18" ::"r"(t6));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v14, %0, v18" ::"r"(t7));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));

//         // Load one row of B
//         asm volatile("vle8.v v18, (%0);" ::"r"(b));
//         b += N;

//         if (k == K - 1)
//             break;

//         a = (const int8_t *)a_ + ++k;
//         asm volatile("vsext.vf2 v20, v20");
//         asm volatile("vmacc.vx v0, %0, v20" ::"r"(t0));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v2, %0, v20" ::"r"(t1));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v6, %0, v20" ::"r"(t3));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v8, %0, v20" ::"r"(t4));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v10, %0, v20" ::"r"(t5));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v12, %0, v20" ::"r"(t6));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
//         a += K;
//         asm volatile("vmacc.vx v14, %0, v20" ::"r"(t7));
//         asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));
//     }

//     // Last iteration: store results
//     asm volatile("vsetivli zero, 0, e16, m2, ta, ma");
//     asm volatile("vsext.vf2 v20, v20");
//     asm volatile("vmacc.vx v0, %0, v20" ::"r"(t0));
//     asm volatile("vse16.v v0, (%0);" ::"r"(c));
//     c += N;
//     asm volatile("vmacc.vx v2, %0, v20" ::"r"(t1));
//     asm volatile("vse16.v v2, (%0);" ::"r"(c));
//     c += N;
//     asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
//     asm volatile("vse16.v v4, (%0);" ::"r"(c));
//     c += N;
//     asm volatile("vmacc.vx v6, %0, v20" ::"r"(t3));
//     asm volatile("vse16.v v6, (%0);" ::"r"(c));
//     c += N;
//     asm volatile("vmacc.vx v8, %0, v20" ::"r"(t4));
//     asm volatile("vse16.v v8, (%0);" ::"r"(c));
//     c += N;
//     asm volatile("vmacc.vx v10, %0, v20" ::"r"(t5));
//     asm volatile("vse16.v v10, (%0);" ::"r"(c));
//     c += N;
//     asm volatile("vmacc.vx v12, %0, v20" ::"r"(t6));
//     asm volatile("vse16.v v12, (%0);" ::"r"(c));
//     c += N;
//     asm volatile("vmacc.vx v14, %0, v20" ::"r"(t7));
//     asm volatile("vse16.v v14, (%0);" ::"r"(c));
// }

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

        asm volatile("vwmacc.vx v0, %0, v18" ::"r"(t0));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
        a += K;

        asm volatile("vwmacc.vx v2, %0, v18" ::"r"(t1));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v4, %0, v18" ::"r"(t2));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v6, %0, v18" ::"r"(t3));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v8, %0, v18" ::"r"(t4));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v10, %0, v18" ::"r"(t5));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v12, %0, v18" ::"r"(t6));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v14, %0, v18" ::"r"(t7));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));

        // Load one row of B
        asm volatile("vle8.v v18, (%0);" ::"r"(b));
        b += N;

        if (k == K - 1)
            break;

        a = (const int8_t *)a_ + ++k;

        asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t0));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t0) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t1));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t1) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t2) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t3));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t3) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v8, %0, v20" ::"r"(t4));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t4) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t5));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t5) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t6));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t6) : [a] "r"(a));
        a += K;
        asm volatile("vwmacc.vx v14, %0, v20" ::"r"(t7));
        asm volatile("lb %[t], (%[a])" : [t] "=r"(t7) : [a] "r"(a));
    }

    // Last iteration: store results
    asm volatile("vwmacc.vx v0, %0, v20" ::"r"(t0));
    asm volatile("vwmacc.vx v2, %0, v20" ::"r"(t1));
    asm volatile("vwmacc.vx v4, %0, v20" ::"r"(t2));
    asm volatile("vwmacc.vx v6, %0, v20" ::"r"(t3));
    asm volatile("vwmacc.vx v8, %0, v20" ::"r"(t4));
    asm volatile("vwmacc.vx v10, %0, v20" ::"r"(t5));
    asm volatile("vwmacc.vx v12, %0, v20" ::"r"(t6));
    asm volatile("vwmacc.vx v14, %0, v20" ::"r"(t7));
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
