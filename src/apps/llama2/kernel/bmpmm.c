#include "bmpmm.h"

// #define DEBUG
// 算子1: 混合精度矩阵乘法（使用自定义指令）
#define DEFINE_KERNEL(K_VAL)                                                               \
    void mixed_precision_matmul_##K_VAL(int16_t *result, const int8_t *a, const int8_t *w) \
    {                                                                                      \
        asm volatile("bmpcfg " #K_VAL "\n\t");                                              \
        asm volatile("bmple 0(%0), a\n\t" ::"r"(a) : "memory");                             \
        asm volatile("bmple 0(%0), w\n\t" ::"r"(w) : "memory");                             \
        asm volatile("bmpmm\n\t" ::);                                                       \
        asm volatile("bmpse 0(%0)\n\t" ::"r"(result) : "memory");                           \
    }

// // 实例化你需要的 K 值
// DEFINE_KERNEL(16)
// DEFINE_KERNEL(32)
// DEFINE_KERNEL(64)
// DEFINE_KERNEL(128)
// DEFINE_KERNEL(256)
// DEFINE_KERNEL(480)

// mixed_kernel_func get_mixed_kernel(int K)
// {
//     switch (K)
//     {
//     case 16:
//         return mixed_precision_matmul_16;
//     case 32:
//         return mixed_precision_matmul_32;
//     case 64:
//         return mixed_precision_matmul_64;
//     case 128:
//         return mixed_precision_matmul_128;
//     case 256:
//         return mixed_precision_matmul_256;
//     case 480:
//         return mixed_precision_matmul_480;
//     }
// }

void static inline get_config(unsigned long int K)
{
    switch (K)
    {
    case 8:
        asm volatile("bmpcfg 0, 8, 16, 32, 1, 1\n\t");
        break;
    case 16:
        asm volatile("bmpcfg 0, 16, 16, 32, 1, 1\n\t");
        break;
    case 24:
        asm volatile("bmpcfg 0, 24, 16, 32, 1, 1\n\t");
        break;
    case 32:
        asm volatile("bmpcfg 0, 32, 16, 32, 1, 1\n\t");
        break;
    case 40:
        asm volatile("bmpcfg 0, 40, 16, 32, 1, 1\n\t");
    case 48:
        asm volatile("bmpcfg 0, 48, 16, 32, 1, 1\n\t");
        break;
    case 56:
        asm volatile("bmpcfg 0, 56, 16, 32, 1, 1\n\t");
        break;
    case 64:
        asm volatile("bmpcfg 0, 64, 16, 32, 1, 1\n\t");
        break;
    case 72:
        asm volatile("bmpcfg 0, 72, 16, 32, 1, 1\n\t");
        break;
    case 80:
        asm volatile("bmpcfg 0, 80, 16, 32, 1, 1\n\t");
        break;
    case 88:
        asm volatile("bmpcfg 0, 88, 16, 32, 1, 1\n\t");
        break;
    case 96:
        asm volatile("bmpcfg 0, 96, 16, 32, 1, 1\n\t");
        break;
    case 104:
        asm volatile("bmpcfg 0, 104, 16, 32, 1, 1\n\t");
        break;
    case 112:
        asm volatile("bmpcfg 0, 112, 16, 32, 1, 1\n\t");
        break;
    case 120:
        asm volatile("bmpcfg 0, 120, 16, 32, 1, 1\n\t");
        break;
    case 128:
        asm volatile("bmpcfg 0, 128, 16, 32, 1, 1\n\t");
        break;
    case 136:
        asm volatile("bmpcfg 0, 136, 16, 32, 1, 1\n\t");
        break;
    case 144:
        asm volatile("bmpcfg 0, 144, 16, 32, 1, 1\n\t");
        break;
    case 152:
        asm volatile("bmpcfg 0, 152, 16, 32, 1, 1\n\t");
        break;
    case 160:
        asm volatile("bmpcfg 0, 160, 16, 32, 1, 1\n\t");
        break;
    case 168:
        asm volatile("bmpcfg 0, 168, 16, 32, 1, 1\n\t");
        break;
    case 176:
        asm volatile("bmpcfg 0, 176, 16, 32, 1, 1\n\t");
        break;
    case 184:
        asm volatile("bmpcfg 0, 184, 16, 32, 1, 1\n\t");
        break;
    case 192:
        asm volatile("bmpcfg 0, 192, 16, 32, 1, 1\n\t");
        break;
    case 200:
        asm volatile("bmpcfg 0, 200, 16, 32, 1, 1\n\t");
        break;
    case 208:
        asm volatile("bmpcfg 0, 208, 16, 32, 1, 1\n\t");
        break;
    case 216:
        asm volatile("bmpcfg 0, 216, 16, 32, 1, 1\n\t");
        break;
    case 224:
        asm volatile("bmpcfg 0, 224, 16, 32, 1, 1\n\t");
        break;
    case 232:
        asm volatile("bmpcfg 0, 232, 16, 32, 1, 1\n\t");
        break;
    case 240:
        asm volatile("bmpcfg 0, 240, 16, 32, 1, 1\n\t");
        break;
    case 248:
        asm volatile("bmpcfg 0, 248, 16, 32, 1, 1\n\t");
        break;
    case 256:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 264:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 272:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 280:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 288:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 296:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 304:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 312:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 320:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 328:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 336:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 344:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 352:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 360:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 368:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 376:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 384:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 392:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 400:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 408:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 416:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 424:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 432:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 440:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 448:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 456:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 464:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 472:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    case 480:
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
        break;
    }
}

void binary_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                         const unsigned long int M, const unsigned long int K,
                         const unsigned long int N)
{
    int tm = (M + 15) / 16;
    int tn = (N + 31) / 32;
    int tk = (K + 255) / 256;
    if (tk > 1)
    {
        asm volatile("bmpcfg 0, 256, 16, 32, 1, 1\n\t");
    }
    for (int i = 0; i < tm; i++)
    {
        for (int j = 0; j < tn; j++)
        {
            int16_t *c_ = c + i * N * 16 + j * 32 * 16;
            // int64_t runtime = get_timer();
            // printf("bmm1 timer cycles: %ld\n", runtime);
            for (int k = 0; k < tk; k++)
            {
                if (k == tk - 1){
                    if (((K % 256) % 8)){
                        get_config((8 - (K % 256) % 8) + (K % 256));
                    }else{
                        get_config(K);
                    }
                }
                const int8_t *a_ = a + i * 16 * K + k * 16 * 256;
                const int8_t *b_ = b + j * 32 * K / 8 + k * 32 * 256 / 8;

                asm volatile("bmple 0(%0), a\n\t" ::"r"(a_) : "memory");
                // printf("1\n");
                asm volatile("bmple 0(%0), w\n\t" ::"r"(b_) : "memory");
                // printf("2\n");
                asm volatile("bmpmm\n\t" ::);
                // printf("3\n");
            }
            // runtime = get_timer();
            // printf("bmm2 timer cycles: %ld\n", runtime);
            // printf("4\n");
            asm volatile("bmpse 0(%0)\n\t" ::"r"(c_) : "memory");
            // runtime = get_timer();
            // printf("bmm3 timer cycles: %ld\n", runtime);
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
