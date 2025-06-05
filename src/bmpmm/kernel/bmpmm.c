#include "bmpmm.h"

// #define DEBUG
// 算子1: 混合精度矩阵乘法（使用自定义指令）
#define DEFINE_KERNEL(K_VAL)                                                              \
    void mixed_precision_matmul_##K_VAL(int16_t *result, const int8_t *a, const int8_t *w) \
    {                                                                                     \
        asm volatile("mpcfg " #K_VAL "\n\t");                                             \
        asm volatile("mple 0(%0), a\n\t" ::"r"(a) : "memory");                            \
        asm volatile("mple 0(%0), w\n\t" ::"r"(w) : "memory");                            \
        asm volatile("mpmm\n\t" ::);                                                      \
        asm volatile("mpse 0(%0)\n\t" ::"r"(result) : "memory");                          \
    }

// 实例化你需要的 K 值
DEFINE_KERNEL(16)
DEFINE_KERNEL(32)
DEFINE_KERNEL(64)
DEFINE_KERNEL(128)
DEFINE_KERNEL(256)
DEFINE_KERNEL(480)

mixed_kernel_func get_mixed_kernel(unsigned long K)
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

void vector_int8_matmul(int32_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                        unsigned long M, unsigned long K, unsigned long N)
{
    for (unsigned long i = 0; i < M; i++)
    {
        const int8_t *a_row = &a[i * K];

        for (unsigned long j = 0; j < N; j++)
        {
            const int8_t *b_col = &b[j * K];
            int32_t sum = 0;
            unsigned long k = 0;
            size_t vl;

            // 初始化向量长度和累加器
            asm volatile(
                "vsetvli %0, %1, e8, m1, ta, ma\n"
                "vmv.v.i v4, 0\n" // v4作为32位累加器
                : "=r"(vl) : "r"(K) : "v4");

            for (; k < K;)
            {
                // 动态调整vl
                asm volatile(
                    "vsetvli %0, %1, e8, m1, ta, ma\n"
                    : "=r"(vl) : "r"(K - k) :);

                // 向量乘加操作
                asm volatile(
                    "vle8.v v0, (%[a_ptr])\n" // 加载A行
                    "vle8.v v1, (%[b_ptr])\n" // 加载B列
                    "vwmacc.vv v4, v0, v1\n"  // 向量乘加
                    ::[a_ptr] "r"(&a_row[k]),
                    [b_ptr] "r"(&b_col[k])
                    : "v0", "v1", "v4");
                k += vl;
            }

            // 规约求和
            asm volatile(
                "vmv.v.i v5, 0\n"         // 初始化规约寄存器
                "vredsum.vs v5, v4, v5\n" // 向量求和
                "vmv.x.s %0, v5\n"        // 提取结果
                : "=r"(sum)::"v4", "v5");

            c[i * N + j] = sum;
        }
    }
}