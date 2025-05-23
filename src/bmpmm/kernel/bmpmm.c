#include "bmpmm.h"

// 算子1: 混合精度矩阵乘法（使用自定义指令）
#define DEFINE_KERNEL(K_VAL)                                                              \
    void mixed_precision_matmul_##K_VAL(int8_t *result, const int8_t *a, const int8_t *w) \
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
DEFINE_KERNEL(496)

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
    case 496:
        return mixed_precision_matmul_496;
    }
}

// 算子2: 向量扩展实现的 int8xint8 -> int32 矩阵乘法
void vector_int8_matmul(int32_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
                        unsigned long M, unsigned long K, unsigned long N)
{
#ifdef INTRINSICS
    for (unsigned long i = 0; i < M; i++)
    {
        const int8_t *a_row = &a[i * K];

        for (unsigned long j = 0; j < N; j++)
        {
            const int8_t *b_col = &b[j];

            vint32m1_t acc = vmv_v_x_i32m1(0, vsetvlmax_e32m1());
            unsigned long k = 0;

            while (k < K)
            {
                size_t vl = vsetvl_e8m1(K - k);

                // Load a_row[k : k+vl] as int8
                vint8m1_t va = vle8_v_i8m1(&a_row[k], vl);

                // Load b_col[k : k+vl] with stride N (每次跳 N 个元素)
                vint8m1_t vb = vlse8_v_i8m1(&b_col[k * N], N, vl);

                // widen to int16
                vint16m2_t vaw = vwadd_vx_i16m2(va, 0, vl); // sign extend
                vint16m2_t vbw = vwadd_vx_i16m2(vb, 0, vl);

                // Multiply into int16
                vint16m2_t vmul = vmul_vv_i16m2(vaw, vbw, vl);

                // Widen and accumulate into int32
                acc = vwredsum_vs_i16m2_i32m1(vmul, acc, vl);

                k += vl;
            }

            // Extract scalar sum
            int32_t sum = vmv_x_s_i32m1_i32(acc);

            c[i * N + j] = sum;
        }
    }
#else
    for (unsigned long i = 0; i < M; i++)
    {
        const int8_t *a_row = &a[i * K];
        for (unsigned long j = 0; j < N; j++)
        {
            int32_t sum = 0;
            const int8_t *b_col = &b[j];
            unsigned long k = 0;

            if (K >= MAX_INT8_VL)
            {
                int32_t vec_sum = 0;
                asm volatile(
                    "mv t1, %[a_row]\n"
                    "mv t2, %[b_col]\n"
                    "mv t3, %[K]\n"
                    "li t4, 0\n"
                    "vsetvli zero, %[max_vl], e32, m1, ta, ma\n"
                    "vmv.v.i v0, 0\n"
                    "1:\n"
                    "vsetvli t0, t3, e8, m1, ta, ma\n"
                    "vle8.v v1, (t1)\n"
                    "add t1, t1, t0\n"
                    "lb t5, 0(t2)\n"
                    "vmv.v.x v2, t5\n"
                    "vwmul.vv v3, v1, v2\n"
                    "vsetvli zero, t0, e16, m2, ta, ma\n"
                    "vwredsum.vs v0, v3, v0\n"
                    "add t2, t2, %[N]\n"
                    "sub t3, t3, t0\n"
                    "bnez t3, 1b\n"
                    "vsetvli t0, zero, e32, m1, ta, ma\n"
                    "vmv.x.s %[vec_sum], v0\n"
                    : [vec_sum] "=r"(vec_sum)
                    : [a_row] "r"(a_row), [b_col] "r"(b_col),
                      [K] "r"(K), [N] "r"(N), [max_vl] "r"(MAX_INT8_VL)
                    : "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v1", "v2", "v3");

                sum = vec_sum;
                k = K - (K % MAX_INT8_VL);
            }

            for (; k < K; k++)
            {
                sum += a_row[k] * b_col[k * N];
            }

            c[i * N + j] = sum;
        }
    }
#endif
}