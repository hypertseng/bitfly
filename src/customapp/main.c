#include <stdint.h>
#include <string.h>

#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif


extern int8_t activation_int8[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_int1[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t weight_int8[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t result_int8[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int32_t result_int32[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t standard_int8[] __attribute__((aligned(32 * NR_LANES), section(".l2")));


void imatmul(int32_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
             unsigned long M, unsigned long K, unsigned long N)
{
    printf("M=%lu, K=%lu, N=%lu\n", M, K, N);

    for (unsigned long m = 0; m < M; ++m)
    {
        for (unsigned long n = 0; n < N; n += 64)
        {
            unsigned long vl = MIN(64, N - n);

            // 设置向量长度一次，复用 e8 + m4（提升并行度）
            asm volatile(
                "vsetvli t0, %[vl], e8, m4, ta, ma\n\t" // v0-v3: B向量
                "vmv.v.i v8, 0\n\t"                     // v8-v11: 累加器清零
                :
                : [vl] "r"(vl)
                : "t0", "v8", "v9", "v10", "v11");

            for (unsigned long k = 0; k < K; ++k)
            {
                int8_t a_val = a[m * K + k];
                const int8_t *b_ptr = &b[k * N + n];

                // 加载 B[k][n:n+vl] 到 v0-v3，vwmacc 宽乘累加到 v8-v11
                asm volatile(
                    "vle8.v v0, (%[b_ptr])\n\t"
                    "vsetvli t0, %[vl], e32, m4, ta, ma\n\t" // 设置宽乘结果宽度
                    "vwmacc.vx v8, %[a_val], v0\n\t"
                    :
                    : [b_ptr] "r"(b_ptr),
                      [a_val] "r"(a_val),
                      [vl] "r"(vl)
                    : "t0", "v0", "v8", "v9", "v10", "v11");
            }

            int32_t *c_ptr = &c[m * N + n];

            // 写回结果
            asm volatile(
                "vsetvli t0, %[vl], e32, m4, ta, ma\n\t"
                "vse32.v v8, (%[c_ptr])\n\t"
                :
                : [c_ptr] "r"(c_ptr),
                  [vl] "r"(vl)
                : "t0", "v8", "v9", "v10", "v11");
        }
    }
}

// void imatmul(int32_t *restrict c, const int8_t *restrict a, const int8_t *restrict b,
//              unsigned long M, unsigned long K, unsigned long N)
// {
//     printf("M=%lu, K=%lu, N=%lu\n", M, K, N);
//     unsigned long total_ops = M * K * N;
//     unsigned long completed_ops = 0;
//     unsigned long last_percent = 0;

//     for (unsigned long m = 0; m < M; ++m)
//     {
//         for (unsigned long n = 0; n < N; n += 64)
//         {
//             unsigned long vl = MIN(64, N - n);

//             asm volatile(
//                 "vsetvli t0, %[vl], e8, m4, ta, ma\n\t"
//                 "vmv.v.i v8, 0\n\t"
//                 :
//                 : [vl] "r"(vl)
//                 : "t0", "v8", "v9", "v10", "v11");

//             for (unsigned long k = 0; k < K; ++k)
//             {
//                 int8_t a_val = a[m * K + k];
//                 const int8_t *b_ptr = &b[k * N + n];

//                 asm volatile(
//                     "vle8.v v0, (%[b_ptr])\n\t"
//                     "vsetvli t0, %[vl], e32, m4, ta, ma\n\t"
//                     "vwmacc.vx v8, %[a_val], v0\n\t"
//                     :
//                     : [b_ptr] "r"(b_ptr),
//                       [a_val] "r"(a_val),
//                       [vl] "r"(vl)
//                     : "t0", "v0", "v8", "v9", "v10", "v11");

//                 // 更新进度
//                 completed_ops += vl;
//                 unsigned long percent = (completed_ops * 100) / total_ops;
//                 if (percent > last_percent)
//                 {
//                     printf("Progress: %lu%%\n", percent);
//                     last_percent = percent;
//                 }
//             }

//             int32_t *c_ptr = &c[m * N + n];
//             asm volatile(
//                 "vsetvli t0, %[vl], e32, m4, ta, ma\n\t"
//                 "vse32.v v8, (%[c_ptr])\n\t"
//                 :
//                 : [c_ptr] "r"(c_ptr),
//                   [vl] "r"(vl)
//                 : "t0", "v8", "v9", "v10", "v11");
//         }
//     }
//     printf("Completed 100%%\n");
// }

#define M 16
#define N 32
#define K 64

int main()
{
    uint32_t imm = 0x40;
    // size_t avl = 8;
    // size_t vl;
    // asm volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"(avl));
    // printf("vl=%d\n", vl);
    // vle
    size_t vl;
    size_t avl = 512; // 目标加载元素数
    
    // 设置 VL，SEW=8, LMUL=8
    asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"(avl));
    // 加载 activation 的前 64 个 int8 元素到向量寄存器 v0
    asm volatile("vle8.v v31, (%0)" ::"r"(activation_int8));
    int64_t runtime_m, runtime_v;
    start_timer();
    asm volatile("mpcfg  %0\n\t" ::"i"(imm));
    // asm volatile("mple 0(%0), a\n\t" ::"r"(activation_int8) : "memory");
    asm volatile("mple 0(%0), w\n\t" ::"r"(weight_int8) : "memory");
    // asm volatile("mpmm\n\t" ::);
    // asm volatile("mpse 0(%0)\n\t" ::"r"(result_int8) : "memory");
    stop_timer();
    runtime_m = get_timer();
    float performance = 2.0 * M * K * N / runtime_m;
    float utilization = 100 * performance / (16.0 * NR_LANES);

    printf("The execution took %d cycles.\n", runtime_m);
    // printf("The performance is %f OP/cycle (%f%% utilization).\n", performance,
    //        utilization);
    // // 比较resul 与 standard值是否相等
    // // 验证计算结果
    // int errors = 0;
    // for (int i = 0; i < M; ++i)
    // {
    //     if (result[i] != standard[i])
    //     {
    //         printf("Mismatch at [%d]: got %d, expected %d\n", i, result[i], standard[i]);
    //         errors++;
    //     }
    // }

    // if (errors == 0)
    // {
    //     printf("All results matched!\n");
    // }
    // else
    // {
    //     printf("Total mismatches: %d\n", errors);
    // }
    start_timer();
    imatmul(result_int32, activation_int8, weight_int8, M, K, N);
    stop_timer();
    runtime_v = get_timer();
    performance = 2.0 * M * K * N / runtime_v;
    utilization = 100 * performance / (16.0 * NR_LANES);

    printf("The execution took %d cycles.\n", runtime_v);
    printf("The performance is %f OP/cycle (%f%% utilization).\n", performance,
           utilization);

    // return errors;
}