#ifndef _BMPMM_H_
#define _BMPMM_H_

#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// 根据VLEN计算最大向量位数
#define MAX_VLEN_BITS VLEN
#define MAX_VLEN_BYTES (MAX_VLEN_BITS / 8)
#define MAX_INT8_VL (MAX_VLEN_BITS / 8)
#define MAX_INT32_VL (MAX_VLEN_BITS / 32)

// 算子1: 使用自定义指令加速的混合精度矩阵乘法
#define DECLARE_KERNEL(K) void mixed_precision_matmul_##K(int16_t *result, const int8_t *a, const int8_t *w);

DECLARE_KERNEL(16)
DECLARE_KERNEL(32)
DECLARE_KERNEL(64)
DECLARE_KERNEL(128)
DECLARE_KERNEL(256)
DECLARE_KERNEL(480)

typedef void (*mixed_kernel_func)(int16_t *, const int8_t *, const int8_t *);

typedef struct
{
    int8_t *activation_lp;
    int8_t *weight_lp;
    int16_t *result_lp;
    int8_t *activation_hp;
    int8_t *weight_hp;
    int32_t *result_hp;
    int32_t *result_torch;
} KernelData;

// 算子2: 使用向量扩展实现的 int8xint8 -> int32 的矩阵乘法
void vector_int8_matmul(int32_t *result, const int8_t *a, const int8_t *b, int M, int K, int N);

#endif // BMPMM_H
