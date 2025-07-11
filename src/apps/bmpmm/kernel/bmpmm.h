#ifndef _BMPMM_H_
#define _BMPMM_H_

#include <stdint.h>
#include <string.h>
#include "runtime.h"

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

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
#define MAX_INT16_VL (MAX_VLEN_BITS / 16)

typedef struct
{
    int8_t *activation_lp;
    int8_t *weight_lp;
    int16_t *result_lp;
    int8_t *activation_hp;
    int8_t *weight_hp;
    int16_t *result_hp;
    int16_t *result_torch;
} KernelData;

void binary_mixed_matmul(int16_t *c, const int8_t *a, const int8_t *b,
                         const unsigned long int M, const unsigned long int K,
                         const unsigned long int N);

// 算子2: 使用向量扩展实现的 int8xint8 -> int16 的矩阵乘法
void vector_int8_matmul(int16_t *result, const int8_t *a, const int8_t *b, unsigned long int M, unsigned long int K, unsigned long int N);
void matmul_vec_slice_init();
void matmul_vec(int16_t *c, const int8_t *a, const int8_t *b,
                const unsigned long int K, const unsigned long int N);

void scalar_matmul(int16_t *result, const int8_t *a, const int8_t *b, unsigned long int M, unsigned long int K, unsigned long int N);
#endif // BMPMM_H
