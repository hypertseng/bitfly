#ifndef MYTEST_DATA_H
#define MYTEST_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int8_t *activation_lp;
    int8_t *weight_lp;
    int16_t *result_lp;
    int8_t *activation_hp;
    int8_t *weight_hp;
    int16_t *result_hp;
    int16_t *result_torch;
} BenchKernelData;

#define DECLARE_KERNEL_DATA(TAG)             extern int8_t activation_lp_##TAG[];     extern int8_t weight_lp_##TAG[];         extern int16_t result_lp_##TAG[];        extern int8_t activation_hp_##TAG[];     extern int8_t weight_hp_##TAG[];         extern int16_t result_hp_##TAG[];        extern int16_t result_torch_##TAG[];

BenchKernelData get_bench_kernel_data(int index);
BenchKernelData get_bench_kernel_data_by_layer(const char *layer);

#ifdef __cplusplus
}
#endif

#endif
