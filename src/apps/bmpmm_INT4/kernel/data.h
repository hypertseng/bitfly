#ifndef BMPMM_COMMON_DATA_H
#define BMPMM_COMMON_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
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
    } KernelData;

#define DECLARE_KERNEL_DATA(K)             \
    extern int8_t activation_lp_##K[];  \
    extern int8_t weight_lp_##K[];      \
    extern int16_t result_lp_##K[];     \
    extern int8_t activation_hp_##K[];  \
    extern int8_t weight_hp_##K[];      \
    extern int16_t result_hp_##K[];     \
    extern int16_t result_torch_##K[];

#define DECLARE_KERNEL_DATA_SQUARE(s)         \
    extern int8_t activation_lp_square_##s[]; \
    extern int8_t weight_lp_square_##s[];     \
    extern int16_t result_lp_square_##s[];    \
    extern int8_t activation_hp_square_##s[]; \
    extern int8_t weight_hp_square_##s[];     \
    extern int16_t result_hp_square_##s[];    \
    extern int16_t result_torch_square_##s[];

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

    KernelData get_kernel_data(int x);
    BenchKernelData get_bench_kernel_data(int index);

#ifdef __cplusplus
}
#endif

#endif
