#ifndef BMPMM_BENCH_COMMON_H
#define BMPMM_BENCH_COMMON_H

#include <stdint.h>

typedef struct
{
    unsigned long mtile;
    unsigned long ntile;
    unsigned long ktile;
    unsigned long gm;
    unsigned long gn;
    unsigned long prec;
} bmpmm_exec_cfg_t;

typedef struct
{
    unsigned long M;
    unsigned long N;
    unsigned long K;
    bmpmm_exec_cfg_t cfg;
} bmpmm_bench_case_t;

typedef struct
{
    int8_t *activation_lp;
    int8_t *weight_lp;
    int16_t *result_lp;
} bmpmm_bench_data_t;

#define BMPMM_DEFINE_BENCH_CASES_G(var_name, prec_value, ktile_value, gm_0_3, gn_0_3, gm_4, gn_4) \
    static const bmpmm_bench_case_t var_name[5] = { \
        {128UL, 128UL, 896UL,  {8UL, 64UL, (ktile_value), (gm_0_3), (gn_0_3), (prec_value)}}, \
        {128UL, 256UL, 640UL,  {8UL, 64UL, (ktile_value), (gm_0_3), (gn_0_3), (prec_value)}}, \
        {128UL, 256UL, 1536UL, {8UL, 64UL, (ktile_value), (gm_0_3), (gn_0_3), (prec_value)}}, \
        {128UL, 256UL, 2048UL, {8UL, 64UL, (ktile_value), (gm_0_3), (gn_0_3), (prec_value)}}, \
        {128UL, 320UL, 960UL,  {8UL, 64UL, (ktile_value), (gm_4), (gn_4), (prec_value)}}, \
    }

#define BMPMM_DEFINE_BENCH_CASES(var_name, prec_value, ktile_value) \
    BMPMM_DEFINE_BENCH_CASES_G(var_name, prec_value, ktile_value, 1UL, 2UL, 2UL, 1UL)

#endif
