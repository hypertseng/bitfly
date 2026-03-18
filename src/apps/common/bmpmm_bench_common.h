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
    const char *scale;
    const char *model;
    const char *layer;
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

#endif
