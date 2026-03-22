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


typedef struct
{
    int valid;
    unsigned long M;
    unsigned long N;
    unsigned long K;
    bmpmm_exec_cfg_t cfg;
    int64_t runtime;
    int64_t aux_cycles;
    int16_t sample[4];
    int first_case_index;
} bmpmm_runtime_cache_entry_t;

#define BMPMM_RUNTIME_CACHE_CAP 64

static inline int bmpmm_exec_cfg_equal(const bmpmm_exec_cfg_t *a, const bmpmm_exec_cfg_t *b)
{
    return a->mtile == b->mtile && a->ntile == b->ntile && a->ktile == b->ktile &&
           a->gm == b->gm && a->gn == b->gn && a->prec == b->prec;
}

static inline bmpmm_runtime_cache_entry_t *bmpmm_runtime_cache_lookup(
    bmpmm_runtime_cache_entry_t *entries, int count, const bmpmm_bench_case_t *sc, int match_cfg)
{
    for (int i = 0; i < count; ++i)
    {
        bmpmm_runtime_cache_entry_t *entry = &entries[i];
        if (!entry->valid)
            continue;
        if (entry->M != sc->M || entry->N != sc->N || entry->K != sc->K)
            continue;
        if (match_cfg && !bmpmm_exec_cfg_equal(&entry->cfg, &sc->cfg))
            continue;
        return entry;
    }
    return 0;
}

static inline void bmpmm_runtime_cache_store(
    bmpmm_runtime_cache_entry_t *entries, int *count, int capacity,
    const bmpmm_bench_case_t *sc, int case_index,
    int64_t runtime, int64_t aux_cycles, const int16_t *sample_src)
{
    if (*count >= capacity)
        return;

    bmpmm_runtime_cache_entry_t *entry = &entries[*count];
    entry->valid = 1;
    entry->M = sc->M;
    entry->N = sc->N;
    entry->K = sc->K;
    entry->cfg = sc->cfg;
    entry->runtime = runtime;
    entry->aux_cycles = aux_cycles;
    entry->first_case_index = case_index;
    for (int i = 0; i < 4; ++i)
        entry->sample[i] = sample_src ? sample_src[i] : 0;
    *count += 1;
}

#endif
