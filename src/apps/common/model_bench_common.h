#ifndef MODEL_BENCH_COMMON_H
#define MODEL_BENCH_COMMON_H

#include <stdint.h>

typedef struct
{
    const char *scale;
    const char *model;
    const char *layer;
    unsigned long M;
    unsigned long N;
    unsigned long K;
} model_bench_case_t;

typedef struct
{
    int valid;
    unsigned long M;
    unsigned long N;
    unsigned long K;
    int64_t runtime;
    int64_t aux_cycles;
    int16_t sample[4];
    int first_case_index;
} model_runtime_cache_entry_t;

#define MODEL_RUNTIME_CACHE_CAP 64

static inline model_runtime_cache_entry_t *model_runtime_cache_lookup(
    model_runtime_cache_entry_t *entries, int count, const model_bench_case_t *sc)
{
    for (int i = 0; i < count; ++i)
    {
        model_runtime_cache_entry_t *entry = &entries[i];
        if (!entry->valid)
            continue;
        if (entry->M != sc->M || entry->N != sc->N || entry->K != sc->K)
            continue;
        return entry;
    }
    return 0;
}

static inline void model_runtime_cache_store(
    model_runtime_cache_entry_t *entries, int *count, int capacity,
    const model_bench_case_t *sc, int case_index,
    int64_t runtime, int64_t aux_cycles, const int16_t *sample_src)
{
    if (*count >= capacity)
        return;

    model_runtime_cache_entry_t *entry = &entries[*count];
    entry->valid = 1;
    entry->M = sc->M;
    entry->N = sc->N;
    entry->K = sc->K;
    entry->runtime = runtime;
    entry->aux_cycles = aux_cycles;
    entry->first_case_index = case_index;
    for (int i = 0; i < 4; ++i)
        entry->sample[i] = sample_src ? sample_src[i] : 0;
    *count += 1;
}

#endif
