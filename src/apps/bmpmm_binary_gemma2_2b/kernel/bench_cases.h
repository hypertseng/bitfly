#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 7

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.q_proj", 128UL, 2048UL, 2304UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.k_proj", 128UL, 1024UL, 2304UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.v_proj", 128UL, 1024UL, 2304UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.o_proj", 128UL, 2304UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.mlp.gate_proj", 128UL, 9216UL, 2304UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.mlp.up_proj", 128UL, 9216UL, 2304UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.mlp.down_proj", 128UL, 2304UL, 9216UL, {8UL, 64UL, 128UL, 1UL, 2UL, 0UL}},
};

#endif
