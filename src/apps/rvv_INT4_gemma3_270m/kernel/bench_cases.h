#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 7

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.q_proj", 128UL, 1024UL, 640UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.k_proj", 128UL, 256UL, 640UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.v_proj", 128UL, 256UL, 640UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.o_proj", 128UL, 640UL, 1024UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.mlp.gate_proj", 128UL, 2048UL, 640UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.mlp.up_proj", 128UL, 2048UL, 640UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.mlp.down_proj", 128UL, 640UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
};

#endif
