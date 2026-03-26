#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 7

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.q_proj", 128UL, 1536UL, 1536UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.k_proj", 128UL, 256UL, 1536UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.v_proj", 128UL, 256UL, 1536UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.o_proj", 128UL, 1536UL, 1536UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.mlp.gate_proj", 128UL, 8960UL, 1536UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.mlp.up_proj", 128UL, 8960UL, 1536UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.mlp.down_proj", 128UL, 1536UL, 8960UL, {16UL, 32UL, 64UL, 1UL, 1UL, 3UL}},
};

#endif
