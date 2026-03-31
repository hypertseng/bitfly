#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 7

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"stablelm", "stabilityai/stablelm-2-1_6b", "model.layers.0.self_attn.q_proj", 128UL, 2048UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"stablelm", "stabilityai/stablelm-2-1_6b", "model.layers.0.self_attn.k_proj", 128UL, 2048UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"stablelm", "stabilityai/stablelm-2-1_6b", "model.layers.0.self_attn.v_proj", 128UL, 2048UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"stablelm", "stabilityai/stablelm-2-1_6b", "model.layers.0.self_attn.o_proj", 128UL, 2048UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"stablelm", "stabilityai/stablelm-2-1_6b", "model.layers.0.mlp.gate_proj", 128UL, 5632UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"stablelm", "stabilityai/stablelm-2-1_6b", "model.layers.0.mlp.up_proj", 128UL, 5632UL, 2048UL, {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {"stablelm", "stabilityai/stablelm-2-1_6b", "model.layers.0.mlp.down_proj", 128UL, 2048UL, 5632UL, {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
};

#endif
