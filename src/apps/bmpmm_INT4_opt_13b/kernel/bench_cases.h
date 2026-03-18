#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 6

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.q_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 32UL, 1UL, 1UL, 3UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.k_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 32UL, 1UL, 1UL, 3UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.v_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 32UL, 1UL, 1UL, 3UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.out_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 32UL, 1UL, 1UL, 3UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.fc1", 128UL, 8192UL, 2048UL, {8UL, 64UL, 32UL, 1UL, 1UL, 3UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.fc2", 128UL, 2048UL, 8192UL, {8UL, 64UL, 32UL, 1UL, 1UL, 3UL}},
};

#endif
