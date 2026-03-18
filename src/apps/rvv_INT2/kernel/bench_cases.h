#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 34

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.q_proj", 128UL, 1024UL, 640UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.k_proj", 128UL, 256UL, 640UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.v_proj", 128UL, 256UL, 640UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.self_attn.o_proj", 128UL, 640UL, 1024UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.mlp.gate_proj", 128UL, 2048UL, 640UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.mlp.up_proj", 128UL, 2048UL, 640UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"tiny", "google/gemma-3-270m", "model.layers.0.mlp.down_proj", 128UL, 640UL, 2048UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"small", "Qwen/Qwen2.5-0.5B", "model.layers.0.self_attn.q_proj", 128UL, 896UL, 896UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"small", "Qwen/Qwen2.5-0.5B", "model.layers.0.self_attn.k_proj", 128UL, 128UL, 896UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"small", "Qwen/Qwen2.5-0.5B", "model.layers.0.self_attn.v_proj", 128UL, 128UL, 896UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"small", "Qwen/Qwen2.5-0.5B", "model.layers.0.self_attn.o_proj", 128UL, 896UL, 896UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"small", "Qwen/Qwen2.5-0.5B", "model.layers.0.mlp.gate_proj", 128UL, 4864UL, 896UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"small", "Qwen/Qwen2.5-0.5B", "model.layers.0.mlp.up_proj", 128UL, 4864UL, 896UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"small", "Qwen/Qwen2.5-0.5B", "model.layers.0.mlp.down_proj", 128UL, 896UL, 4864UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.q_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.k_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.v_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.self_attn.out_proj", 128UL, 2048UL, 2048UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.fc1", 128UL, 8192UL, 2048UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"medium", "facebook/opt-1.3b", "model.decoder.layers.0.fc2", 128UL, 2048UL, 8192UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.q_proj", 128UL, 1536UL, 1536UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.k_proj", 128UL, 256UL, 1536UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.v_proj", 128UL, 256UL, 1536UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.self_attn.o_proj", 128UL, 1536UL, 1536UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.mlp.gate_proj", 128UL, 8960UL, 1536UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.mlp.up_proj", 128UL, 8960UL, 1536UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"midlarge", "Qwen/Qwen2.5-1.5B", "model.layers.0.mlp.down_proj", 128UL, 1536UL, 8960UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.q_proj", 128UL, 2048UL, 2304UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.k_proj", 128UL, 1024UL, 2304UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.v_proj", 128UL, 1024UL, 2304UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.self_attn.o_proj", 128UL, 2304UL, 2048UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.mlp.gate_proj", 128UL, 9216UL, 2304UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.mlp.up_proj", 128UL, 9216UL, 2304UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
    {"large", "google/gemma-2-2b", "model.layers.0.mlp.down_proj", 128UL, 2304UL, 9216UL, {8UL, 64UL, 64UL, 2UL, 1UL, 2UL}},
};

#endif
