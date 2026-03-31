#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/model_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 7

static const model_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"smol360m", "HuggingFaceTB/SmolLM2-360M", "model.layers.0.self_attn.q_proj", 128UL, 960UL, 960UL},
    {"smol360m", "HuggingFaceTB/SmolLM2-360M", "model.layers.0.self_attn.k_proj", 128UL, 320UL, 960UL},
    {"smol360m", "HuggingFaceTB/SmolLM2-360M", "model.layers.0.self_attn.v_proj", 128UL, 320UL, 960UL},
    {"smol360m", "HuggingFaceTB/SmolLM2-360M", "model.layers.0.self_attn.o_proj", 128UL, 960UL, 960UL},
    {"smol360m", "HuggingFaceTB/SmolLM2-360M", "model.layers.0.mlp.gate_proj", 128UL, 2560UL, 960UL},
    {"smol360m", "HuggingFaceTB/SmolLM2-360M", "model.layers.0.mlp.up_proj", 128UL, 2560UL, 960UL},
    {"smol360m", "HuggingFaceTB/SmolLM2-360M", "model.layers.0.mlp.down_proj", 128UL, 960UL, 2560UL},
};

#endif
