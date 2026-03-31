#ifndef BMPMM_BENCH_CASES_GENERATED_H
#define BMPMM_BENCH_CASES_GENERATED_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 4

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"code3b", "replit/replit-code-v1-3b", "transformer.blocks.0.attn.Wqkv", 128UL, 7680UL, 2560UL, {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
    {"code3b", "replit/replit-code-v1-3b", "transformer.blocks.0.attn.out_proj", 128UL, 2560UL, 2560UL, {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
    {"code3b", "replit/replit-code-v1-3b", "transformer.blocks.0.ffn.up_proj", 128UL, 10240UL, 2560UL, {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
    {"code3b", "replit/replit-code-v1-3b", "transformer.blocks.0.ffn.down_proj", 128UL, 2560UL, 10240UL, {8UL, 64UL, 128UL, 1UL, 2UL, 0UL}},
};

#endif
