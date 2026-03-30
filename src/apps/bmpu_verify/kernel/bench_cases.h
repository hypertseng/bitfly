#ifndef MYTEST_BENCH_CASES_H
#define MYTEST_BENCH_CASES_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 14

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"tiny", "bitfly-bmpu_verify", "binary_mt8_nt16_kt64_g2x2", 64UL, 64UL, 64UL, {8UL, 16UL, 64UL, 2UL, 2UL, 0UL}},
    {"tiny", "bitfly-bmpu_verify", "binary_mt16_nt16_kt64_g4x1", 128UL, 64UL, 64UL, {16UL, 16UL, 64UL, 4UL, 1UL, 0UL}},
    {"tiny", "bitfly-bmpu_verify", "binary_mt8_nt64_kt64_g1x2", 64UL, 128UL, 64UL, {8UL, 64UL, 64UL, 1UL, 2UL, 0UL}},
    {"tiny", "bitfly-bmpu_verify", "binary_mt32_nt32_kt32_g1x1", 64UL, 64UL, 32UL, {32UL, 32UL, 32UL, 1UL, 1UL, 0UL}},
    {"tiny", "bitfly-bmpu_verify", "binary_mt24_nt32_kt32_g1x1", 48UL, 64UL, 32UL, {24UL, 32UL, 32UL, 1UL, 1UL, 0UL}},
    {"tiny", "bitfly-bmpu_verify", "int2_mt8_nt32_kt64_g2x2", 64UL, 128UL, 64UL, {8UL, 32UL, 64UL, 2UL, 2UL, 2UL}},
    {"tiny", "bitfly-bmpu_verify", "int2_mt16_nt16_kt64_g1x4", 64UL, 64UL, 64UL, {16UL, 16UL, 64UL, 1UL, 4UL, 2UL}},
    {"tiny", "bitfly-bmpu_verify", "int4_mt8_nt16_kt32_g2x2", 64UL, 64UL, 32UL, {8UL, 16UL, 32UL, 2UL, 2UL, 3UL}},
    {"tiny", "bitfly-bmpu_verify", "binary_mt8_nt16_kt64_g8x1", 64UL, 64UL, 64UL, {8UL, 16UL, 64UL, 8UL, 1UL, 0UL}},
    {"tiny", "bitfly-bmpu_verify", "binary_mt8_nt16_kt64_g1x8", 68UL, 72UL, 64UL, {8UL, 16UL, 64UL, 1UL, 8UL, 0UL}},
    {"tiny", "bitfly-bmpu_verify", "int2_mt8_nt16_kt64_g1x8", 68UL, 72UL, 64UL, {8UL, 16UL, 64UL, 1UL, 8UL, 2UL}},
    {"tiny", "bitfly-bmpu_verify", "int2_mt8_nt16_kt64_g8x1", 64UL, 64UL, 64UL, {8UL, 16UL, 64UL, 8UL, 1UL, 2UL}},
    {"tiny", "bitfly-bmpu_verify", "int4_mt8_nt16_kt64_g8x1", 64UL, 64UL, 64UL, {8UL, 16UL, 64UL, 8UL, 1UL, 3UL}},
    {"tiny", "bitfly-bmpu_verify", "int4_mt8_nt16_kt64_g1x8", 68UL, 72UL, 64UL, {8UL, 16UL, 64UL, 1UL, 8UL, 3UL}},
};

#endif
