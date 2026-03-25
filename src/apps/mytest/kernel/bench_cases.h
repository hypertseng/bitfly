#ifndef MYTEST_BENCH_CASES_H
#define MYTEST_BENCH_CASES_H

#include "../../common/bmpmm_bench_common.h"

#define BMPMM_BENCH_CASE_COUNT 1

static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {
    {"tiny", "bitfly-mytest", "binary_mt16_nt16_kt64_g4x1", 64UL, 64UL, 64UL, {16UL, 16UL, 64UL, 4UL, 1UL, 0UL}},
};

#endif
