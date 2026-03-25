#include "data.h"
#include "data_decls.h"

#include <string.h>

typedef struct
{
    const char *layer;
    BenchKernelData data;
} BenchKernelDataMap;

static const BenchKernelDataMap kBenchKernelDataMap[] = {
    {
        .layer = "binary_mt8_nt16_kt64_g2x2",
        .data = {
            .activation_lp = activation_lp_case1,
            .weight_lp = weight_lp_case1,
            .result_lp = result_lp_case1,
            .activation_hp = activation_hp_case1,
            .weight_hp = weight_hp_case1,
            .result_hp = result_hp_case1,
            .result_torch = result_torch_case1,
        },
    },
    {
        .layer = "binary_mt16_nt16_kt64_g4x1",
        .data = {
            .activation_lp = activation_lp_case2,
            .weight_lp = weight_lp_case2,
            .result_lp = result_lp_case2,
            .activation_hp = activation_hp_case2,
            .weight_hp = weight_hp_case2,
            .result_hp = result_hp_case2,
            .result_torch = result_torch_case2,
        },
    },
    {
        .layer = "binary_mt8_nt64_kt64_g1x2",
        .data = {
            .activation_lp = activation_lp_case3,
            .weight_lp = weight_lp_case3,
            .result_lp = result_lp_case3,
            .activation_hp = activation_hp_case3,
            .weight_hp = weight_hp_case3,
            .result_hp = result_hp_case3,
            .result_torch = result_torch_case3,
        },
    },
    {
        .layer = "binary_mt32_nt32_kt32_g1x1",
        .data = {
            .activation_lp = activation_lp_case4,
            .weight_lp = weight_lp_case4,
            .result_lp = result_lp_case4,
            .activation_hp = activation_hp_case4,
            .weight_hp = weight_hp_case4,
            .result_hp = result_hp_case4,
            .result_torch = result_torch_case4,
        },
    },
    {
        .layer = "binary_mt24_nt32_kt32_g1x1",
        .data = {
            .activation_lp = activation_lp_case5,
            .weight_lp = weight_lp_case5,
            .result_lp = result_lp_case5,
            .activation_hp = activation_hp_case5,
            .weight_hp = weight_hp_case5,
            .result_hp = result_hp_case5,
            .result_torch = result_torch_case5,
        },
    },
    {
        .layer = "int2_mt8_nt32_kt64_g2x2",
        .data = {
            .activation_lp = activation_lp_case6,
            .weight_lp = weight_lp_case6,
            .result_lp = result_lp_case6,
            .activation_hp = activation_hp_case6,
            .weight_hp = weight_hp_case6,
            .result_hp = result_hp_case6,
            .result_torch = result_torch_case6,
        },
    },
    {
        .layer = "int2_mt16_nt16_kt64_g1x4",
        .data = {
            .activation_lp = activation_lp_case7,
            .weight_lp = weight_lp_case7,
            .result_lp = result_lp_case7,
            .activation_hp = activation_hp_case7,
            .weight_hp = weight_hp_case7,
            .result_hp = result_hp_case7,
            .result_torch = result_torch_case7,
        },
    },
    {
        .layer = "int4_mt8_nt16_kt32_g2x2",
        .data = {
            .activation_lp = activation_lp_case8,
            .weight_lp = weight_lp_case8,
            .result_lp = result_lp_case8,
            .activation_hp = activation_hp_case8,
            .weight_hp = weight_hp_case8,
            .result_hp = result_hp_case8,
            .result_torch = result_torch_case8,
        },
    },
};

BenchKernelData get_bench_kernel_data(int index)
{
    if (index < 0 || index >= (int)(sizeof(kBenchKernelDataMap) / sizeof(kBenchKernelDataMap[0])))
        return (BenchKernelData){0};
    return kBenchKernelDataMap[index].data;
}

BenchKernelData get_bench_kernel_data_by_layer(const char *layer)
{
    if (!layer)
        return (BenchKernelData){0};

    for (unsigned long i = 0; i < sizeof(kBenchKernelDataMap) / sizeof(kBenchKernelDataMap[0]); ++i)
    {
        if (strcmp(layer, kBenchKernelDataMap[i].layer) == 0)
            return kBenchKernelDataMap[i].data;
    }

    return (BenchKernelData){0};
}
