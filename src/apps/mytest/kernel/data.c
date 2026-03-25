#include "data.h"
#include "data_decls.h"

BenchKernelData get_bench_kernel_data(int index)
{
    switch (index)
    {
    case 0:
        return (BenchKernelData){
            .activation_lp = activation_lp_case1,
            .weight_lp = weight_lp_case1,
            .result_lp = result_lp_case1,
            .activation_hp = activation_hp_case1,
            .weight_hp = weight_hp_case1,
            .result_hp = result_hp_case1,
            .result_torch = result_torch_case1,
        };
    case 1:
        return (BenchKernelData){
            .activation_lp = activation_lp_case2,
            .weight_lp = weight_lp_case2,
            .result_lp = result_lp_case2,
            .activation_hp = activation_hp_case2,
            .weight_hp = weight_hp_case2,
            .result_hp = result_hp_case2,
            .result_torch = result_torch_case2,
        };
    case 2:
        return (BenchKernelData){
            .activation_lp = activation_lp_case3,
            .weight_lp = weight_lp_case3,
            .result_lp = result_lp_case3,
            .activation_hp = activation_hp_case3,
            .weight_hp = weight_hp_case3,
            .result_hp = result_hp_case3,
            .result_torch = result_torch_case3,
        };
    case 3:
        return (BenchKernelData){
            .activation_lp = activation_lp_case4,
            .weight_lp = weight_lp_case4,
            .result_lp = result_lp_case4,
            .activation_hp = activation_hp_case4,
            .weight_hp = weight_hp_case4,
            .result_hp = result_hp_case4,
            .result_torch = result_torch_case4,
        };
    case 4:
        return (BenchKernelData){
            .activation_lp = activation_lp_case5,
            .weight_lp = weight_lp_case5,
            .result_lp = result_lp_case5,
            .activation_hp = activation_hp_case5,
            .weight_hp = weight_hp_case5,
            .result_hp = result_hp_case5,
            .result_torch = result_torch_case5,
        };
    case 5:
        return (BenchKernelData){
            .activation_lp = activation_lp_case6,
            .weight_lp = weight_lp_case6,
            .result_lp = result_lp_case6,
            .activation_hp = activation_hp_case6,
            .weight_hp = weight_hp_case6,
            .result_hp = result_hp_case6,
            .result_torch = result_torch_case6,
        };
    case 6:
        return (BenchKernelData){
            .activation_lp = activation_lp_case7,
            .weight_lp = weight_lp_case7,
            .result_lp = result_lp_case7,
            .activation_hp = activation_hp_case7,
            .weight_hp = weight_hp_case7,
            .result_hp = result_hp_case7,
            .result_torch = result_torch_case7,
        };
    case 7:
        return (BenchKernelData){
            .activation_lp = activation_lp_case8,
            .weight_lp = weight_lp_case8,
            .result_lp = result_lp_case8,
            .activation_hp = activation_hp_case8,
            .weight_hp = weight_hp_case8,
            .result_hp = result_hp_case8,
            .result_torch = result_torch_case8,
        };
    default:
        return (BenchKernelData){0};
    }
}
