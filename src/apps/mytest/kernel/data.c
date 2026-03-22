#include "data.h"
#include "data_decls.h"

BenchKernelData get_bench_kernel_data(int index)
{
    switch (index)
    {
#ifdef HAS_KERNEL_DATA_CASE_BINARY
    case 0:
        return (BenchKernelData){
            .activation_lp = activation_lp_case_binary,
            .weight_lp = weight_lp_case_binary,
            .result_lp = result_lp_case_binary,
            .activation_hp = activation_hp_case_binary,
            .weight_hp = weight_hp_case_binary,
            .result_hp = result_hp_case_binary,
            .result_torch = result_torch_case_binary,
        };
#endif
#ifdef HAS_KERNEL_DATA_CASE_INT2
    case 1:
        return (BenchKernelData){
            .activation_lp = activation_lp_case_int2,
            .weight_lp = weight_lp_case_int2,
            .result_lp = result_lp_case_int2,
            .activation_hp = activation_hp_case_int2,
            .weight_hp = weight_hp_case_int2,
            .result_hp = result_hp_case_int2,
            .result_torch = result_torch_case_int2,
        };
#endif
#ifdef HAS_KERNEL_DATA_CASE_INT4
    case 2:
        return (BenchKernelData){
            .activation_lp = activation_lp_case_int4,
            .weight_lp = weight_lp_case_int4,
            .result_lp = result_lp_case_int4,
            .activation_hp = activation_hp_case_int4,
            .weight_hp = weight_hp_case_int4,
            .result_hp = result_hp_case_int4,
            .result_torch = result_torch_case_int4,
        };
#endif
    default:
        return (BenchKernelData){0};
    }
}
