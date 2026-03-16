#include "data.h"

#include "data_decls.h"

KernelData
get_kernel_data(int x)
{
    switch (x)
    {
#ifdef HAS_KERNEL_DATA_SQUARE_16
    case 16:
        return (KernelData){
            .activation_lp = activation_lp_square_16,
            .weight_lp = weight_lp_square_16,
            .result_lp = result_lp_square_16,
            .activation_hp = activation_hp_square_16,
            .weight_hp = weight_hp_square_16,
            .result_hp = result_hp_square_16,
            .result_torch = result_torch_square_16};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_32
    case 32:
        return (KernelData){
            .activation_lp = activation_lp_square_32,
            .weight_lp = weight_lp_square_32,
            .result_lp = result_lp_square_32,
            .activation_hp = activation_hp_square_32,
            .weight_hp = weight_hp_square_32,
            .result_hp = result_hp_square_32,
            .result_torch = result_torch_square_32};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_64
    case 64:
        return (KernelData){
            .activation_lp = activation_lp_square_64,
            .weight_lp = weight_lp_square_64,
            .result_lp = result_lp_square_64,
            .activation_hp = activation_hp_square_64,
            .weight_hp = weight_hp_square_64,
            .result_hp = result_hp_square_64,
            .result_torch = result_torch_square_64};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_128
    case 128:
        return (KernelData){
            .activation_lp = activation_lp_square_128,
            .weight_lp = weight_lp_square_128,
            .result_lp = result_lp_square_128,
            .activation_hp = activation_hp_square_128,
            .weight_hp = weight_hp_square_128,
            .result_hp = result_hp_square_128,
            .result_torch = result_torch_square_128};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_256
    case 256:
        return (KernelData){
            .activation_lp = activation_lp_square_256,
            .weight_lp = weight_lp_square_256,
            .result_lp = result_lp_square_256,
            .activation_hp = activation_hp_square_256,
            .weight_hp = weight_hp_square_256,
            .result_hp = result_hp_square_256,
            .result_torch = result_torch_square_256};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_512
    case 512:
        return (KernelData){
            .activation_lp = activation_lp_square_512,
            .weight_lp = weight_lp_square_512,
            .result_lp = result_lp_square_512,
            .activation_hp = activation_hp_square_512,
            .weight_hp = weight_hp_square_512,
            .result_hp = result_hp_square_512,
            .result_torch = result_torch_square_512};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_1024
    case 1024:
        return (KernelData){
            .activation_lp = activation_lp_square_1024,
            .weight_lp = weight_lp_square_1024,
            .result_lp = result_lp_square_1024,
            .activation_hp = activation_hp_square_1024,
            .weight_hp = weight_hp_square_1024,
            .result_hp = result_hp_square_1024,
            .result_torch = result_torch_square_1024};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_2048
    case 2048:
        return (KernelData){
            .activation_lp = activation_lp_square_2048,
            .weight_lp = weight_lp_square_2048,
            .result_lp = result_lp_square_2048,
            .activation_hp = activation_hp_square_2048,
            .weight_hp = weight_hp_square_2048,
            .result_hp = result_hp_square_2048,
            .result_torch = result_torch_square_2048};
#endif
#ifdef HAS_KERNEL_DATA_SQUARE_4096
    case 4096:
        return (KernelData){
            .activation_lp = activation_lp_square_4096,
            .weight_lp = weight_lp_square_4096,
            .result_lp = result_lp_square_4096,
            .activation_hp = activation_hp_square_4096,
            .weight_hp = weight_hp_square_4096,
            .result_hp = result_hp_square_4096,
            .result_torch = result_torch_square_4096};
#endif
    default:
        return (KernelData){0};
    }
}

BenchKernelData
get_bench_kernel_data(int index)
{
    switch (index)
    {
#ifdef HAS_KERNEL_DATA_TOP1
    case 0:
        return (BenchKernelData){
            .activation_lp = activation_lp_top1,
            .weight_lp = weight_lp_top1,
            .result_lp = result_lp_top1};
#endif
#ifdef HAS_KERNEL_DATA_TOP2
    case 1:
        return (BenchKernelData){
            .activation_lp = activation_lp_top2,
            .weight_lp = weight_lp_top2,
            .result_lp = result_lp_top2};
#endif
#ifdef HAS_KERNEL_DATA_TOP3
    case 2:
        return (BenchKernelData){
            .activation_lp = activation_lp_top3,
            .weight_lp = weight_lp_top3,
            .result_lp = result_lp_top3};
#endif
#ifdef HAS_KERNEL_DATA_TOP4
    case 3:
        return (BenchKernelData){
            .activation_lp = activation_lp_top4,
            .weight_lp = weight_lp_top4,
            .result_lp = result_lp_top4};
#endif
#ifdef HAS_KERNEL_DATA_TOP5
    case 4:
        return (BenchKernelData){
            .activation_lp = activation_lp_top5,
            .weight_lp = weight_lp_top5,
            .result_lp = result_lp_top5};
#endif
    default:
        return (BenchKernelData){0};
    }
}
