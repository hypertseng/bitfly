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
#ifdef HAS_KERNEL_DATA_CASE1
    case 0:
        return (BenchKernelData){
            .activation_lp = activation_lp_case1,
            .weight_lp = weight_lp_case1,
            .result_lp = result_lp_case1,
            .activation_hp = activation_hp_case1,
            .weight_hp = weight_hp_case1,
            .result_hp = result_hp_case1,
            .result_torch = result_torch_case1};
#endif
#ifdef HAS_KERNEL_DATA_CASE2
    case 1:
        return (BenchKernelData){
            .activation_lp = activation_lp_case2,
            .weight_lp = weight_lp_case2,
            .result_lp = result_lp_case2,
            .activation_hp = activation_hp_case2,
            .weight_hp = weight_hp_case2,
            .result_hp = result_hp_case2,
            .result_torch = result_torch_case2};
#endif
#ifdef HAS_KERNEL_DATA_CASE3
    case 2:
        return (BenchKernelData){
            .activation_lp = activation_lp_case3,
            .weight_lp = weight_lp_case3,
            .result_lp = result_lp_case3,
            .activation_hp = activation_hp_case3,
            .weight_hp = weight_hp_case3,
            .result_hp = result_hp_case3,
            .result_torch = result_torch_case3};
#endif
#ifdef HAS_KERNEL_DATA_CASE4
    case 3:
        return (BenchKernelData){
            .activation_lp = activation_lp_case4,
            .weight_lp = weight_lp_case4,
            .result_lp = result_lp_case4,
            .activation_hp = activation_hp_case4,
            .weight_hp = weight_hp_case4,
            .result_hp = result_hp_case4,
            .result_torch = result_torch_case4};
#endif
#ifdef HAS_KERNEL_DATA_CASE5
    case 4:
        return (BenchKernelData){
            .activation_lp = activation_lp_case5,
            .weight_lp = weight_lp_case5,
            .result_lp = result_lp_case5,
            .activation_hp = activation_hp_case5,
            .weight_hp = weight_hp_case5,
            .result_hp = result_hp_case5,
            .result_torch = result_torch_case5};
#endif
#ifdef HAS_KERNEL_DATA_CASE6
    case 5:
        return (BenchKernelData){
            .activation_lp = activation_lp_case6,
            .weight_lp = weight_lp_case6,
            .result_lp = result_lp_case6,
            .activation_hp = activation_hp_case6,
            .weight_hp = weight_hp_case6,
            .result_hp = result_hp_case6,
            .result_torch = result_torch_case6};
#endif
#ifdef HAS_KERNEL_DATA_CASE7
    case 6:
        return (BenchKernelData){
            .activation_lp = activation_lp_case7,
            .weight_lp = weight_lp_case7,
            .result_lp = result_lp_case7,
            .activation_hp = activation_hp_case7,
            .weight_hp = weight_hp_case7,
            .result_hp = result_hp_case7,
            .result_torch = result_torch_case7};
#endif
#ifdef HAS_KERNEL_DATA_CASE8
    case 7:
        return (BenchKernelData){
            .activation_lp = activation_lp_case8,
            .weight_lp = weight_lp_case8,
            .result_lp = result_lp_case8,
            .activation_hp = activation_hp_case8,
            .weight_hp = weight_hp_case8,
            .result_hp = result_hp_case8,
            .result_torch = result_torch_case8};
#endif
#ifdef HAS_KERNEL_DATA_CASE9
    case 8:
        return (BenchKernelData){
            .activation_lp = activation_lp_case9,
            .weight_lp = weight_lp_case9,
            .result_lp = result_lp_case9,
            .activation_hp = activation_hp_case9,
            .weight_hp = weight_hp_case9,
            .result_hp = result_hp_case9,
            .result_torch = result_torch_case9};
#endif
#ifdef HAS_KERNEL_DATA_CASE10
    case 9:
        return (BenchKernelData){
            .activation_lp = activation_lp_case10,
            .weight_lp = weight_lp_case10,
            .result_lp = result_lp_case10,
            .activation_hp = activation_hp_case10,
            .weight_hp = weight_hp_case10,
            .result_hp = result_hp_case10,
            .result_torch = result_torch_case10};
#endif
#ifdef HAS_KERNEL_DATA_CASE11
    case 10:
        return (BenchKernelData){
            .activation_lp = activation_lp_case11,
            .weight_lp = weight_lp_case11,
            .result_lp = result_lp_case11,
            .activation_hp = activation_hp_case11,
            .weight_hp = weight_hp_case11,
            .result_hp = result_hp_case11,
            .result_torch = result_torch_case11};
#endif
#ifdef HAS_KERNEL_DATA_CASE12
    case 11:
        return (BenchKernelData){
            .activation_lp = activation_lp_case12,
            .weight_lp = weight_lp_case12,
            .result_lp = result_lp_case12,
            .activation_hp = activation_hp_case12,
            .weight_hp = weight_hp_case12,
            .result_hp = result_hp_case12,
            .result_torch = result_torch_case12};
#endif
#ifdef HAS_KERNEL_DATA_CASE13
    case 12:
        return (BenchKernelData){
            .activation_lp = activation_lp_case13,
            .weight_lp = weight_lp_case13,
            .result_lp = result_lp_case13,
            .activation_hp = activation_hp_case13,
            .weight_hp = weight_hp_case13,
            .result_hp = result_hp_case13,
            .result_torch = result_torch_case13};
#endif
#ifdef HAS_KERNEL_DATA_CASE14
    case 13:
        return (BenchKernelData){
            .activation_lp = activation_lp_case14,
            .weight_lp = weight_lp_case14,
            .result_lp = result_lp_case14,
            .activation_hp = activation_hp_case14,
            .weight_hp = weight_hp_case14,
            .result_hp = result_hp_case14,
            .result_torch = result_torch_case14};
#endif
#ifdef HAS_KERNEL_DATA_CASE15
    case 14:
        return (BenchKernelData){
            .activation_lp = activation_lp_case15,
            .weight_lp = weight_lp_case15,
            .result_lp = result_lp_case15,
            .activation_hp = activation_hp_case15,
            .weight_hp = weight_hp_case15,
            .result_hp = result_hp_case15,
            .result_torch = result_torch_case15};
#endif
#ifdef HAS_KERNEL_DATA_CASE16
    case 15:
        return (BenchKernelData){
            .activation_lp = activation_lp_case16,
            .weight_lp = weight_lp_case16,
            .result_lp = result_lp_case16,
            .activation_hp = activation_hp_case16,
            .weight_hp = weight_hp_case16,
            .result_hp = result_hp_case16,
            .result_torch = result_torch_case16};
#endif
#ifdef HAS_KERNEL_DATA_CASE17
    case 16:
        return (BenchKernelData){
            .activation_lp = activation_lp_case17,
            .weight_lp = weight_lp_case17,
            .result_lp = result_lp_case17,
            .activation_hp = activation_hp_case17,
            .weight_hp = weight_hp_case17,
            .result_hp = result_hp_case17,
            .result_torch = result_torch_case17};
#endif
#ifdef HAS_KERNEL_DATA_CASE18
    case 17:
        return (BenchKernelData){
            .activation_lp = activation_lp_case18,
            .weight_lp = weight_lp_case18,
            .result_lp = result_lp_case18,
            .activation_hp = activation_hp_case18,
            .weight_hp = weight_hp_case18,
            .result_hp = result_hp_case18,
            .result_torch = result_torch_case18};
#endif
#ifdef HAS_KERNEL_DATA_CASE19
    case 18:
        return (BenchKernelData){
            .activation_lp = activation_lp_case19,
            .weight_lp = weight_lp_case19,
            .result_lp = result_lp_case19,
            .activation_hp = activation_hp_case19,
            .weight_hp = weight_hp_case19,
            .result_hp = result_hp_case19,
            .result_torch = result_torch_case19};
#endif
#ifdef HAS_KERNEL_DATA_CASE20
    case 19:
        return (BenchKernelData){
            .activation_lp = activation_lp_case20,
            .weight_lp = weight_lp_case20,
            .result_lp = result_lp_case20,
            .activation_hp = activation_hp_case20,
            .weight_hp = weight_hp_case20,
            .result_hp = result_hp_case20,
            .result_torch = result_torch_case20};
#endif
#ifdef HAS_KERNEL_DATA_CASE21
    case 20:
        return (BenchKernelData){
            .activation_lp = activation_lp_case21,
            .weight_lp = weight_lp_case21,
            .result_lp = result_lp_case21,
            .activation_hp = activation_hp_case21,
            .weight_hp = weight_hp_case21,
            .result_hp = result_hp_case21,
            .result_torch = result_torch_case21};
#endif
#ifdef HAS_KERNEL_DATA_CASE22
    case 21:
        return (BenchKernelData){
            .activation_lp = activation_lp_case22,
            .weight_lp = weight_lp_case22,
            .result_lp = result_lp_case22,
            .activation_hp = activation_hp_case22,
            .weight_hp = weight_hp_case22,
            .result_hp = result_hp_case22,
            .result_torch = result_torch_case22};
#endif
#ifdef HAS_KERNEL_DATA_CASE23
    case 22:
        return (BenchKernelData){
            .activation_lp = activation_lp_case23,
            .weight_lp = weight_lp_case23,
            .result_lp = result_lp_case23,
            .activation_hp = activation_hp_case23,
            .weight_hp = weight_hp_case23,
            .result_hp = result_hp_case23,
            .result_torch = result_torch_case23};
#endif
#ifdef HAS_KERNEL_DATA_CASE24
    case 23:
        return (BenchKernelData){
            .activation_lp = activation_lp_case24,
            .weight_lp = weight_lp_case24,
            .result_lp = result_lp_case24,
            .activation_hp = activation_hp_case24,
            .weight_hp = weight_hp_case24,
            .result_hp = result_hp_case24,
            .result_torch = result_torch_case24};
#endif
#ifdef HAS_KERNEL_DATA_CASE25
    case 24:
        return (BenchKernelData){
            .activation_lp = activation_lp_case25,
            .weight_lp = weight_lp_case25,
            .result_lp = result_lp_case25,
            .activation_hp = activation_hp_case25,
            .weight_hp = weight_hp_case25,
            .result_hp = result_hp_case25,
            .result_torch = result_torch_case25};
#endif
#ifdef HAS_KERNEL_DATA_CASE26
    case 25:
        return (BenchKernelData){
            .activation_lp = activation_lp_case26,
            .weight_lp = weight_lp_case26,
            .result_lp = result_lp_case26,
            .activation_hp = activation_hp_case26,
            .weight_hp = weight_hp_case26,
            .result_hp = result_hp_case26,
            .result_torch = result_torch_case26};
#endif
#ifdef HAS_KERNEL_DATA_CASE27
    case 26:
        return (BenchKernelData){
            .activation_lp = activation_lp_case27,
            .weight_lp = weight_lp_case27,
            .result_lp = result_lp_case27,
            .activation_hp = activation_hp_case27,
            .weight_hp = weight_hp_case27,
            .result_hp = result_hp_case27,
            .result_torch = result_torch_case27};
#endif
#ifdef HAS_KERNEL_DATA_CASE28
    case 27:
        return (BenchKernelData){
            .activation_lp = activation_lp_case28,
            .weight_lp = weight_lp_case28,
            .result_lp = result_lp_case28,
            .activation_hp = activation_hp_case28,
            .weight_hp = weight_hp_case28,
            .result_hp = result_hp_case28,
            .result_torch = result_torch_case28};
#endif
#ifdef HAS_KERNEL_DATA_CASE29
    case 28:
        return (BenchKernelData){
            .activation_lp = activation_lp_case29,
            .weight_lp = weight_lp_case29,
            .result_lp = result_lp_case29,
            .activation_hp = activation_hp_case29,
            .weight_hp = weight_hp_case29,
            .result_hp = result_hp_case29,
            .result_torch = result_torch_case29};
#endif
#ifdef HAS_KERNEL_DATA_CASE30
    case 29:
        return (BenchKernelData){
            .activation_lp = activation_lp_case30,
            .weight_lp = weight_lp_case30,
            .result_lp = result_lp_case30,
            .activation_hp = activation_hp_case30,
            .weight_hp = weight_hp_case30,
            .result_hp = result_hp_case30,
            .result_torch = result_torch_case30};
#endif
#ifdef HAS_KERNEL_DATA_CASE31
    case 30:
        return (BenchKernelData){
            .activation_lp = activation_lp_case31,
            .weight_lp = weight_lp_case31,
            .result_lp = result_lp_case31,
            .activation_hp = activation_hp_case31,
            .weight_hp = weight_hp_case31,
            .result_hp = result_hp_case31,
            .result_torch = result_torch_case31};
#endif
#ifdef HAS_KERNEL_DATA_CASE32
    case 31:
        return (BenchKernelData){
            .activation_lp = activation_lp_case32,
            .weight_lp = weight_lp_case32,
            .result_lp = result_lp_case32,
            .activation_hp = activation_hp_case32,
            .weight_hp = weight_hp_case32,
            .result_hp = result_hp_case32,
            .result_torch = result_torch_case32};
#endif
#ifdef HAS_KERNEL_DATA_CASE33
    case 32:
        return (BenchKernelData){
            .activation_lp = activation_lp_case33,
            .weight_lp = weight_lp_case33,
            .result_lp = result_lp_case33,
            .activation_hp = activation_hp_case33,
            .weight_hp = weight_hp_case33,
            .result_hp = result_hp_case33,
            .result_torch = result_torch_case33};
#endif
#ifdef HAS_KERNEL_DATA_CASE34
    case 33:
        return (BenchKernelData){
            .activation_lp = activation_lp_case34,
            .weight_lp = weight_lp_case34,
            .result_lp = result_lp_case34,
            .activation_hp = activation_hp_case34,
            .weight_hp = weight_hp_case34,
            .result_hp = result_hp_case34,
            .result_torch = result_torch_case34};
#endif
    default:
        return (BenchKernelData){0};
    }
}
