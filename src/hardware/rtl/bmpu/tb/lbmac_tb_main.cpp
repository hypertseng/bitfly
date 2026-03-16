#include "Vlbmac.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>

static uint64_t pack_acts_1_to_8()
{
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i)
    {
        uint8_t a = static_cast<uint8_t>(i + 1); // act[i] = i+1
        bits |= (static_cast<uint64_t>(a) << ((7 - i) * 8));
    }
    return bits;
}

static int signed12(uint32_t v)
{
    v &= 0xFFFu;
    if (v & 0x800u)
        return static_cast<int>(v) - 4096;
    return static_cast<int>(v);
}

static int8_t get_act(int idx)
{
    return static_cast<int8_t>(idx + 1);
}

static int run_case(Vlbmac *top, uint8_t mode, uint8_t weights, int expect, const char *tag)
{
    top->mode_i = mode;
    top->weights = weights;
    top->eval();
    int got = signed12(top->result);
    if (got != expect)
    {
        std::cerr << "[LBMAC_TB] FAIL " << tag << " got=" << got << " exp=" << expect << "\n";
        return 1;
    }
    std::cout << "[LBMAC_TB] PASS " << tag << " got=" << got << "\n";
    return 0;
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    auto *top = new Vlbmac;

    top->activations = pack_acts_1_to_8();

    // binary all +1
    if (run_case(top, 0b00, 0xFF, 36, "binary_all_pos"))
        return 2;
    // binary all -1
    if (run_case(top, 0b00, 0x00, -36, "binary_all_neg"))
        return 3;

    // ternary: w = (+mask) + (-mask)
    // pos mask: idx0..3, neg mask: idx4..7 => 1+2+3+4-(5+6+7+8) = -16
    top->mode_i = 0b01;
    top->weights = 0x0F;
    top->eval();
    int ternary_pos = signed12(top->result);
    top->mode_i = 0b10;
    top->weights = 0xF0;
    top->eval();
    int ternary_neg = signed12(top->result);
    int ternary_total = ternary_pos + ternary_neg;
    if (ternary_total != -16)
    {
        std::cerr << "[LBMAC_TB] FAIL ternary total got=" << ternary_total << " exp=-16\n";
        return 4;
    }
    std::cout << "[LBMAC_TB] PASS ternary total got=" << ternary_total << "\n";

    // int2 sanity: low plane mask=0xFF, sign plane mask=0x01
    // weight per idx: idx0 => -1, idx1..7 => +1 => sum = -1*1 + 2+...+8 = 34
    top->mode_i = 0b01;
    top->weights = 0xFF;
    top->eval();
    int int2_low = signed12(top->result); // <<0
    top->mode_i = 0b10;
    top->weights = 0x01;
    top->eval();
    int int2_sign = signed12(top->result) << 1; // <<1
    int int2_total = int2_low + int2_sign;
    if (int2_total != 34)
    {
        std::cerr << "[LBMAC_TB] FAIL int2 total got=" << int2_total << " exp=34\n";
        return 5;
    }
    std::cout << "[LBMAC_TB] PASS int2 total got=" << int2_total << "\n";

    std::cout << "[LBMAC_TB] ALL TESTS PASSED\n";
    delete top;
    return 0;
}
