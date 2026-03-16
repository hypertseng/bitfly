#include "Vsa.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>

static void tick(Vsa *top)
{
    top->clk_i = 0;
    top->eval();
    top->clk_i = 1;
    top->eval();
}

static int expected_cycles(int prec, int kdim)
{
    int planes = 1;
    switch (prec)
    {
    case 0:
        planes = 1;
        break; // binary
    case 1:
        planes = 2;
        break; // ternary
    case 2:
        planes = 2;
        break; // int2
    case 3:
        planes = 4;
        break; // int4
    default:
        planes = 1;
        break;
    }
    return (4 - 1) + ((kdim / 8) * planes) + 4;
}

static bool out_row_is_zero(const VlWide<4> &row)
{
    return row[0] == 0 && row[1] == 0 && row[2] == 0 && row[3] == 0;
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    auto *top = new Vsa;

    top->rst_ni = 0;
    top->valid_i = 0;
    top->output_en_i = 0;
    top->k_dim_i = 0;
    top->prec_i = 0;

    for (int i = 0; i < 4; ++i)
        top->bmpu_act_operand_i[i] = 0;
    for (int j = 0; j < 4; ++j)
        top->bmpu_wgt_operand_i[j] = 0;

    for (int i = 0; i < 5; ++i)
        tick(top);
    top->rst_ni = 1;
    for (int i = 0; i < 2; ++i)
        tick(top);

    const int kdim = 64;
    for (int prec = 0; prec <= 3; ++prec)
    {
        const int exp = expected_cycles(prec, kdim);
        top->k_dim_i = kdim;
        top->prec_i = prec;
        top->valid_i = 1;
        top->output_en_i = 0;

        int cyc = 0;
        while (!top->sa_done_o && cyc < exp + 20)
        {
            tick(top);
            ++cyc;
        }

        if (!top->sa_done_o)
        {
            std::cerr << "[SA_TB] timeout prec=" << prec << " exp=" << exp << "\n";
            return 2;
        }
        if (cyc != exp)
        {
            std::cerr << "[SA_TB] cycle mismatch prec=" << prec << " got=" << cyc << " exp=" << exp << "\n";
            return 3;
        }

        top->output_en_i = 1;
        top->eval();
        for (int r = 0; r < 4; ++r)
        {
            if (!out_row_is_zero(top->output_data_o[r]))
            {
                std::cerr << "[SA_TB] non-zero output at prec=" << prec << " row=" << r << "\n";
                return 4;
            }
        }

        top->valid_i = 0;
        top->output_en_i = 0;
        tick(top);
        tick(top);

        std::cout << "[SA_TB] PASS prec=" << prec << " cycles=" << cyc << "\n";
    }

    std::cout << "[SA_TB] ALL TESTS PASSED\n";
    delete top;
    return 0;
}
