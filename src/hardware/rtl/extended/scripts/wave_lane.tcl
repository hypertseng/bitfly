# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>

add wave -noupdate -group Ara -group Lane[$1] -group sequencer /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_lane_sequencer/*

add wave -noupdate -group Ara -group Lane[$1] -group operand_requester /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_requester/*
for {set requester 0}  {$requester < [examine -radix dec ara_pkg::NrOperandQueues]} {incr requester} {
    add wave -noupdate -group Ara -group Lane[$1] -group operand_requester -group requester[$requester] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_requester/gen_operand_requester[$requester]/*
}
for {set bank 0}  {$bank < [examine -radix dec ara_pkg::NrVRFBanksPerLane]} {incr bank} {
    add wave -noupdate -group Ara -group Lane[$1] -group operand_requester -group arbiters /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_requester/gen_vrf_arbiters[$bank]/*
}

add wave -noupdate -group Ara -group Lane[$1] -group vector_regfile /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vrf/*
for {set bank 0}  {$bank < [examine -radix dec ara_pkg::NrVRFBanksPerLane]} {incr bank} {
    add wave -noupdate -group Ara -group Lane[$1] -group vector_regfile -group vrf_bank[$bank] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vrf/gen_banks[$bank]/data_sram/*
    add wave -noupdate -group Ara -group Lane[$1] -group vector_regfile -group vrf_bank[$bank] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vrf/gen_banks[$bank]/data_sram/sram
}

add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group alu_a /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_alu_a/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group alu_b /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_alu_b/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group mfpu_a /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_mfpu_a/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group mfpu_b /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_mfpu_b/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group mfpu_c /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_mfpu_c/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group mfpu_c /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_mfpu_c/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group st_mask_a /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_st_mask_a/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group slide_addrgen_a /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_slide_addrgen_a/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group mask_b /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_mask_b/*
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group mask_m /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/i_operand_queue_mask_m/*
for {set queue 0}  {$queue < 4} {incr queue} {
    add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group bmpu_act -group act[$queue] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/gen_bmpu_act_queues[$queue]/i_bmpu_act_queue/*
    add wave -noupdate -group Ara -group Lane[$1] -group operand_queues -group bmpu_wgt -group wgt[$queue] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/gen_bmpu_wgt_queues[$queue]/i_bmpu_wgt_queue/*
}
add wave -noupdate -group Ara -group Lane[$1] -group operand_queues /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_operand_queues/*

add wave -noupdate -group Ara -group Lane[$1] -group valu -group simd_alu /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_valu/i_simd_alu/*
add wave -noupdate -group Ara -group Lane[$1] -group valu /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_valu/*

add wave -noupdate -group Ara -group Lane[$1] -group vmfpu -group simd_vmul_ew64 /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/i_simd_mul_ew64/*
add wave -noupdate -group Ara -group Lane[$1] -group vmfpu -group simd_vmul_ew32 /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/i_simd_mul_ew32/*
add wave -noupdate -group Ara -group Lane[$1] -group vmfpu -group simd_vmul_ew16 /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/i_simd_mul_ew16/*
add wave -noupdate -group Ara -group Lane[$1] -group vmfpu -group simd_vmul_ew8 /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/i_simd_mul_ew8/*
add wave -noupdate -group Ara -group Lane[$1] -group vmfpu -group simd_vdiv /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/i_simd_div/*
add wave -noupdate -group Ara -group Lane[$1] -group vmfpu -group simd_vdiv -group serdiv /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/i_simd_div/i_serdiv/*
add wave -noupdate -group Ara -group Lane[$1] -group vmfpu -group fpnew /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/fpu_gen/i_fpnew_bulk/*
add wave -noupdate -group Ara -group Lane[$1] -group vmfpu /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_vfus/i_vmfpu/*

add wave -noupdate -group Ara -group Lane[$1] -group bmpu /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_bmpu/*
add wave -noupdate -group Ara -group Lane[$1] -group bmpu -group sa /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_bmpu/u_sa/*
for {set row 0}  {$row < 4} {incr row} {
    for {set col 0}  {$col < 4} {incr col} {
        add wave -noupdate -group Ara -group Lane[$1] -group bmpu -group sa -group pe[$row][$col] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_bmpu/u_sa/row_gen[$row]/col_gen[$col]/u_lbmac/*
        add wave -noupdate -group Ara -group Lane[$1] -group bmpu -group sa -group pe[$row][$col] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_bmpu/u_sa/row_gen[$row]/col_gen[$col]/u_lbmac/partial_sum_reg
    }
}
for {set row 0}  {$row < 4} {incr row} {
    for {set col 0}  {$col < 4} {incr col} {
        for {set id 0}  {$id < 8} {incr id} {
                    add wave -noupdate -group Ara -group Lane[$1] -group bmpu -group sa -group pe[$row][$col] -group lbmac[$row][$col][$id] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/i_bmpu/u_sa/row_gen[$row]/col_gen[$col]/u_pe/pe_array[$id]/u_lbmac/*
        }
    }
}


add wave -noupdate -group Ara -group Lane[$1] /ara_tb/dut/i_ara_soc/i_system/i_ara/gen_lanes[$1]/i_lane/*
