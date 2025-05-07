// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
// Description:
// This stage holds the operand queues, holding elements for the VRFs.

module operand_queues_stage
  import ara_pkg::*;
  import rvv_pkg::*;
  import cf_math_pkg::idx_width;
#(
    parameter int unsigned  NrLanes             = 0,
    parameter int unsigned  VLEN                = 0,
    // Support for floating-point data types
    parameter fpu_support_e FPUSupport          = FPUSupportHalfSingleDouble,
    parameter type          operand_queue_cmd_t = logic
) (
    input logic clk_i,
    input logic rst_ni,
    input logic [idx_width(NrLanes)-1:0] lane_id_i,
    // Interface with the Vector Register File
    input elen_t [NrOperandQueues-1:0] operand_i,
    input logic [NrOperandQueues-1:0] operand_valid_i,
    // Input with the Operand Requester
    input logic [NrOperandQueues-1:0] operand_issued_i,
    output logic [NrOperandQueues-1:0] operand_queue_ready_o,
    input operand_queue_cmd_t [NrOperandQueues-1:0] operand_queue_cmd_i,
    input logic [NrOperandQueues-1:0] operand_queue_cmd_valid_i,
    // Support for store exception flush
    input logic lsu_ex_flush_i,
    output logic lsu_ex_flush_o,
    // Interface with the Lane Sequencer
    output logic mask_b_cmd_pop_o,
    // Interface with the Lane
    output logic sldu_addrgen_cmd_pop_o,
    // Interface with the VFUs
    // ALU
    output elen_t [1:0] alu_operand_o,
    output logic [1:0] alu_operand_valid_o,
    input logic [1:0] alu_operand_ready_i,
    // Multiplier/FPU
    output elen_t [2:0] mfpu_operand_o,
    output logic [2:0] mfpu_operand_valid_o,
    input logic [2:0] mfpu_operand_ready_i,
    // Store unit
    output elen_t stu_operand_o,
    output logic stu_operand_valid_o,
    input logic stu_operand_ready_i,
    // Slide Unit/Address Generation unit
    output elen_t sldu_addrgen_operand_o,
    output target_fu_e sldu_addrgen_operand_target_fu_o,
    output logic sldu_addrgen_operand_valid_o,
    input logic sldu_addrgen_operand_ready_i,
    // Mask unit
    output elen_t [1:0] mask_operand_o,
    output logic [1:0] mask_operand_valid_o,
    input logic [1:0] mask_operand_ready_i,
    // SA 队列接口
    output elen_t mpu_act_operand_o [3:0],  // SA 激活输入
    output logic [3:0] mpu_act_operand_valid_o,
    input logic [3:0] mpu_act_operand_ready_i,

    output elen_t mpu_wgt_operand_o[3:0],        // SA 权重输入
    output logic  [3:0] mpu_wgt_operand_valid_o,
    input  logic  [3:0] mpu_wgt_operand_ready_i,

    input  elen_t mpu_output_data_o [3:0],   // SA 输出
    input  logic  [3:0] mpu_output_valid_o,
    output logic  [3:0] mpu_output_ready_i,

    input logic  mpu_output_en_i // MPU 输出使能
);

  `include "common_cells/registers.svh"

  // STU flush support
  `FF(lsu_ex_flush_o, lsu_ex_flush_i, 1'b0, clk_i, rst_ni);

  ///////////
  //  ALU  //
  ///////////

  operand_queue #(
      .CmdBufDepth        (ValuInsnQueueDepth),
      .DataBufDepth       (5),
      .FPUSupport         (FPUSupportNone),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .SupportIntExt2     (1'b1),
      .SupportIntExt4     (1'b1),
      .SupportIntExt8     (1'b1),
      .SupportReduct      (1'b1),
      .SupportNtrVal      (1'b0),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_alu_a (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (1'b0),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[AluA]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[AluA]),
      .cmd_pop_o                (  /* Unused */),
      .operand_i                (operand_i[AluA]),
      .operand_valid_i          (operand_valid_i[AluA]),
      .operand_issued_i         (operand_issued_i[AluA]),
      .operand_queue_ready_o    (operand_queue_ready_o[AluA]),
      .operand_o                (alu_operand_o[0]),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (alu_operand_valid_o[0]),
      .operand_ready_i          (alu_operand_ready_i[0])
  );

  operand_queue #(
      .CmdBufDepth        (ValuInsnQueueDepth),
      .DataBufDepth       (5),
      .FPUSupport         (FPUSupportNone),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .SupportIntExt2     (1'b1),
      .SupportIntExt4     (1'b1),
      .SupportIntExt8     (1'b1),
      .SupportReduct      (1'b1),
      .SupportNtrVal      (1'b1),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_alu_b (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (1'b0),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[AluB]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[AluB]),
      .cmd_pop_o                (  /* Unused */),
      .operand_i                (operand_i[AluB]),
      .operand_valid_i          (operand_valid_i[AluB]),
      .operand_issued_i         (operand_issued_i[AluB]),
      .operand_queue_ready_o    (operand_queue_ready_o[AluB]),
      .operand_o                (alu_operand_o[1]),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (alu_operand_valid_o[1]),
      .operand_ready_i          (alu_operand_ready_i[1])
  );

  //////////////////////
  //  Multiplier/FPU  //
  //////////////////////

  operand_queue #(
      .CmdBufDepth        (MfpuInsnQueueDepth),
      .DataBufDepth       (5),
      .FPUSupport         (FPUSupport),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .SupportIntExt2     (1'b1),
      .SupportReduct      (1'b1),
      .SupportNtrVal      (1'b0),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_mfpu_a (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (1'b0),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[MulFPUA]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MulFPUA]),
      .cmd_pop_o                (  /* Unused */),
      .operand_i                (operand_i[MulFPUA]),
      .operand_valid_i          (operand_valid_i[MulFPUA]),
      .operand_issued_i         (operand_issued_i[MulFPUA]),
      .operand_queue_ready_o    (operand_queue_ready_o[MulFPUA]),
      .operand_o                (mfpu_operand_o[0]),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (mfpu_operand_valid_o[0]),
      .operand_ready_i          (mfpu_operand_ready_i[0])
  );

  operand_queue #(
      .CmdBufDepth        (MfpuInsnQueueDepth),
      .DataBufDepth       (5),
      .FPUSupport         (FPUSupport),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .SupportIntExt2     (1'b1),
      .SupportReduct      (1'b1),
      .SupportNtrVal      (1'b1),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_mfpu_b (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (1'b0),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[MulFPUB]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MulFPUB]),
      .cmd_pop_o                (  /* Unused */),
      .operand_i                (operand_i[MulFPUB]),
      .operand_valid_i          (operand_valid_i[MulFPUB]),
      .operand_issued_i         (operand_issued_i[MulFPUB]),
      .operand_queue_ready_o    (operand_queue_ready_o[MulFPUB]),
      .operand_o                (mfpu_operand_o[1]),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (mfpu_operand_valid_o[1]),
      .operand_ready_i          (mfpu_operand_ready_i[1])
  );

  operand_queue #(
      .CmdBufDepth        (MfpuInsnQueueDepth),
      .DataBufDepth       (5),
      .FPUSupport         (FPUSupport),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .SupportIntExt2     (1'b1),
      .SupportReduct      (1'b1),
      .SupportNtrVal      (1'b1),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_mfpu_c (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (1'b0),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[MulFPUC]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MulFPUC]),
      .cmd_pop_o                (  /* Unused */),
      .operand_i                (operand_i[MulFPUC]),
      .operand_valid_i          (operand_valid_i[MulFPUC]),
      .operand_issued_i         (operand_issued_i[MulFPUC]),
      .operand_queue_ready_o    (operand_queue_ready_o[MulFPUC]),
      .operand_o                (mfpu_operand_o[2]),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (mfpu_operand_valid_o[2]),
      .operand_ready_i          (mfpu_operand_ready_i[2])
  );

  ///////////////////////
  //  Load/Store Unit  //
  ///////////////////////

  operand_queue #(
      .CmdBufDepth        (VstuInsnQueueDepth + MaskuInsnQueueDepth),
      .DataBufDepth       (2),
      .FPUSupport         (FPUSupportNone),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_st_mask_a (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (lsu_ex_flush_o),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[StA]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[StA]),
      .cmd_pop_o                (  /* Unused */),
      .operand_i                (operand_i[StA]),
      .operand_valid_i          (operand_valid_i[StA]),
      .operand_issued_i         (operand_issued_i[StA]),
      .operand_queue_ready_o    (operand_queue_ready_o[StA]),
      .operand_o                (stu_operand_o),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (stu_operand_valid_o),
      .operand_ready_i          (stu_operand_ready_i)
  );

  /****************
   *  Slide Unit  *
   ****************/

  operand_queue #(
      .CmdBufDepth        (VlduInsnQueueDepth),
      .DataBufDepth       (2),
      .AccessCmdPop       (1'b1),
      .FPUSupport         (FPUSupportNone),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_slide_addrgen_a (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (lsu_ex_flush_o),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[SlideAddrGenA]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[SlideAddrGenA]),
      .cmd_pop_o                (sldu_addrgen_cmd_pop_o),
      .operand_i                (operand_i[SlideAddrGenA]),
      .operand_valid_i          (operand_valid_i[SlideAddrGenA]),
      .operand_issued_i         (operand_issued_i[SlideAddrGenA]),
      .operand_queue_ready_o    (operand_queue_ready_o[SlideAddrGenA]),
      .operand_o                (sldu_addrgen_operand_o),
      .operand_target_fu_o      (sldu_addrgen_operand_target_fu_o),
      .operand_valid_o          (sldu_addrgen_operand_valid_o),
      .operand_ready_i          (sldu_addrgen_operand_ready_i)
  );

  /////////////////
  //  Mask Unit  //
  /////////////////

  operand_queue #(
      .CmdBufDepth        (MaskuInsnQueueDepth + VrgatherOpQueueBufDepth),
      .DataBufDepth       (MaskuInsnQueueDepth + VrgatherOpQueueBufDepth),
      .AccessCmdPop       (1'b1),
      .FPUSupport         (FPUSupportNone),
      .SupportIntExt2     (1'b1),
      .SupportIntExt4     (1'b1),
      .SupportIntExt8     (1'b1),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_mask_b (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (1'b0),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[MaskB]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MaskB]),
      .cmd_pop_o                (mask_b_cmd_pop_o),
      .operand_i                (operand_i[MaskB]),
      .operand_valid_i          (operand_valid_i[MaskB]),
      .operand_issued_i         (operand_issued_i[MaskB]),
      .operand_queue_ready_o    (operand_queue_ready_o[MaskB]),
      .operand_o                (mask_operand_o[1]),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (mask_operand_valid_o[1]),
      .operand_ready_i          (mask_operand_ready_i[1])
  );

  operand_queue #(
      .CmdBufDepth        (MaskuInsnQueueDepth),
      .DataBufDepth       (1),
      .FPUSupport         (FPUSupportNone),
      .NrLanes            (NrLanes),
      .VLEN               (VLEN),
      .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_mask_m (
      .clk_i                    (clk_i),
      .rst_ni                   (rst_ni),
      .flush_i                  (lsu_ex_flush_o),
      .lane_id_i                (lane_id_i),
      .operand_queue_cmd_i      (operand_queue_cmd_i[MaskM]),
      .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MaskM]),
      .cmd_pop_o                (  /* Unused */),
      .operand_i                (operand_i[MaskM]),
      .operand_valid_i          (operand_valid_i[MaskM]),
      .operand_issued_i         (operand_issued_i[MaskM]),
      .operand_queue_ready_o    (operand_queue_ready_o[MaskM]),
      .operand_o                (mask_operand_o[0]),
      .operand_target_fu_o      (  /* Unused */),
      .operand_valid_o          (mask_operand_valid_o[0]),
      .operand_ready_i          (mask_operand_ready_i[0])
  );

  // Checks
  if (VrgatherOpQueueBufDepth % 2 != 0)
    $fatal(1, "Parameter VrgatherOpQueueBufDepth must be power of 2.");

  // 模式控制信号
  logic queue_mode; // 0:activation输入模式, 1:输出模式
  logic activation_processed;
  elen_t shared_operand_o [3:0];
  logic [3:0] shared_operand_valid_o;
  logic [NrLanes-1:0] queue_ready_internal;

  // 共享的SA激活值/输出操作数队列
  generate
    for (genvar i = 0; i < 4; i++) begin : gen_mpu_act_out_queues
      // 共享队列实例
      operand_queue #(
          .CmdBufDepth        (MpuInsnQueueDepth),
          .DataBufDepth       (5),
          .FPUSupport         (FPUSupportNone),
          .NrLanes            (NrLanes),
          .VLEN               (VLEN),
          .operand_queue_cmd_t(operand_queue_cmd_t)
      ) i_mpu_shared_queue (
          .clk_i                    (clk_i),
          .rst_ni                   (rst_ni),
          .flush_i                  (1'b0),
          .lane_id_i                (lane_id_i),
          // 命令输入根据模式选择
          .operand_queue_cmd_i      (queue_mode ? operand_queue_cmd_i[MPUOut0+i] : operand_queue_cmd_i[MPUAct0+i]),
          .operand_queue_cmd_valid_i(queue_mode ? operand_queue_cmd_valid_i[MPUOut0+i] : operand_queue_cmd_valid_i[MPUAct0+i]),
          .cmd_pop_o                (),
          // 数据输入根据模式选择
          .operand_i                (queue_mode ? operand_i[MPUOut0+i] : operand_i[MPUAct0+i]),
          .operand_valid_i          (queue_mode ? operand_valid_i[MPUOut0+i] : operand_valid_i[MPUAct0+i]),
          .operand_issued_i         (queue_mode ? 1'b0 : operand_issued_i[MPUAct0+i]),
          .operand_queue_ready_o    (queue_ready_internal[i]),
          // 输出数据
          .operand_o                (shared_operand_o[i]),
          .operand_valid_o          (shared_operand_valid_o[i]),
          .operand_ready_i          (queue_mode ? mpu_output_ready_i[i] : mpu_act_operand_ready_i[i])
      );
      
      // 输出分配
      assign mpu_act_operand_o[i] = !queue_mode ? shared_operand_o[i] : '0;
      assign mpu_act_operand_valid_o[i] = !queue_mode ? shared_operand_valid_o[i] : '0;
      logic [VLEN-1:0] shared_output_data[4];
      assign shared_output_data[i] = shared_operand_o[i];
      // assign mpu_output_data_o[i] = queue_mode ? shared_operand_o[i] : '0;
      assign mpu_output_valid_o[i] = queue_mode ? shared_operand_valid_o[i] : '0;

      assign operand_queue_ready_o[MPUOut0+i] = queue_mode ? queue_ready_internal[i] : '0;
      assign operand_queue_ready_o[MPUAct0+i] = !queue_mode ? queue_ready_internal[i] : '0;
    end
  endgenerate

  assign queue_mode = mpu_output_en_i;

  // SA权重队列
  generate
    for (genvar i = 0; i < 4; i++) begin : gen_mpu_wgt_queues
      operand_queue #(
          .CmdBufDepth        (MpuInsnQueueDepth),
          .DataBufDepth       (5),
          .FPUSupport         (FPUSupportNone),
          .NrLanes            (NrLanes),
          .VLEN               (VLEN),
          .operand_queue_cmd_t(operand_queue_cmd_t)
      ) i_mpu_wgt_queue (
          .clk_i                    (clk_i),
          .rst_ni                   (rst_ni),
          .flush_i                  (1'b0),
          .lane_id_i                (lane_id_i),
          .operand_queue_cmd_i      (operand_queue_cmd_i[MPUWgt0+i]),
          .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MPUWgt0+i]),
          .cmd_pop_o                (),
          .operand_i                (operand_i[MPUWgt0+i]),
          .operand_valid_i          (operand_valid_i[MPUWgt0+i]),
          .operand_issued_i         (operand_issued_i[MPUWgt0+i]),
          .operand_queue_ready_o    (operand_queue_ready_o[MPUWgt0+i]),
          .operand_o                (mpu_wgt_operand_o[i]),
          .operand_valid_o          (mpu_wgt_operand_valid_o[i]),
          .operand_ready_i          (mpu_wgt_operand_ready_i[i])
      );
    end
  endgenerate

endmodule : operand_queues_stage
