// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
// Description:
// This stage holds the operand queues, holding elements for the VRFs.

module operand_queues_stage import ara_pkg::*; import rvv_pkg::*; import cf_math_pkg::idx_width; #(
    parameter int     unsigned NrLanes          = 0,
    parameter int     unsigned VLEN             = 0,
    // Support for floating-point data types
    parameter fpu_support_e FPUSupport          = FPUSupportHalfSingleDouble,
    parameter type          operand_queue_cmd_t = logic
  ) (
    input  logic                                     clk_i,
    input  logic                                     rst_ni,
    input  logic            [idx_width(NrLanes)-1:0] lane_id_i,
    // Interface with the Vector Register File
    input  elen_t              [NrOperandQueues-1:0] operand_i,
    input  logic               [NrOperandQueues-1:0] operand_valid_i,
    // Input with the Operand Requester
    input  logic               [NrOperandQueues-1:0] operand_issued_i,
    output logic               [NrOperandQueues-1:0] operand_queue_ready_o,
    input  operand_queue_cmd_t [NrOperandQueues-1:0] operand_queue_cmd_i,
    input  logic               [NrOperandQueues-1:0] operand_queue_cmd_valid_i,
    // Support for store exception flush
    input  logic                                     lsu_ex_flush_i,
    output logic                                     lsu_ex_flush_o,
    // Interface with the Lane Sequencer
    output logic                                     mask_b_cmd_pop_o,
    // Interface with the Lane
    output logic                                     sldu_addrgen_cmd_pop_o,
    // Interface with the VFUs
    // ALU
    output elen_t              [1:0]                 alu_operand_o,
    output logic               [1:0]                 alu_operand_valid_o,
    input  logic               [1:0]                 alu_operand_ready_i,
    // Multiplier/FPU
    output elen_t              [2:0]                 mfpu_operand_o,
    output logic               [2:0]                 mfpu_operand_valid_o,
    input  logic               [2:0]                 mfpu_operand_ready_i,
    // Store unit
    output elen_t                                    stu_operand_o,
    output logic                                     stu_operand_valid_o,
    input  logic                                     stu_operand_ready_i,
    // Slide Unit/Address Generation unit
    output elen_t                                    sldu_addrgen_operand_o,
    output target_fu_e                               sldu_addrgen_operand_target_fu_o,
    output logic                                     sldu_addrgen_operand_valid_o,
    input  logic                                     sldu_addrgen_operand_ready_i,
    // Mask unit
    output elen_t              [1:0]                 mask_operand_o,
    output logic               [1:0]                 mask_operand_valid_o,
    input  logic               [1:0]                 mask_operand_ready_i,
    // SA 队列接口
    output elen_t              [3:0] sa_act_operand_o,      // SA 激活输入
    output logic               [3:0] sa_act_operand_valid_o,
    input  logic               [3:0] sa_act_operand_ready_i,

    output elen_t              [3:0] sa_op_operand_o,       // SA 操作输入
    output logic               [3:0] sa_op_operand_valid_o,
    input  logic               [3:0] sa_op_operand_ready_i,

    input  elen_t              [3:0] sa_output_data_i,      // SA 输出
    input  logic               [3:0] sa_output_valid_i,
    output logic               [3:0] sa_output_ready_o
  );

  `include "common_cells/registers.svh"

  // STU flush support
  `FF(lsu_ex_flush_o, lsu_ex_flush_i, 1'b0, clk_i, rst_ni);

  ///////////
  //  ALU  //
  ///////////

  operand_queue #(
    .CmdBufDepth        (ValuInsnQueueDepth   ),
    .DataBufDepth       (5                    ),
    .FPUSupport         (FPUSupportNone       ),
    .NrLanes            (NrLanes              ),
    .VLEN               (VLEN                 ),
    .SupportIntExt2     (1'b1                 ),
    .SupportIntExt4     (1'b1                 ),
    .SupportIntExt8     (1'b1                 ),
    .SupportReduct      (1'b1                 ),
    .SupportNtrVal      (1'b0                 ),
    .operand_queue_cmd_t(operand_queue_cmd_t  )
  ) i_operand_queue_alu_a (
    .clk_i                    (clk_i                          ),
    .rst_ni                   (rst_ni                         ),
    .flush_i                  (1'b0                           ),
    .lane_id_i                (lane_id_i                      ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[AluA]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[AluA]),
    .cmd_pop_o                (/* Unused */                   ),
    .operand_i                (operand_i[AluA]                ),
    .operand_valid_i          (operand_valid_i[AluA]          ),
    .operand_issued_i         (operand_issued_i[AluA]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[AluA]    ),
    .operand_o                (alu_operand_o[0]               ),
    .operand_target_fu_o      (/* Unused */                   ),
    .operand_valid_o          (alu_operand_valid_o[0]         ),
    .operand_ready_i          (alu_operand_ready_i[0]         )
  );

  operand_queue #(
    .CmdBufDepth        (ValuInsnQueueDepth   ),
    .DataBufDepth       (5                    ),
    .FPUSupport         (FPUSupportNone       ),
    .NrLanes            (NrLanes              ),
    .VLEN               (VLEN                 ),
    .SupportIntExt2     (1'b1                 ),
    .SupportIntExt4     (1'b1                 ),
    .SupportIntExt8     (1'b1                 ),
    .SupportReduct      (1'b1                 ),
    .SupportNtrVal      (1'b1                 ),
    .operand_queue_cmd_t(operand_queue_cmd_t  )
  ) i_operand_queue_alu_b (
    .clk_i                    (clk_i                          ),
    .rst_ni                   (rst_ni                         ),
    .flush_i                  (1'b0                           ),
    .lane_id_i                (lane_id_i                      ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[AluB]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[AluB]),
    .cmd_pop_o                (/* Unused */                   ),
    .operand_i                (operand_i[AluB]                ),
    .operand_valid_i          (operand_valid_i[AluB]          ),
    .operand_issued_i         (operand_issued_i[AluB]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[AluB]    ),
    .operand_o                (alu_operand_o[1]               ),
    .operand_target_fu_o      (/* Unused */                   ),
    .operand_valid_o          (alu_operand_valid_o[1]         ),
    .operand_ready_i          (alu_operand_ready_i[1]         )
  );

  //////////////////////
  //  Multiplier/FPU  //
  //////////////////////

  operand_queue #(
    .CmdBufDepth        (MfpuInsnQueueDepth   ),
    .DataBufDepth       (5                    ),
    .FPUSupport         (FPUSupport           ),
    .NrLanes            (NrLanes              ),
    .VLEN               (VLEN                 ),
    .SupportIntExt2     (1'b1                 ),
    .SupportReduct      (1'b1                 ),
    .SupportNtrVal      (1'b0                 ),
    .operand_queue_cmd_t(operand_queue_cmd_t)
  ) i_operand_queue_mfpu_a (
    .clk_i                    (clk_i                             ),
    .rst_ni                   (rst_ni                            ),
    .flush_i                  (1'b0                              ),
    .lane_id_i                (lane_id_i                         ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[MulFPUA]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MulFPUA]),
    .cmd_pop_o                (/* Unused */                      ),
    .operand_i                (operand_i[MulFPUA]                ),
    .operand_valid_i          (operand_valid_i[MulFPUA]          ),
    .operand_issued_i         (operand_issued_i[MulFPUA]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[MulFPUA]    ),
    .operand_o                (mfpu_operand_o[0]                 ),
    .operand_target_fu_o      (/* Unused */                      ),
    .operand_valid_o          (mfpu_operand_valid_o[0]           ),
    .operand_ready_i          (mfpu_operand_ready_i[0]           )
  );

  operand_queue #(
    .CmdBufDepth        (MfpuInsnQueueDepth   ),
    .DataBufDepth       (5                    ),
    .FPUSupport         (FPUSupport           ),
    .NrLanes            (NrLanes              ),
    .VLEN               (VLEN                 ),
    .SupportIntExt2     (1'b1                 ),
    .SupportReduct      (1'b1                 ),
    .SupportNtrVal      (1'b1                 ),
    .operand_queue_cmd_t(operand_queue_cmd_t  )
  ) i_operand_queue_mfpu_b (
    .clk_i                    (clk_i                             ),
    .rst_ni                   (rst_ni                            ),
    .flush_i                  (1'b0                              ),
    .lane_id_i                (lane_id_i                         ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[MulFPUB]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MulFPUB]),
    .cmd_pop_o                (/* Unused */                      ),
    .operand_i                (operand_i[MulFPUB]                ),
    .operand_valid_i          (operand_valid_i[MulFPUB]          ),
    .operand_issued_i         (operand_issued_i[MulFPUB]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[MulFPUB]    ),
    .operand_o                (mfpu_operand_o[1]                 ),
    .operand_target_fu_o      (/* Unused */                      ),
    .operand_valid_o          (mfpu_operand_valid_o[1]           ),
    .operand_ready_i          (mfpu_operand_ready_i[1]           )
  );

  operand_queue #(
    .CmdBufDepth        (MfpuInsnQueueDepth   ),
    .DataBufDepth       (5                    ),
    .FPUSupport         (FPUSupport           ),
    .NrLanes            (NrLanes              ),
    .VLEN               (VLEN                 ),
    .SupportIntExt2     (1'b1                 ),
    .SupportReduct      (1'b1                 ),
    .SupportNtrVal      (1'b1                 ),
    .operand_queue_cmd_t(operand_queue_cmd_t  )
  ) i_operand_queue_mfpu_c (
    .clk_i                    (clk_i                             ),
    .rst_ni                   (rst_ni                            ),
    .flush_i                  (1'b0                              ),
    .lane_id_i                (lane_id_i                         ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[MulFPUC]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MulFPUC]),
    .cmd_pop_o                (/* Unused */                      ),
    .operand_i                (operand_i[MulFPUC]                ),
    .operand_valid_i          (operand_valid_i[MulFPUC]          ),
    .operand_issued_i         (operand_issued_i[MulFPUC]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[MulFPUC]    ),
    .operand_o                (mfpu_operand_o[2]                 ),
    .operand_target_fu_o      (/* Unused */                      ),
    .operand_valid_o          (mfpu_operand_valid_o[2]           ),
    .operand_ready_i          (mfpu_operand_ready_i[2]           )
  );

  ///////////////////////
  //  Load/Store Unit  //
  ///////////////////////

  operand_queue #(
    .CmdBufDepth        (VstuInsnQueueDepth + MaskuInsnQueueDepth),
    .DataBufDepth       (2                                       ),
    .FPUSupport         (FPUSupportNone                          ),
    .NrLanes            (NrLanes                                 ),
    .VLEN               (VLEN                                    ),
    .operand_queue_cmd_t(operand_queue_cmd_t                     )
  ) i_operand_queue_st_mask_a (
    .clk_i                    (clk_i                         ),
    .rst_ni                   (rst_ni                        ),
    .flush_i                  (lsu_ex_flush_o                ),
    .lane_id_i                (lane_id_i                     ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[StA]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[StA]),
    .cmd_pop_o                (/* Unused */                  ),
    .operand_i                (operand_i[StA]                ),
    .operand_valid_i          (operand_valid_i[StA]          ),
    .operand_issued_i         (operand_issued_i[StA]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[StA]    ),
    .operand_o                (stu_operand_o                 ),
    .operand_target_fu_o      (/* Unused */                  ),
    .operand_valid_o          (stu_operand_valid_o           ),
    .operand_ready_i          (stu_operand_ready_i           )
  );

  /****************
   *  Slide Unit  *
   ****************/

  operand_queue #(
    .CmdBufDepth        (VlduInsnQueueDepth   ),
    .DataBufDepth       (2                    ),
    .AccessCmdPop       (1'b1                 ),
    .FPUSupport         (FPUSupportNone       ),
    .NrLanes            (NrLanes              ),
    .VLEN               (VLEN                 ),
    .operand_queue_cmd_t(operand_queue_cmd_t  )
  ) i_operand_queue_slide_addrgen_a (
    .clk_i                    (clk_i                                   ),
    .rst_ni                   (rst_ni                                  ),
    .flush_i                  (lsu_ex_flush_o                          ),
    .lane_id_i                (lane_id_i                               ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[SlideAddrGenA]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[SlideAddrGenA]),
    .cmd_pop_o                (sldu_addrgen_cmd_pop_o                  ),
    .operand_i                (operand_i[SlideAddrGenA]                ),
    .operand_valid_i          (operand_valid_i[SlideAddrGenA]          ),
    .operand_issued_i         (operand_issued_i[SlideAddrGenA]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[SlideAddrGenA]    ),
    .operand_o                (sldu_addrgen_operand_o                  ),
    .operand_target_fu_o      (sldu_addrgen_operand_target_fu_o        ),
    .operand_valid_o          (sldu_addrgen_operand_valid_o            ),
    .operand_ready_i          (sldu_addrgen_operand_ready_i            )
  );

  /////////////////
  //  Mask Unit  //
  /////////////////

  operand_queue #(
    .CmdBufDepth        (MaskuInsnQueueDepth + VrgatherOpQueueBufDepth ),
    .DataBufDepth       (MaskuInsnQueueDepth + VrgatherOpQueueBufDepth ),
    .AccessCmdPop       (1'b1                                          ),
    .FPUSupport         (FPUSupportNone                                ),
    .SupportIntExt2     (1'b1                                          ),
    .SupportIntExt4     (1'b1                                          ),
    .SupportIntExt8     (1'b1                                          ),
    .NrLanes            (NrLanes                                       ),
    .VLEN               (VLEN                                          ),
    .operand_queue_cmd_t(operand_queue_cmd_t                           )
  ) i_operand_queue_mask_b (
    .clk_i                    (clk_i                           ),
    .rst_ni                   (rst_ni                          ),
    .flush_i                  (1'b0                            ),
    .lane_id_i                (lane_id_i                       ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[MaskB]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MaskB]),
    .cmd_pop_o                (mask_b_cmd_pop_o                ),
    .operand_i                (operand_i[MaskB]                ),
    .operand_valid_i          (operand_valid_i[MaskB]          ),
    .operand_issued_i         (operand_issued_i[MaskB]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[MaskB]    ),
    .operand_o                (mask_operand_o[1]               ),
    .operand_target_fu_o      (/* Unused */                    ),
    .operand_valid_o          (mask_operand_valid_o[1]         ),
    .operand_ready_i          (mask_operand_ready_i[1]         )
  );

  operand_queue #(
    .CmdBufDepth        (MaskuInsnQueueDepth  ),
    .DataBufDepth       (1                    ),
    .FPUSupport         (FPUSupportNone       ),
    .NrLanes            (NrLanes              ),
    .VLEN               (VLEN                 ),
    .operand_queue_cmd_t(operand_queue_cmd_t  )
  ) i_operand_queue_mask_m (
    .clk_i                    (clk_i                           ),
    .rst_ni                   (rst_ni                          ),
    .flush_i                  (lsu_ex_flush_o                  ),
    .lane_id_i                (lane_id_i                       ),
    .operand_queue_cmd_i      (operand_queue_cmd_i[MaskM]      ),
    .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[MaskM]),
    .cmd_pop_o                (/* Unused */                    ),
    .operand_i                (operand_i[MaskM]                ),
    .operand_valid_i          (operand_valid_i[MaskM]          ),
    .operand_issued_i         (operand_issued_i[MaskM]         ),
    .operand_queue_ready_o    (operand_queue_ready_o[MaskM]    ),
    .operand_o                (mask_operand_o[0]               ),
    .operand_target_fu_o      (/* Unused */                    ),
    .operand_valid_o          (mask_operand_valid_o[0]         ),
    .operand_ready_i          (mask_operand_ready_i[0]         )
  );

  // Checks
  if (VrgatherOpQueueBufDepth % 2 != 0) $fatal(1, "Parameter VrgatherOpQueueBufDepth must be power of 2.");

  // 实例化 SA 激活值操作数队列 (复用 Store Unit 队列结构)
  generate
    for (genvar i = 0; i < 4; i++) begin : gen_sa_act_queues
      operand_queue #(
        .CmdBufDepth        (VstuInsnQueueDepth),
        .DataBufDepth       (5),
        .FPUSupport         (FPUSupportNone),
        .NrLanes            (NrLanes),
        .VLEN               (VLEN),
        .operand_queue_cmd_t(operand_queue_cmd_t)
      ) i_sa_act_queue (
        .clk_i                    (clk_i),
        .rst_ni                   (rst_ni),
        .flush_i                  (lsu_ex_flush_o),
        .lane_id_i                (lane_id_i),
        .operand_queue_cmd_i      (operand_queue_cmd_i[SAAct0 + i]),
        .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[SAAct0 + i]),
        .cmd_pop_o                (/* 无命令弹出 */),
        .operand_i                (operand_i[SAAct0 + i]),
        .operand_valid_i          (operand_valid_i[SAAct0 + i]),
        .operand_issued_i         (operand_issued_i[SAAct0 + i]),
        .operand_queue_ready_o    (operand_queue_ready_o[SAAct0 + i]),
        .operand_o                (sa_act_operand_o[i]),
        .operand_valid_o          (sa_act_operand_valid_o[i]),
        .operand_ready_i          (sa_act_operand_ready_i[i])
      );
    end
  endgenerate

  // 实例化 SA 操作队列 (复用 ALU 队列结构)
  generate
    for (genvar i = 0; i < 4; i++) begin : gen_sa_op_queues
      operand_queue #(
        .CmdBufDepth        (ValuInsnQueueDepth),
        .DataBufDepth       (5),
        .FPUSupport         (FPUSupportNone),
        .NrLanes            (NrLanes),
        .VLEN               (VLEN),
        .operand_queue_cmd_t(operand_queue_cmd_t)
      ) i_sa_op_queue (
        .clk_i                    (clk_i),
        .rst_ni                   (rst_ni),
        .flush_i                  (1'b0),
        .lane_id_i                (lane_id_i),
        .operand_queue_cmd_i      (operand_queue_cmd_i[SAOp0 + i]),
        .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[SAOp0 + i]),
        .cmd_pop_o                (/* 无命令弹出 */),
        .operand_i                (operand_i[SAOp0 + i]),
        .operand_valid_i          (operand_valid_i[SAOp0 + i]),
        .operand_issued_i         (operand_issued_i[SAOp0 + i]),
        .operand_queue_ready_o    (operand_queue_ready_o[SAOp0 + i]),
        .operand_o                (sa_op_operand_o[i]),
        .operand_valid_o          (sa_op_operand_valid_o[i]),
        .operand_ready_i          (sa_op_operand_ready_i[i])
      );
    end
  endgenerate

  // SA 输出队列 (复用 Slide Unit 队列结构)
  generate
    for (genvar i = 0; i < 4; i++) begin : gen_sa_output_queues
      operand_queue #(
        .CmdBufDepth        (VlduInsnQueueDepth),
        .DataBufDepth       (2),
        .AccessCmdPop       (1'b1),
        .FPUSupport         (FPUSupportNone),
        .operand_queue_cmd_t(operand_queue_cmd_t)
      ) i_sa_output_queue (
        .clk_i                    (clk_i),
        .rst_ni                   (rst_ni),
        .flush_i                  (lsu_ex_flush_o),
        .lane_id_i                (lane_id_i),
        .operand_queue_cmd_i      (operand_queue_cmd_i[SAOut0 + i]),
        .operand_queue_cmd_valid_i(operand_queue_cmd_valid_i[SAOut0 + i]),
        .cmd_pop_o                (/* 输出队列命令弹出逻辑 */),
        .operand_i                (sa_output_data_i[i]),
        .operand_valid_i          (sa_output_valid_i[i]),
        .operand_issued_i         (/* SA 输出不需要 issued 信号 */),
        .operand_queue_ready_o    (sa_output_ready_o[i]),
        .operand_o                (/* 连接到目标单元 */),
        .operand_valid_o          (/* 输出到目标单元 */),
        .operand_ready_i          (/* 目标单元 ready 信号 */)
      );
    end
  endgenerate

endmodule : operand_queues_stage
