`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 15:39:54
// Design Name: 
// Module Name: pe
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module pe #(
    parameter BIT_ACT    = 8,  // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1   // 权值位宽（int1）
) (
    input  logic [ 7:0] weights,      // 8-bit 权重（1:保持原值，0:取反）
    input  logic [63:0] activations,  // 64-bit 激活输入（8x int8）
    output logic [11:0] result        // 12-bit 有符号结果
);

  // ----------------------
  // 加权激活值计算
  // ----------------------
  logic signed [7:0] weighted_activations[8];  // 8个加权后的激活值（int8）

  always_comb begin
    for (int i = 0; i < 8; i++) begin
      // 权重为0时取反（补码操作），注意激活值按高位到低位排列
      // activations[63:56] 对应 i=0（第一个int8）
      weighted_activations[i] = weights[i] ? 
          activations[( (7 - i) * 8 ) +:8] :  // 高位在前
          ~activations[( (7 - i) * 8 ) +:8] + 1;
    end
  end

  // ----------------------
  // 树形累加结构（优化时序）
  // ----------------------
  logic signed [10:0] sum_stage1[4];  // 第一级累加（11-bit）
  logic signed [11:0] sum_stage2[2];  // 第二级累加（12-bit）

  always_comb begin
    // 第一级：两两相加
    sum_stage1[0] = weighted_activations[0] + weighted_activations[1];
    sum_stage1[1] = weighted_activations[2] + weighted_activations[3];
    sum_stage1[2] = weighted_activations[4] + weighted_activations[5];
    sum_stage1[3] = weighted_activations[6] + weighted_activations[7];

    // 第二级：两两相加
    sum_stage2[0] = sum_stage1[0] + sum_stage1[1];
    sum_stage2[1] = sum_stage1[2] + sum_stage1[3];

    // 最终结果
    result = sum_stage2[0] + sum_stage2[1];
  end

endmodule
