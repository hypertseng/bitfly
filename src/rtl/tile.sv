`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 15:45:42
// Design Name: 
// Module Name: tile
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


module tile (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        en,
    input  logic        output_en,         // 输出使能信号
    input  logic [63:0] activations,       // 64-bit 激活输入（8x int8）
    input  logic [63:0] weights,           // 64-bit 权重输入（8x int1）
    input  logic [63:0] input_output_reg,  // 来自其他 tile 的 output_reg 值
    output logic [63:0] weight_out,        // 权重寄存器输出
    output logic [63:0] activation_out,    // 激活寄存器输出
    output logic [63:0] output_out         // 输出寄存器输出（8x int8）
);

  // 流水线寄存器
  logic [63:0] activation_reg;  // 64-bit 激活寄存器
  logic [7:0] weight_regs[8];  // 8 个 8-bit 权重寄存器
  logic signed [15:0] patial_sum_regs[8];  // 8 个 16-bit 累加寄存器
  logic signed [7:0] output_regs[8];  // 8 个 8-bit 输出寄存器

  // PE 输出（unpacked array）
  logic signed [15:0] pe_outputs[8];  // PE 输出

  // 寄存器更新逻辑
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      activation_reg <= '0;
      for (int i = 0; i < 8; i++) begin
        weight_regs[i] <= '0;
        patial_sum_regs[i] <= '0;
        output_regs[i] <= '0;
      end
    end else begin
      // 优先级控制：output_en 优先于 en
      if (output_en) begin
        // 写入来自其他 tile 的新值
        for (int i = 0; i < 8; i++) begin
          output_regs[i] <= $signed(input_output_reg[i*8+:8]);  // 将 int8 转为 int8
        end
      end else if (en) begin
        // 累加写入 patial_sum_regs
        activation_reg <= activations;  // 直接存储 64-bit 激活数据

        // 权重分发：将64-bit输入拆分为8组8-bit
        for (int i = 0; i < 8; i++) begin
          weight_regs[i] <= weights[i*8+:8];
        end

        // 累加器：将 pe_outputs 与 patial_sum_regs 按部分累加
        for (int i = 0; i < 8; i++) begin
          logic signed [15:0] temp_sum;  // 16 位有符号数，用于累加
          temp_sum = $signed(pe_outputs[i]) + $signed(patial_sum_regs[i]);
          patial_sum_regs[i] <= temp_sum;  // 直接存储 16 位结果
        end

        // 将 patial_sum_regs 的值转换为 int8 并存储到 output_regs
        for (int i = 0; i < 8; i++) begin
          logic signed [7:0] saturated_value;  // 饱和后的 int8 值
          // 饱和处理
          if (patial_sum_regs[i] > $signed({1'b0, {7{1'b1}}})) begin
            saturated_value = 8'sh7F;  // 最大值 127
          end else if (patial_sum_regs[i] < $signed({1'b1, {7{1'b0}}})) begin
            //            $display("%d",$signed(patial_sum_regs[i]));
            //            $display("%d",$signed({1'b1, {7{1'b0}}}));
            saturated_value = 8'sh80;  // 最小值 -128
          end else begin
            saturated_value = patial_sum_regs[i][7:0];  // 取低 8 位
            //            $display("%d",$signed(saturated_value));
          end
          output_regs[i] <= saturated_value;  // 存储到 output_regs
        end
      end
    end
  end

  // PE阵列实例化
  generate
    for (genvar i = 0; i < 8; i++) begin : pe_array
      pe u_pe (
          .weights    (weights[i*8+:8]),  // 每个 PE 获得 8-bit 权重
          .activations(activations),      // 每个 PE 获得 8-int8 激活
          .result     (pe_outputs[i])     // PE 输出
      );
    end
  endgenerate

  // 输出逻辑
  assign weight_out = {
    weight_regs[7],
    weight_regs[6],
    weight_regs[5],
    weight_regs[4],
    weight_regs[3],
    weight_regs[2],
    weight_regs[1],
    weight_regs[0]
  };
  assign activation_out = activation_reg;
  
  // 输出当前 output_regs 的值
  always_comb begin
    for (int i = 0; i < 8; i++) begin
      output_out[i*8+:8] = output_regs[i];  // 输出 output_regs 的值
    end
  end
    
endmodule
