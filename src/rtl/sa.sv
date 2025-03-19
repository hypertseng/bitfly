`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/18 22:54:38
// Design Name: 
// Module Name: sa
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


module sa #(
    parameter ROWS       = 4,  // 阵列行数（可配置）
    parameter COLS       = 4,  // 阵列列数（可配置）
    parameter BIT_ACT    = 8,  // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1   // 权值位宽（int1）
) (
    input logic clk,
    input logic rst_n,
    input logic en,  // 全局使能
    input logic output_en,  // 输出使能（启动输出阶段）
    // 输入数据接口（梯形输入，64 位打包数据）
    input logic [63:0] act_in[ROWS],  // 激活输入（每行一个 64 位打包数据）
    input logic [63:0] weight_in[COLS],  // 权值输入（每列一个 64 位打包数据）
    // 输出数据接口（64 位打包数据）
    output logic [63:0] output_data[ROWS]  // 最终输出接口（64 位打包）
);

  // ----------------------
  // Tile 间信号连接定义
  // ----------------------
  logic [63:0] act_reg[ROWS][COLS];  // 激活寄存器（向右传递，64 位打包）
  logic [63:0] weight_reg[ROWS][COLS];  // 权值寄存器（向下传递，64 位打包）
  logic [63:0] output_reg[ROWS][COLS];  // 输出寄存器（向左传递，64 位打包）

  // ----------------------
  // 生成脉动阵列
  // ----------------------
  generate
    for (genvar i = 0; i < ROWS; i++) begin : row_gen
      for (genvar j = 0; j < COLS; j++) begin : col_gen
        tile u_tile (
            .clk             (clk),
            .rst_n           (rst_n),
            .en              (en),
            .output_en       (output_en),
            // 激活输入：来自左侧或外部输入（64 位打包）
            .activations     ((j == 0) ? act_in[i] : act_reg[i][j-1]),
            // 权值输入：来自上方或外部输入（64 位打包）
            .weights         ((i == 0) ? weight_in[j] : weight_reg[i-1][j]),
            // 输出寄存器输入：来自右侧（64 位打包）
            .input_output_reg((j == COLS - 1) ? '0 : output_reg[i][j+1]),
            // 激活输出：向右传递（64 位打包）
            .activation_out  (act_reg[i][j]),
            // 权值输出：向下传递（64 位打包）
            .weight_out      (weight_reg[i][j]),
            // 输出寄存器：向左传递（64 位打包）
            .output_out      (output_reg[i][j])
        );
      end
    end
  endgenerate

  // ----------------------
  // 捕获最终输出（64 位打包）
  // ----------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < ROWS; i++) begin
        output_data[i] <= '0;
      end
    end else if (output_en) begin
      for (int i = 0; i < ROWS; i++) begin
          output_data[i] <= output_reg[i][0];
      end
    end
  end

endmodule
