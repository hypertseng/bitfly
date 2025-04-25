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
    input  logic        clk,
    input  logic        rst_n,
    input  logic        en,                 // 全局使能
    input  logic        output_en,          // 输出使能（启动输出阶段）
    // 输入数据接口（64 位打包数据）
    input  logic [63:0] act_in     [ROWS],  // 激活输入（每行一个 64 位数据）
    input  logic [63:0] weight_in  [COLS],  // 权值输入（每列一个 64 位数据）
    // 输出数据接口（64 位打包数据）
    output logic [63:0] output_data[ROWS]   // 最终输出（每行一个 64 位数据）
);

  // ----------------------
  // 参数校验（仿真时检查）
  // ----------------------
  initial begin
    assert (ROWS > 0 && COLS > 0)
    else $error("ROWS/COLS must be > 0");
    assert (BIT_ACT == 8)
    else $error("BIT_ACT must be 8");
    assert (BIT_WEIGHT == 1)
    else $error("BIT_WEIGHT must be 1");
  end

  // ----------------------
  // Tile 间信号连接（优化为寄存器传输）
  // ----------------------
  logic [63:0] act_reg   [ROWS][COLS];  // 激活向右传递
  logic [63:0] weight_reg[ROWS][COLS];  // 权值向下传递
  logic [63:0] output_reg[ROWS][COLS];  // 输出向左传递

  // ----------------------
  // 脉动阵列生成
  // ----------------------
  generate
    for (genvar i = 0; i < ROWS; i++) begin : row_gen
      for (genvar j = 0; j < COLS; j++) begin : col_gen
        tile #(
            .BIT_ACT   (BIT_ACT),
            .BIT_WEIGHT(BIT_WEIGHT)
        ) u_tile (
            .clk             (clk),
            .rst_n           (rst_n),
            .en              (en),
            .output_en       (output_en),
            // 激活输入：来自左侧或外部输入
            .activations     ((j == 0) ? act_in[i] : act_reg[i][j-1]),
            // 权值输入：来自上方或外部输入
            .weights         ((i == 0) ? weight_in[j] : weight_reg[i-1][j]),
            // 输出寄存器输入：来自右侧或置零
            .input_output_reg((j == COLS - 1) ? '0 : output_reg[i][j+1]),
            // 激活输出：向右传递
            .activation_out  (act_reg[i][j]),
            // 权值输出：向下传递
            .weight_out      (weight_reg[i][j]),
            // 输出寄存器：向左传递
            .output_out      (output_reg[i][j])
        );
      end
    end
  endgenerate

  // ----------------------
  // 输出捕获逻辑（流水线优化）
  // ----------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      foreach (output_data[i]) output_data[i] <= '0;
    end else if (output_en) begin
      foreach (output_data[i]) output_data[i] <= output_reg[i][0];
    end
  end

endmodule
