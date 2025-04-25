`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/18 22:55:09
// Design Name: 
// Module Name: tb_sa
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


module tb_sa;

  // ----------------------
  // 参数定义
  // ----------------------
  parameter ROWS = 4;  // 阵列行数
  parameter COLS = 4;  // 阵列列数
  parameter BIT_ACT = 8;  // 激活值位宽（int8）
  parameter BIT_WEIGHT = 1;  // 权值位宽（int1）
  parameter CLK_PERIOD = 10;  // 时钟周期（10ns）

  // ----------------------
  // 信号定义
  // ----------------------
  logic clk;
  logic rst_n;
  logic en;
  logic output_en;
  logic [63:0] act_in[ROWS];  // 激活输入（每行一个 64 位打包数据）
  logic [63:0] weight_in[COLS];  // 权值输入（每列一个 64 位打包数据）
  logic [63:0] output_data[ROWS];  // 输出矩阵

  // ----------------------
  // 多维数组定义
  // ----------------------
  typedef logic [7:0] act_row_t[32];  // 每行 32 个 int8
  act_row_t act_matrix[4];  // 4 行，每行 32 个 int8

  typedef logic weight_row_t[32];  // 每行 32 个 int1
  weight_row_t weight_matrix[32];  // 32 行，每行 32 个 int1

  // ----------------------
  // 实例化被测模块
  // ----------------------
  sa #(
      .ROWS(ROWS),
      .COLS(COLS),
      .BIT_ACT(BIT_ACT),
      .BIT_WEIGHT(BIT_WEIGHT)
  ) u_sa (
      .clk(clk),
      .rst_n(rst_n),
      .en(en),
      .output_en(output_en),
      .act_in(act_in),
      .weight_in(weight_in),
      .output_data(output_data)
  );

  // ----------------------
  // 时钟生成
  // ----------------------
  initial begin
    clk = 0;
    forever #(CLK_PERIOD / 2) clk = ~clk;
  end

  // ----------------------
  // 测试逻辑
  // ----------------------
  initial begin
    // 初始化信号
    rst_n = 0;
    en = 0;
    output_en = 0;
    for (int i = 0; i < ROWS; i++) act_in[i] = 0;
    for (int j = 0; j < COLS; j++) weight_in[j] = 0;
    #(CLK_PERIOD * 2);

    // 释放复位
    rst_n = 1;
    #(CLK_PERIOD);

    // 启动计算
    en = 1;

    // ----------------------
    // 初始化激活矩阵（4x32，每个元素为 int8）
    // ----------------------
    for (int i = 0; i < 4; i++) begin
      for (int j = 0; j < 32; j++) begin
        act_matrix[i][j] = 1;  // 填充激活值为 1
      end
    end

    // ----------------------
    // 初始化权值矩阵（32x32，每个元素为 int1）
    // ----------------------
    for (int i = 0; i < 32; i++) begin
      for (int j = 0; j < 32; j++) begin
        weight_matrix[i][j] = 1;  // 填充权值为 1
      end
    end

    // ----------------------
    // 按梯形输入激活/权值数据
    // ----------------------
    for (int cycle = 0; cycle < ROWS + COLS + ROWS; cycle++) begin
      if (cycle < ROWS - 1 + COLS) begin
        for (int i = 0; i < ROWS; i++) begin
          if ((i <= cycle && cycle < ROWS) || (i > cycle % ROWS && cycle >= ROWS)) begin
            act_in[i] = {
              act_matrix[i][cycle+7],
              act_matrix[i][cycle+6],
              act_matrix[i][cycle+5],
              act_matrix[i][cycle+4],
              act_matrix[i][cycle+3],
              act_matrix[i][cycle+2],
              act_matrix[i][cycle+1],
              act_matrix[i][cycle]
            };
          end else begin
            act_in[i] = 0;  // 超出范围时填充 0
          end
        end
        for (int j = 0; j < COLS; j++) begin
          if ((j <= cycle && cycle < COLS) || (j > cycle % COLS && cycle >= COLS)) begin
            weight_in[j] = {
              weight_matrix[cycle+7][j+7],
              weight_matrix[cycle+7][j+6],
              weight_matrix[cycle+7][j+5],
              weight_matrix[cycle+7][j+4],
              weight_matrix[cycle+7][j+3],
              weight_matrix[cycle+7][j+2],
              weight_matrix[cycle+7][j+1],
              weight_matrix[cycle+7][j],
              weight_matrix[cycle+6][j+7],
              weight_matrix[cycle+6][j+6],
              weight_matrix[cycle+6][j+5],
              weight_matrix[cycle+6][j+4],
              weight_matrix[cycle+6][j+3],
              weight_matrix[cycle+6][j+2],
              weight_matrix[cycle+6][j+1],
              weight_matrix[cycle+6][j],
              weight_matrix[cycle+5][j+7],
              weight_matrix[cycle+5][j+6],
              weight_matrix[cycle+5][j+5],
              weight_matrix[cycle+5][j+4],
              weight_matrix[cycle+5][j+3],
              weight_matrix[cycle+5][j+2],
              weight_matrix[cycle+5][j+1],
              weight_matrix[cycle+5][j],
              weight_matrix[cycle+4][j+7],
              weight_matrix[cycle+4][j+6],
              weight_matrix[cycle+4][j+5],
              weight_matrix[cycle+4][j+4],
              weight_matrix[cycle+4][j+3],
              weight_matrix[cycle+4][j+2],
              weight_matrix[cycle+4][j+1],
              weight_matrix[cycle+4][j],
              weight_matrix[cycle+3][j+7],
              weight_matrix[cycle+3][j+6],
              weight_matrix[cycle+3][j+5],
              weight_matrix[cycle+3][j+4],
              weight_matrix[cycle+3][j+3],
              weight_matrix[cycle+3][j+2],
              weight_matrix[cycle+3][j+1],
              weight_matrix[cycle+3][j],
              weight_matrix[cycle+2][j+7],
              weight_matrix[cycle+2][j+6],
              weight_matrix[cycle+2][j+5],
              weight_matrix[cycle+2][j+4],
              weight_matrix[cycle+2][j+3],
              weight_matrix[cycle+2][j+2],
              weight_matrix[cycle+2][j+1],
              weight_matrix[cycle+2][j],
              weight_matrix[cycle+1][j+7],
              weight_matrix[cycle+1][j+6],
              weight_matrix[cycle+1][j+5],
              weight_matrix[cycle+1][j+4],
              weight_matrix[cycle+1][j+3],
              weight_matrix[cycle+1][j+2],
              weight_matrix[cycle+1][j+1],
              weight_matrix[cycle+1][j],
              weight_matrix[cycle][j+7],
              weight_matrix[cycle][j+6],
              weight_matrix[cycle][j+5],
              weight_matrix[cycle][j+4],
              weight_matrix[cycle][j+3],
              weight_matrix[cycle][j+2],
              weight_matrix[cycle][j+1],
              weight_matrix[cycle][j]
            };
          end else begin
            weight_in[j] = 0;  // 超出范围时填充 0
          end
        end
      end else begin
        for (int i = 0; i < ROWS; i++) begin
          act_in[i] = 0;
        end
        for (int j = 0; j < COLS; j++) begin
          weight_in[j] = 0;
        end
      end
      #(CLK_PERIOD);
    end

    output_en = 1;
    // 打印输出结果
    #(CLK_PERIOD);
    for (int i = 0; i < COLS; i++) begin
      $display("Output Data:");
      for (int i = 0; i < ROWS; i++) begin
        $display("output_data[%0d] = %h", i, output_data[i]);
      end
    end
    output_en = 0;

    // 结束仿真
    $stop;
  end

endmodule
