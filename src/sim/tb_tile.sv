`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 16:07:47
// Design Name: 
// Module Name: tb_tile
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


module tb_tile ();

  // Inputs
  logic        clk;
  logic        rst_n;
  logic        en;
  logic        output_en;  // 输出使能信号
  logic [63:0] activations;  // 64-bit 激活输入（8x int8）
  logic [63:0] weights;  // 64-bit 权重输入（8x int1）
  logic [63:0] input_output_reg;  // 来自其他 tile 的 output_reg 值

  // Outputs
  logic [63:0] weight_out;  // 权重寄存器输出
  logic [63:0] activation_out;  // 激活寄存器输出
  logic [63:0] output_out;  // 输出寄存器输出（8x int8）

  // 实例化被测模块
  tile u_tile (
      .clk(clk),
      .rst_n(rst_n),
      .en(en),
      .output_en(output_en),
      .activations(activations),
      .weights(weights),
      .input_output_reg(input_output_reg),
      .weight_out(weight_out),
      .activation_out(activation_out),
      .output_out(output_out)
  );

  // 时钟生成
  always #5 clk = ~clk;

  // 测试用例
  initial begin
    // 初始化
    clk = 0;
    rst_n = 0;
    en = 0;
    output_en = 0;
    activations = '0;
    weights = '0;
    input_output_reg = 'x;

    // 复位
    #20 rst_n = 1;

    // 测试用例1：基本测试
    $display("\nTest Case 1: Basic Calculation");
    en          = 1;
    activations = {8'h01, 8'h02, 8'h03, 8'h04, 8'h05, 8'h06, 8'h07, 8'h08};  // 8x int8
    weights     = {8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF};  // 8x int1（全1）
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out: %h", weight_out);
    $display("Activation Out: %h %h %h %h %h %h %h %h", $signed(activation_out[63:56]),
             $signed(activation_out[55:48]), $signed(activation_out[47:40]),
             $signed(activation_out[39:32]), $signed(activation_out[31:24]),
             $signed(activation_out[23:16]), $signed(activation_out[15:8]),
             $signed(activation_out[7:0]));
    $display("Output Out: %d%d%d%d%d%d%d%d (Expected: 36 36 36 36 36 36 36 36)",
             $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
             $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
             $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 测试用例2：带符号测试
    $display("\nTest Case 2: Signed Calculation");
    activations = {8'hFF, 8'hFE, 8'hFD, 8'hFC, 8'hFB, 8'hFA, 8'hF9, 8'hF8};  // -1 to -8
    weights     = {8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF};  // 8x int1（全1）
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out: %h", weight_out);
    $display("Activation Out: %h %h %h %h %h %h %h %h", $signed(activation_out[63:56]),
             $signed(activation_out[55:48]), $signed(activation_out[47:40]),
             $signed(activation_out[39:32]), $signed(activation_out[31:24]),
             $signed(activation_out[23:16]), $signed(activation_out[15:8]),
             $signed(activation_out[7:0]));
    $display("Output Out: %d%d%d%d%d%d%d%d (Expected: -36 -36 -36 -36 -36 -36 -36 -36)",
             $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
             $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
             $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 测试用例3：累加测试
    $display("\nTest Case 3: Accumulation Test");
    en = 1;
    activations = {8'h10, 8'h10, 8'h10, 8'h10, 8'h10, 8'h10, 8'h10, 8'h10};  // 每个值为 16
    weights = {
      8'b00000001,
      8'b00000001,
      8'b00000001,
      8'b00000001,
      8'b00000001,
      8'b00000001,
      8'b00000001,
      8'b00000001
    };  // 每个值的最后一位是 1
    #10;

    $display("Weight Out after 1st cycle: %h", weight_out);
    $display("Activation Out after 1st cycle: %h %h %h %h %h %h %h %h",
             $signed(activation_out[63:56]), $signed(activation_out[55:48]),
             $signed(activation_out[47:40]), $signed(activation_out[39:32]),
             $signed(activation_out[31:24]), $signed(activation_out[23:16]),
             $signed(activation_out[15:8]), $signed(activation_out[7:0]));

    // 第二次输入
    activations = {8'h10, 8'h10, 8'h10, 8'h10, 8'h10, 8'h10, 8'h10, 8'h10};  // 每个值为 16
    weights = {
      8'b11111100,
      8'b11111100,
      8'b11111100,
      8'b11111100,
      8'b11111100,
      8'b11111100,
      8'b11111100,
      8'b11111100
    };
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out after 2nd cycle: %h", weight_out);
    $display("Activation Out after 2nd cycle: %h %h %h %h %h %h %h %h",
             $signed(activation_out[63:56]), $signed(activation_out[55:48]),
             $signed(activation_out[47:40]), $signed(activation_out[39:32]),
             $signed(activation_out[31:24]), $signed(activation_out[23:16]),
             $signed(activation_out[15:8]), $signed(activation_out[7:0]));
    $display(
        "Output Out after 2nd cycle: %d%d%d%d%d%d%d%d (Expected: -32 -32 -32 -32 -32 -32 -32 -32)",
        $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
        $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
        $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 测试用例4：全零输入测试
    $display("\nTest Case 4: All Zero Input");
    activations = {8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00}; // 全零激活
    weights = {8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00};     // 全零权重
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out: %h", weight_out);
    $display("Activation Out: %h %h %h %h %h %h %h %h", $signed(activation_out[63:56]),
             $signed(activation_out[55:48]), $signed(activation_out[47:40]),
             $signed(activation_out[39:32]), $signed(activation_out[31:24]),
             $signed(activation_out[23:16]), $signed(activation_out[15:8]),
             $signed(activation_out[7:0]));
    $display("Output Out: %d%d%d%d%d%d%d%d (Expected: 0 0 0 0 0 0 0 0)",
             $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
             $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
             $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 测试用例5：最大值测试
    $display("\nTest Case 5: Maximum Value Test");
    activations = {8'h7F, 8'h7F, 8'h7F, 8'h7F, 8'h7F, 8'h7F, 8'h7F, 8'h7F}; // 最大正值
    weights = {8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF};     // 全1权重
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out: %h", weight_out);
    $display("Activation Out: %h %h %h %h %h %h %h %h", $signed(activation_out[63:56]),
             $signed(activation_out[55:48]), $signed(activation_out[47:40]),
             $signed(activation_out[39:32]), $signed(activation_out[31:24]),
             $signed(activation_out[23:16]), $signed(activation_out[15:8]),
             $signed(activation_out[7:0]));
    $display("Output Out: %d%d%d%d%d%d%d%d (Expected: 127 127 127 127 127 127 127 127)",
             $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
             $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
             $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 测试用例6：最小值测试
    $display("\nTest Case 6: Minimum Value Test");
    activations = {8'h80, 8'h80, 8'h80, 8'h80, 8'h80, 8'h80, 8'h80, 8'h80}; // 最小负值
    weights = {8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF};     // 全1权重
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out: %h", weight_out);
    $display("Activation Out: %h %h %h %h %h %h %h %h", $signed(activation_out[63:56]),
             $signed(activation_out[55:48]), $signed(activation_out[47:40]),
             $signed(activation_out[39:32]), $signed(activation_out[31:24]),
             $signed(activation_out[23:16]), $signed(activation_out[15:8]),
             $signed(activation_out[7:0]));
    $display("Output Out: %d%d%d%d%d%d%d%d (Expected: -128 -128 -128 -128 -128 -128 -128 -128)",
             $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
             $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
             $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 测试用例7：混合符号测试
    $display("\nTest Case 7: Mixed Sign Test");
    activations = {8'h7F, 8'ha0, 8'h01, 8'hFF, 8'h7F, 8'ha0, 8'h01, 8'hFF}; // 混合正负值
    weights = {8'hFF, 8'hFF, 8'h00, 8'h00, 8'hFF, 8'hFF, 8'h00, 8'h00};     // 部分权重为1
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out: %h", weight_out);
    $display("Activation Out: %h %h %h %h %h %h %h %h", $signed(activation_out[63:56]),
             $signed(activation_out[55:48]), $signed(activation_out[47:40]),
             $signed(activation_out[39:32]), $signed(activation_out[31:24]),
             $signed(activation_out[23:16]), $signed(activation_out[15:8]),
             $signed(activation_out[7:0]));
    $display("Output Out: %d%d%d%d%d%d%d%d (Expected: 62 62 -62 -62 62 62 -62 -62)",
             $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
             $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
             $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 测试用例8：权重全零测试
    $display("\nTest Case 8: All Zero Weights");
    activations = {8'h01, 8'h02, 8'h03, 8'h04, 8'h05, 8'h06, 8'h07, 8'h08}; // 8x int8
    weights = {8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00};     // 全零权重
    #10;

    // 使能 output_en 并输出结果
    #10 output_en = 1;
    $display("Weight Out: %h", weight_out);
    $display("Activation Out: %h %h %h %h %h %h %h %h", $signed(activation_out[63:56]),
             $signed(activation_out[55:48]), $signed(activation_out[47:40]),
             $signed(activation_out[39:32]), $signed(activation_out[31:24]),
             $signed(activation_out[23:16]), $signed(activation_out[15:8]),
             $signed(activation_out[7:0]));
    $display("Output Out: %d%d%d%d%d%d%d%d (Expected: -36 -36 -36 -36 -36 -36 -36 -36)",
             $signed(output_out[63:56]), $signed(output_out[55:48]), $signed(output_out[47:40]),
             $signed(output_out[39:32]), $signed(output_out[31:24]), $signed(output_out[23:16]),
             $signed(output_out[15:8]), $signed(output_out[7:0]));
    output_en = 0;

    // 复位
    rst_n = 0;
    #10 rst_n = 1;

    // 结束仿真
    $finish;
  end

  // 波形记录
  initial begin
    $dumpfile("waveform.vcd");
    $dumpvars(0, tb_tile);
  end

endmodule
