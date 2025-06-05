module tile #(
    parameter BIT_ACT    = 8,   // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1    // 权值位宽（int1）
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         en,
    input  logic         output_en,         // 输出使能信号
    input  logic [ 63:0] activations,       // 64-bit 激活输入（8x int8）
    input  logic [ 63:0] weights,           // 64-bit 权重输入（8x int1）
    input  logic [127:0] input_output_reg,  // 来自其他 tile 的 output_reg 值
    output logic [ 63:0] weight_out,        // 权重寄存器输出
    output logic [ 63:0] activation_out,    // 激活寄存器输出
    output logic [127:0] output_out         // 输出寄存器输出（8x int8）
);

  //--- 寄存器定义 ---//
  logic [63:0] activation_reg;  // 激活寄存器
  logic [63:0] weight_reg;  // 权重寄存器（8x 8-bit）
  logic signed [16:0] partial_sum_reg[8];  // 累加寄存器（17-bit）
  logic [127:0] output_reg;  // 输出寄存器（8x int16）

  //--- PE输出接口 ---//
  logic signed [11:0] pe_outputs[8];  // PE输出（12-bit）

  //--- 寄存器更新逻辑 ---//
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      activation_reg <= '0;
      weight_reg     <= '0;
      output_reg     <= '0;
      foreach (partial_sum_reg[i]) partial_sum_reg[i] <= '0;
    end else begin
      if (output_en) begin
        // 模式1：写入其他tile的输出值
        output_reg <= input_output_reg;
      end else if (en) begin
        // 模式2：正常计算流程
        activation_reg <= activations;  // 激活向右传递
        weight_reg     <= weights;  // 权重向下传递

        // 累加逻辑：PE输出与当前累加值相加
        foreach (partial_sum_reg[i]) begin
          partial_sum_reg[i] <= partial_sum_reg[i] + pe_outputs[i];
        end

        // 饱和处理：必须在时序逻辑中更新输出寄存器！
        foreach (partial_sum_reg[i]) begin
          logic signed [15:0] saturated;
          automatic logic signed [16:0] current_sum = partial_sum_reg[i];

          // 比较时使用 17-bit 的阈值
          if (current_sum > 17'sd32767) begin
            saturated = 16'sh7FFF;  // 最大值 32767
          end else if (current_sum < -17'sd32768) begin
            saturated = 16'sh8000;  // 最小值 -32768
          end else begin
            saturated = current_sum[15:0];  // 直接截断低16位
          end

          output_reg[i*16+:16] <= saturated;
        end
      end
    end
  end

  //--- 输出信号直接连接寄存器 ---//
  assign weight_out     = weight_reg;  // 向下传递权重
  assign activation_out = activation_reg;  // 向右传递激活
  assign output_out     = output_reg;  // 向左传递输出

  //--- PE阵列实例化（关键修复：权重索引对齐）---//
  generate
    for (genvar i = 0; i < 8; i++) begin : pe_array
      pe #(
          .BIT_ACT   (BIT_ACT),
          .BIT_WEIGHT(BIT_WEIGHT)
      ) u_pe (
          .weights    (weights[i*8+:8]),  // 第i个8-bit段
          .activations(activations),      // 所有PE共享激活输入
          .result     (pe_outputs[i])     // 12-bit输出
      );
    end
  endgenerate
endmodule
