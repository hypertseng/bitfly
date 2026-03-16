module pe #(
    parameter BIT_ACT    = 8,   // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1    // 权值位宽（int1）
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         en,
    input  logic         clear_i,
    input  logic         output_en,         // 输出使能信号
    input  logic [ 63:0] activations,       // 64-bit 激活输入（8x int8）
    input  logic [ 63:0] weights,           // 64-bit 权重输入（8x int1）
    input  logic [ 2:0]  shift_amt_i,       // 当前 bit-plane 的位权移位
    input  logic [ 1:0]  lbmac_mode_i,      // lbmac 计算模式
    input  logic [127:0] input_output_reg,  // 来自其他 pe 的 output_reg 值
    output logic [ 63:0] weight_out,        // 权重寄存器输出
    output logic [ 63:0] activation_out,    // 激活寄存器输出
    output logic [127:0] output_out         // 输出寄存器输出（8x int8）
);

  //--- 寄存器定义 ---//
  logic [63:0] activation_reg;  // 激活寄存器
  logic [63:0] weight_reg;  // 权重寄存器（8x 8-bit）
  logic signed [20:0] partial_sum_reg[8];  // 累加寄存器（21-bit for multi-precision）
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
    end else if (clear_i) begin
      activation_reg <= '0;
      weight_reg     <= '0;
      output_reg     <= '0;
      foreach (partial_sum_reg[i]) partial_sum_reg[i] <= '0;
    end else begin
      if (en) begin
        // 模式2：正常计算流程
        activation_reg <= activations;  // 激活向右传递
        weight_reg     <= weights;  // 权重向下传递

        // 累加逻辑：对每个 lbmac 输出按对应的位权进行移位后累加
        foreach (partial_sum_reg[i]) begin
          automatic logic signed [20:0] next_sum;
          automatic logic signed [15:0] local_sat;
          automatic logic signed [15:0] casc_in;
          automatic logic signed [16:0] casc_sum;

          next_sum = partial_sum_reg[i] + ($signed(pe_outputs[i]) <<< shift_amt_i);
          partial_sum_reg[i] <= next_sum;

          if (next_sum > 21'sd32767) begin
            local_sat = 16'sh7FFF;  // 最大值 32767
          end else if (next_sum < -21'sd32768) begin
            local_sat = 16'sh8000;  // 最小值 -32768
          end else begin
            local_sat = next_sum[15:0];
          end

          casc_in = $signed(input_output_reg[i*16+:16]);
          // output_en=1 时输出本地结果；否则与右侧 PE 累加实现级联归并。
          casc_sum = output_en ? $signed(local_sat) : ($signed(local_sat) + casc_in);
          if (casc_sum > 17'sd32767) begin
            output_reg[i*16+:16] <= 16'sh7FFF;
          end else if (casc_sum < -17'sd32768) begin
            output_reg[i*16+:16] <= 16'sh8000;
          end else begin
            output_reg[i*16+:16] <= casc_sum[15:0];
          end
        end
      end
    end
  end

  //--- 输出信号直接连接寄存器 ---//
  assign weight_out     = weight_reg;  // 向下传递权重
  assign activation_out = activation_reg;  // 向右传递激活
  assign output_out     = output_reg;  // 向左传递输出

  //--- lbmac阵列实例化---//
  generate
    for (genvar i = 0; i < 8; i++) begin : pe_array
      lbmac #(
          .BIT_ACT   (BIT_ACT),
          .BIT_WEIGHT(BIT_WEIGHT)
      ) u_lbmac (
          .mode_i     (lbmac_mode_i),
          .weights    (weights[i*8+:8]),  // 第i个8-bit段
          .activations(activations),      // 所有PE共享激活输入
          .result     (pe_outputs[i])     // 12-bit输出
      );
    end
  endgenerate
endmodule
