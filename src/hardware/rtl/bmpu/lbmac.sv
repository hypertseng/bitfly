module lbmac #(
    parameter BIT_ACT    = 8,  // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1   // 单个bit-plane宽度；1/2/4bit通过prec_i分多拍处理
) (
  input  logic        [ 1:0] mode_i,       // 00:bipolar(1bit), 01:unsigned plane, 10:signed plane
    input  logic        [ 7:0] weights,      // 8-bit 权重
    input  logic        [63:0] activations,  // 64-bit 激活输入（8x int8）
    output logic signed [11:0] result        // 12-bit 有符号结果
);

  // ----------------------
  // 加权激活值计算
  // ----------------------
logic signed [8:0] weighted_activations[8];

always_comb begin
  for (int i = 0; i < 8; i++) begin
    automatic logic signed [7:0] act;
    automatic logic signed [8:0] neg;

    act = activations[((7 - i) * 8) +: 8];
    neg = -act;  //防止 -(-128) 溢出

    unique case (mode_i)
      2'b00: weighted_activations[i] = weights[i] ? act : neg;      // binary: +1 / -1
      2'b01: weighted_activations[i] = weights[i] ? act : 9'sd0;    // unsigned plane: +1 / 0
      2'b10: weighted_activations[i] = weights[i] ? neg : 9'sd0;    // signed plane: -1 / 0
      default: weighted_activations[i] = 9'sd0;
    endcase
  end
end
  // ----------------------
  // 树形累加结构
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
