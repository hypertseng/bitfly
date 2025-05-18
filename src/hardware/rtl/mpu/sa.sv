module sa import ara_pkg::*; import rvv_pkg::*; #(
    parameter ROWS       = 4,  // 阵列行数（可配置）
    parameter COLS       = 4,  // 阵列列数（可配置）
    parameter BIT_ACT    = 8,  // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1   // 权值位宽（int1）
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    // input  logic        en,                 // 全局使能
    input  logic        output_en,          // 输出使能（仅控制数据输出）
    // 输入数据接口（支持参数化类型 elen_t）
    input  elen_t [ROWS-1:0] act_in,        // 激活输入（每行一个 elen_t 数据）
    input  elen_t [COLS-1:0] weight_in,     // 权值输入（每列一个 elen_t 数据）
    input  logic [8:0]  k_dim,              // k 维度
    // 输出数据接口（支持参数化类型 elen_t）
    output elen_t [ROWS-1:0] output_data,   // 最终输出（每行一个 elen_t 数据）
    output logic        mpu_insn_done_o     // 计算完成信号
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
  // Tile 间信号连接
  // ----------------------
  elen_t act_reg    [ROWS-1:0][COLS-1:0];
  elen_t weight_reg [ROWS-1:0][COLS-1:0];
  elen_t output_reg [ROWS-1:0][COLS-1:0];

  // ----------------------
  // Cycle计数器与计算完成逻辑
  // ----------------------
  logic [15:0] cycle_cnt;
  logic [15:0] compute_cycles;

  always_comb begin
    compute_cycles = ROWS + (k_dim >> 3);  // k_dim除以8
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cycle_cnt <= 0;
    end else if (valid) begin
      if (cycle_cnt < compute_cycles)
        cycle_cnt <= cycle_cnt + 1;
    end else begin
      cycle_cnt <= 0;
    end
  end

  assign mpu_insn_done_o = (cycle_cnt >= compute_cycles);

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
            .en              (valid),
            .output_en       (output_en),
            .activations     ((j == 0) ? act_in[i] : act_reg[i][j-1]),
            .weights         ((i == 0) ? weight_in[j] : weight_reg[i-1][j]),
            .input_output_reg((j == COLS-1) ? '0 : output_reg[i][j+1]),
            .activation_out  (act_reg[i][j]),
            .weight_out      (weight_reg[i][j]),
            .output_out      (output_reg[i][j])
        );
      end
    end
  endgenerate

  // ----------------------
  // 输出捕获逻辑
  // ----------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      foreach (output_data[i])
        output_data[i] <= '0;
    end else if (output_en) begin
      foreach (output_data[i])
        output_data[i] <= output_reg[i][0];
    end
  end

endmodule