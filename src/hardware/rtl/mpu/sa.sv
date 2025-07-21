module sa import ara_pkg::*; import rvv_pkg::*; #(
    parameter ROWS       = 4,  // 阵列行数（可配置）
    parameter COLS       = 4,  // 阵列列数（可配置）
    parameter BIT_ACT    = 8,  // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1   // 权值位宽（int1）
) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        valid_i,
    // input  logic        en,                 // 全局使能
    input  logic        output_en_i,          // 输出使能（仅控制数据输出）
    // 输入数据接口（支持参数化类型 elen_t）
    input  elen_t [ROWS-1:0] mpu_act_operand_i,        // 激活输入（每行一个 elen_t 数据）
    input  elen_t [COLS-1:0] mpu_wgt_operand_i,     // 权值输入（每列一个 elen_t 数据）
    input  logic [8:0]  k_dim_i,              // k 维度
    // 输出数据接口（支持参数化类型 elen_t）
    output logic [127:0] output_data_o [ROWS-1:0],   // 最终输出（每行一个 elen_t 数据）
    output logic        sa_done_o     // 计算完成信号
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
  logic [127:0] output_reg [ROWS-1:0][COLS-1:0];

  // ----------------------
  // Cycle计数器与计算完成逻辑
  // ----------------------
  logic [15:0] cycle_cnt;
  logic [15:0] compute_cycles;

  always_comb begin
    compute_cycles = ROWS - 1 + (k_dim_i / BIT_ACT) + COLS;  // k_dim除以8
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cycle_cnt <= 0;
    end else if (valid_i) begin
      if (cycle_cnt < compute_cycles) begin
        cycle_cnt <= cycle_cnt + 1;
      end else begin
        cycle_cnt <= 1'b0;
      end
    end else begin
      cycle_cnt <= 0;
    end
  end

  assign sa_done_o = (cycle_cnt == compute_cycles);

  // ----------------------
  // 脉动阵列生成
  // ----------------------

  // 处理流水输入导致的x
  elen_t [ROWS-1:0] act_in;
  elen_t [COLS-1:0] wgt_in;
  always_comb begin
    act_in = '0;
    wgt_in = '0;
    
    for (int i = 0; i < ROWS; i++) begin
      if ((i <= cycle_cnt) && (i + (k_dim_i >> 3)) > cycle_cnt) act_in[i] = mpu_act_operand_i[i];
    end
    
    for (int j = 0; j < COLS; j++) begin
      if ((j <= cycle_cnt) && (j + (k_dim_i >> 3)) > cycle_cnt) wgt_in[j] = mpu_wgt_operand_i[j];
    end
  end

  // logic output_en_d, output_en_q;

  generate
    for (genvar i = 0; i < ROWS; i++) begin : row_gen
      for (genvar j = 0; j < COLS; j++) begin : col_gen
        tile #(
            .BIT_ACT   (BIT_ACT),
            .BIT_WEIGHT(BIT_WEIGHT)
        ) u_tile (
            .clk             (clk_i),
            .rst_n           (rst_ni),
            .en              (valid_i),
            .output_en       (output_en_i),
            .activations     ((j == 0) ? act_in[i] : act_reg[i][j-1]),
            .weights         ((i == 0) ? wgt_in[j] : weight_reg[i-1][j]),
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

  always_comb begin
    output_data_o = '{default:'0};  // 初始化输出数据为0
    // 仅在输出使能时更新输出数据
    if (output_en_i) begin
      for (int i = 0; i < ROWS; i++) begin
        output_data_o[i] = output_reg[i][0];
      end
    end
  end

endmodule