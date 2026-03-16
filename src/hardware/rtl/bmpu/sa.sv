module sa import ara_pkg::*; import rvv_pkg::*; #(
    parameter ROWS       = 2,  // 阵列行数
    parameter COLS       = 2,  // 阵列列数
    parameter BIT_ACT    = 8,  // 激活值位宽（int8）
    parameter BIT_WEIGHT = 1   // 权值位宽（int1）
) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        valid_i,
  input  logic        clear_i,
    // input  logic        en,                 // 全局使能
    input  logic        output_en_i,          // 输出使能
    // 输入数据接口（支持参数化类型 elen_t）
    input  elen_t [ROWS-1:0] bmpu_act_operand_i,        // 激活输入（每行一个 elen_t 数据）
    input  elen_t [COLS-1:0] bmpu_wgt_operand_i,     // 权值输入（每列一个 elen_t 数据）
    input  logic [16:0]  k_dim_i,              // k 维度
    input  logic [2:0]   prec_i,               // 精度
    // 输出数据接口
    output logic [127:0] output_data_o [ROWS-1:0],   // 最终输出（每行一个 elen_t 数据）
    output logic        sa_done_o     // 计算完成信号
);

  // ----------------------
  // 参数校验
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
  // pe 间信号连接
  // ----------------------
  elen_t act_reg    [ROWS-1:0][COLS-1:0];
  elen_t weight_reg [ROWS-1:0][COLS-1:0];
  logic [127:0] output_reg [ROWS-1:0][COLS-1:0];

  // ----------------------
  // Cycle计数器与计算完成逻辑
  // ----------------------
  logic [15:0] cycle_cnt;
  logic [15:0] compute_cycles;
  logic [15:0] compute_last;
  logic [2:0]  planes;
  logic [15:0] cycle_eff;
  logic [2:0]  plane_idx;
  logic        in_k_stage;
  logic [2:0]  shift_amt_sa;
  logic [1:0]  lbmac_mode_sa;

  // 根据 prec_i 决定当前指令使用的 bit-plane 数量
  // 0: 1bit(binary), 1: ternary, 2: int2, 3: int4
  // 其他编码退化为 1bit
  always_comb begin
    unique case (prec_i)
      3'd0: planes = 3'd1;
      3'd1: planes = 3'd2;
      3'd2: planes = 3'd2;
      3'd3: planes = 3'd4;
      default: planes = 3'd1;
    endcase
  end

  always_comb begin
    // k_dim_i / BIT_ACT 是原始的 K-slice 数（每个 slice 消费 8 个激活）
    // 多精度时，同一个 K-slice 会串行处理多个 bit-plane，因此在中间
    // K 相关的阶段乘以 planes。
    compute_cycles = ROWS - 1 + (k_dim_i / BIT_ACT) * planes + COLS;
    compute_last   = (compute_cycles == 0) ? 16'd0 : (compute_cycles - 16'd1);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cycle_cnt <= 0;
    end else if (clear_i) begin
      cycle_cnt <= 0;
    end else if (valid_i) begin
      if (cycle_cnt < compute_last) begin
        cycle_cnt <= cycle_cnt + 1;
      end else begin
        cycle_cnt <= 1'b0;
      end
    end else begin
      cycle_cnt <= 0;
    end
  end

  assign sa_done_o = valid_i && (cycle_cnt == compute_last);

  // 为了在不改变原有空间波动行为的前提下插入 bit-plane 的时间复用，
  // 我们将当前 cycle_cnt 映射到一个“等效”的 cycle_eff：
  // - 在前 ROWS-1 个周期（填充阶段）保持不变；
  // - 在中间 K-slice 阶段，将扩展后的 planes*Nslices 压缩回 Nslices；
  // - 在最后 COLS 阶段，同步平移保持顺序。
  always_comb begin
    if (cycle_cnt < ROWS - 1) begin
      cycle_eff = cycle_cnt;
    end else if (cycle_cnt < ROWS - 1 + (k_dim_i / BIT_ACT) * planes) begin
      // 中间 K 相关阶段：去掉前导 ROWS-1，再按 planes 压缩
      cycle_eff = (cycle_cnt - (ROWS - 1)) / planes + (ROWS - 1);
    end else begin
      // 尾部 COLS 阶段：保持相对偏移，保证整体长度一致
      cycle_eff = (k_dim_i / BIT_ACT) + (ROWS - 1)
                + (cycle_cnt - (ROWS - 1 + (k_dim_i / BIT_ACT) * planes));
    end
  end

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
      if ((i <= cycle_eff) && (i + (k_dim_i >> 3)) > cycle_eff) begin
        act_in[i] = bmpu_act_operand_i[i];
      end
    end

    for (int j = 0; j < COLS; j++) begin
      if ((j <= cycle_eff) && (j + (k_dim_i >> 3)) > cycle_eff) begin
        wgt_in[j] = bmpu_wgt_operand_i[j];
      end
    end
  end

  // ----------------------
  // Bit-plane shift and lbmac mode generation
  // ----------------------
  always_comb begin
    // 判断当前是否在 K 相关阶段
    in_k_stage = (cycle_cnt >= (ROWS - 1))
                && (cycle_cnt < (ROWS - 1 + (k_dim_i / BIT_ACT) * planes));

    if (in_k_stage) begin
      plane_idx = (cycle_cnt - (ROWS - 1)) % planes;
    end else begin
      plane_idx = '0;
    end

    // lbmac mode: 00=binary, 01=unsigned plane(+1/0), 10=signed plane(-1/0)
    if (!in_k_stage) begin
      shift_amt_sa = '0;
      lbmac_mode_sa = 2'b00;
    end else begin
      unique case (prec_i)
        3'd0: begin
          // binary: 单拍，+1/-1
          shift_amt_sa = '0;
          lbmac_mode_sa = 2'b00;
        end
        3'd1: begin
          // ternary {-1,0,+1}:
          // plane0: +1/0, plane1: -1/0，二者都不移位
          shift_amt_sa = '0;
          lbmac_mode_sa = (plane_idx == 3'd0) ? 2'b01 : 2'b10;
        end
        3'd2: begin
          // int2 (2's complement): low plane <<0, sign plane <<1
          shift_amt_sa = plane_idx;
          lbmac_mode_sa = (plane_idx == 3'd1) ? 2'b10 : 2'b01;
        end
        3'd3: begin
          // int4 (2's complement): plane0..2 unsigned, plane3 signed
          shift_amt_sa = plane_idx;
          lbmac_mode_sa = (plane_idx == 3'd3) ? 2'b10 : 2'b01;
        end
        default: begin
          shift_amt_sa = '0;
          lbmac_mode_sa = 2'b00;
        end
      endcase
    end
  end

  // logic output_en_d, output_en_q;

  generate
    for (genvar i = 0; i < ROWS; i++) begin : row_gen
      for (genvar j = 0; j < COLS; j++) begin : col_gen
        pe #(
            .BIT_ACT   (BIT_ACT),
            .BIT_WEIGHT(BIT_WEIGHT)
        ) u_pe (
            .clk             (clk_i),
            .rst_n           (rst_ni),
            .en              (valid_i),
          .clear_i         (clear_i),
            .output_en       (output_en_i),
            .activations     ((j == 0) ? act_in[i] : act_reg[i][j-1]),
            .weights         ((i == 0) ? wgt_in[j] : weight_reg[i-1][j]),
            .shift_amt_i     (shift_amt_sa),
            .lbmac_mode_i    (lbmac_mode_sa),
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
    for (int i = 0; i < ROWS; i++) begin
      output_data_o[i] = output_reg[i][0];
    end
  end

endmodule
