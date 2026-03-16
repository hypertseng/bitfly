module sa import ara_pkg::*; import rvv_pkg::*; #(
    parameter int unsigned ROWS       = 2,
    parameter int unsigned COLS       = 2,
    parameter int unsigned BIT_ACT    = 8,
    parameter int unsigned BIT_WEIGHT = 1,
    parameter int unsigned CTX_MAX    = 8,
    localparam int unsigned CTX_IDX_W = (CTX_MAX <= 1) ? 1 : $clog2(CTX_MAX)
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,
    input  logic                   valid_i,
    input  logic                   clear_i,
    input  logic                   ctx_clear_i,
    input  logic [CTX_IDX_W-1:0]   ctx_id_i,
    input  logic [CTX_IDX_W-1:0]   output_ctx_id_i,
    input  logic                   output_en_i,
    input  elen_t [ROWS-1:0]       bmpu_act_operand_i,
    input  elen_t [COLS-1:0]       bmpu_wgt_operand_i,
    input  logic [16:0]            k_dim_i,
    input  logic [2:0]             prec_i,
    output logic [127:0]           output_data_o [ROWS-1:0],
    output logic                   sa_done_o
);

  initial begin
    assert (ROWS > 0 && COLS > 0) else $error("ROWS/COLS must be > 0");
    assert (BIT_ACT == 8) else $error("BIT_ACT must be 8");
    assert (BIT_WEIGHT == 1) else $error("BIT_WEIGHT is per-plane width and must be 1");
  end

  elen_t act_reg [ROWS-1:0][COLS-1:0];
  elen_t weight_reg [ROWS-1:0][COLS-1:0];
  logic [127:0] output_reg_compute [ROWS-1:0][COLS-1:0];
  logic [127:0] output_reg_selected [ROWS-1:0][COLS-1:0];

  logic [15:0] cycle_cnt;
  logic [15:0] compute_cycles;
  logic [15:0] compute_last;
  logic [2:0]  planes;
  logic [15:0] cycle_eff;
  logic [2:0]  plane_idx;
  logic        in_k_stage;
  logic [2:0]  shift_amt_sa;
  logic [1:0]  lbmac_mode_sa;
  elen_t [ROWS-1:0] act_in;
  elen_t [COLS-1:0] wgt_in;

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
    compute_cycles = ROWS - 1 + (k_dim_i / BIT_ACT) * planes + COLS;
    compute_last   = (compute_cycles == 0) ? 16'd0 : (compute_cycles - 16'd1);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cycle_cnt <= '0;
    end else if (clear_i || ctx_clear_i) begin
      cycle_cnt <= '0;
    end else if (valid_i) begin
      if (cycle_cnt < compute_last) cycle_cnt <= cycle_cnt + 1'b1;
      else cycle_cnt <= '0;
    end else begin
      cycle_cnt <= '0;
    end
  end

  assign sa_done_o = valid_i && (cycle_cnt == compute_last);

  always_comb begin
    if (cycle_cnt < ROWS - 1) begin
      cycle_eff = cycle_cnt;
    end else if (cycle_cnt < ROWS - 1 + (k_dim_i / BIT_ACT) * planes) begin
      cycle_eff = (cycle_cnt - (ROWS - 1)) / planes + (ROWS - 1);
    end else begin
      cycle_eff = (k_dim_i / BIT_ACT) + (ROWS - 1)
                + (cycle_cnt - (ROWS - 1 + (k_dim_i / BIT_ACT) * planes));
    end
  end

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

  always_comb begin
    in_k_stage = (cycle_cnt >= (ROWS - 1))
              && (cycle_cnt < (ROWS - 1 + (k_dim_i / BIT_ACT) * planes));

    if (in_k_stage) plane_idx = (cycle_cnt - (ROWS - 1)) % planes;
    else plane_idx = '0;

    if (!in_k_stage) begin
      shift_amt_sa  = '0;
      lbmac_mode_sa = 2'b00;
    end else begin
      unique case (prec_i)
        3'd0: begin
          shift_amt_sa  = '0;
          lbmac_mode_sa = 2'b00;
        end
        3'd1: begin
          shift_amt_sa  = '0;
          lbmac_mode_sa = (plane_idx == 3'd0) ? 2'b01 : 2'b10;
        end
        3'd2: begin
          shift_amt_sa  = plane_idx;
          lbmac_mode_sa = (plane_idx == 3'd1) ? 2'b10 : 2'b01;
        end
        3'd3: begin
          shift_amt_sa  = plane_idx;
          lbmac_mode_sa = (plane_idx == 3'd3) ? 2'b10 : 2'b01;
        end
        default: begin
          shift_amt_sa  = '0;
          lbmac_mode_sa = 2'b00;
        end
      endcase
    end
  end

  generate
    for (genvar i = 0; i < ROWS; i++) begin : row_gen
      for (genvar j = 0; j < COLS; j++) begin : col_gen
        pe #(
            .BIT_ACT   (BIT_ACT),
            .BIT_WEIGHT(BIT_WEIGHT),
            .CTX_MAX   (CTX_MAX)
        ) u_pe (
            .clk              (clk_i),
            .rst_n            (rst_ni),
            .en               (valid_i),
            .clear_i          (clear_i),
            .ctx_clear_i      (ctx_clear_i),
            .ctx_id_i         (ctx_id_i),
            .output_ctx_id_i  (output_ctx_id_i),
            .output_en        (output_en_i),
            .activations      ((j == 0) ? act_in[i] : act_reg[i][j-1]),
            .weights          ((i == 0) ? wgt_in[j] : weight_reg[i-1][j]),
            .shift_amt_i      (shift_amt_sa),
            .lbmac_mode_i     (lbmac_mode_sa),
            .input_output_reg ((j == COLS-1) ? '0 : output_reg_compute[i][j+1]),
            .activation_out   (act_reg[i][j]),
            .weight_out       (weight_reg[i][j]),
            .output_out       (output_reg_compute[i][j]),
            .output_selected_o(output_reg_selected[i][j])
        );
      end
    end
  endgenerate

  always_comb begin
    output_data_o = '{default:'0};
    for (int i = 0; i < ROWS; i++) begin
      output_data_o[i] = output_reg_selected[i][0];
    end
  end
endmodule
