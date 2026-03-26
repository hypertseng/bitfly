module sa import ara_pkg::*; import rvv_pkg::*; #(
    parameter int unsigned ROWS       = 2,
    parameter int unsigned COLS       = 2,
    parameter int unsigned BIT_ACT    = 8,
    parameter int unsigned BIT_WEIGHT = 1,
    parameter int unsigned CTX_MAX    = 8,
    localparam int unsigned CTX_IDX_W = (CTX_MAX <= 1) ? 1 : $clog2(CTX_MAX),
    localparam int unsigned COL_IDX_W = (COLS <= 1) ? 1 : $clog2(COLS)
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,
    input  logic                   valid_i,
    input  logic                   clear_i,
    input  logic                   ctx_clear_i,
    input  logic [CTX_IDX_W-1:0]   ctx_id_i,
    input  logic [CTX_IDX_W-1:0]   output_ctx_id_i,
    input  logic [COL_IDX_W-1:0]   output_col_id_i,
    input  logic                   output_en_i,
    input  elen_t [ROWS-1:0]       bmpu_act_operand_i,
    input  logic [ROWS-1:0]        bmpu_act_operand_valid_i,
    input  elen_t [COLS-1:0]       bmpu_wgt_operand_i,
    input  logic [COLS-1:0]        bmpu_wgt_operand_valid_i,
    input  logic [16:0]            k_dim_i,
    input  logic [2:0]             prec_i,
    output logic [127:0]           output_data_o [ROWS-1:0],
    output logic [ROWS-1:0]        bmpu_act_operand_ready_o,
    output logic [COLS-1:0]        bmpu_wgt_operand_ready_o,
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
  logic [15:0] k_iters;
  logic [ROWS-1:0] act_window;
  logic [COLS-1:0] wgt_window;
  logic [ROWS-1:0] act_window_eff;
  logic [COLS-1:0] wgt_window_eff;
  elen_t [ROWS-1:0] act_in;
  elen_t [COLS-1:0] wgt_in;
  logic             sa_stage_ready;
  logic             sa_step;
  logic             act_consume_step;
  logic             store_mode;
`ifndef SYNTHESIS
  logic             dbg_valid_q;
  logic             dbg_step_q;
`endif

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
    k_iters        = (k_dim_i / BIT_ACT);
    compute_cycles = ROWS - 1 + k_iters * planes + COLS;
    compute_last   = (compute_cycles == 0) ? 16'd0 : (compute_cycles - 16'd1);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cycle_cnt <= '0;
    end else if (clear_i) begin
      cycle_cnt <= '0;
    end else if (sa_step) begin
      if (cycle_cnt < compute_last) cycle_cnt <= cycle_cnt + 1'b1;
      else cycle_cnt <= '0;
    end else if (!valid_i) begin
      cycle_cnt <= '0;
    end
  end

  assign sa_done_o = sa_step && (cycle_cnt == compute_last);

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
    store_mode = valid_i && output_en_i;
    act_in = '0;
    wgt_in = '0;
    act_window = '0;
    wgt_window = '0;

    for (int i = 0; i < ROWS; i++) begin
      if ((cycle_cnt >= i) && (cycle_cnt < (i + k_iters * planes))) begin
        act_window[i] = 1'b1;
        act_in[i] = bmpu_act_operand_i[i];
      end
    end

    for (int j = 0; j < COLS; j++) begin
      if ((cycle_cnt >= j) && (cycle_cnt < (j + k_iters * planes))) begin
        wgt_window[j] = 1'b1;
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

  always_comb begin
    automatic logic act_suffix_ok;
    automatic logic wgt_suffix_ok;
    automatic logic [ROWS-1:0] act_suffix_mask;
    automatic logic [COLS-1:0] wgt_suffix_mask;
    automatic logic [ROWS-1:0] act_prefix_mask;
    automatic logic [COLS-1:0] wgt_prefix_mask;

    act_suffix_ok   = store_mode;
    wgt_suffix_ok   = store_mode;
    act_window_eff  = '0;
    wgt_window_eff  = '0;

    if (!store_mode) begin
      for (int cut = 0; cut <= ROWS; cut++) begin
        act_prefix_mask = '0;
        for (int i = 0; i < cut; i++) begin
          act_prefix_mask[i] = 1'b1;
        end
        act_suffix_mask = act_window & ~act_prefix_mask;
        if ((bmpu_act_operand_valid_i & act_window) == act_suffix_mask) begin
          act_suffix_ok  = 1'b1;
          act_window_eff = act_suffix_mask;
        end
      end

      for (int cut = 0; cut <= COLS; cut++) begin
        wgt_prefix_mask = '0;
        for (int j = 0; j < cut; j++) begin
          wgt_prefix_mask[j] = 1'b1;
        end
        wgt_suffix_mask = wgt_window & ~wgt_prefix_mask;
        if ((bmpu_wgt_operand_valid_i & wgt_window) == wgt_suffix_mask) begin
          wgt_suffix_ok  = 1'b1;
          wgt_window_eff = wgt_suffix_mask;
        end
      end
    end

    sa_stage_ready = act_suffix_ok && wgt_suffix_ok;
    sa_step = store_mode ? 1'b0 : (valid_i && sa_stage_ready);
`ifndef SYNTHESIS
    if (valid_i && !store_mode && !sa_step) begin
      $display("[%0t][SA_STALL] cycle=%0d eff=%0d act_win=%b act_eff=%b wgt_win=%b wgt_eff=%b act_v=%b wgt_v=%b ctx=%0d out_ctx=%0d out_col=%0d",
               $time, cycle_cnt, cycle_eff, act_window, act_window_eff, wgt_window, wgt_window_eff, bmpu_act_operand_valid_i, bmpu_wgt_operand_valid_i,
               ctx_id_i, output_ctx_id_i, output_col_id_i);
    end
`endif
    act_consume_step = sa_step;
    if (in_k_stage && (planes > 3'd1)) begin
      act_consume_step = sa_step && (plane_idx == (planes - 3'd1));
    end
    bmpu_act_operand_ready_o = '0;
    bmpu_wgt_operand_ready_o = '0;
    if (!store_mode) begin
      bmpu_act_operand_ready_o = act_window_eff & {ROWS{act_consume_step}};
      bmpu_wgt_operand_ready_o = wgt_window_eff & {COLS{sa_step}};
    end
  end

  generate
    for (genvar i = 0; i < ROWS; i++) begin : row_gen
      for (genvar j = 0; j < COLS; j++) begin : col_gen
        logic        pe_mac_en;
        logic [2:0]  pe_plane_idx;
        logic [2:0]  pe_shift_amt;
        logic [1:0]  pe_lbmac_mode;

        always_comb begin
          pe_mac_en    = sa_step
                      && (cycle_cnt >= (i + j))
                      && (cycle_cnt < ((i + j) + k_iters * planes));
          pe_plane_idx = '0;
          if (pe_mac_en) begin
            pe_plane_idx = (cycle_cnt - (i + j)) % planes;
          end

          if (!pe_mac_en) begin
            pe_shift_amt  = '0;
            pe_lbmac_mode = 2'b00;
          end else begin
            unique case (prec_i)
              3'd0: begin
                pe_shift_amt  = '0;
                pe_lbmac_mode = 2'b00;
              end
              3'd1: begin
                pe_shift_amt  = '0;
                pe_lbmac_mode = (pe_plane_idx == 3'd0) ? 2'b01 : 2'b10;
              end
              3'd2: begin
                pe_shift_amt  = pe_plane_idx;
                pe_lbmac_mode = (pe_plane_idx == 3'd1) ? 2'b10 : 2'b01;
              end
              3'd3: begin
                pe_shift_amt  = pe_plane_idx;
                pe_lbmac_mode = (pe_plane_idx == 3'd3) ? 2'b10 : 2'b01;
              end
              default: begin
                pe_shift_amt  = '0;
                pe_lbmac_mode = 2'b00;
              end
            endcase
          end
        end

        pe #(
            .BIT_ACT   (BIT_ACT),
            .BIT_WEIGHT(BIT_WEIGHT),
            .CTX_MAX   (CTX_MAX)
        ) u_pe (
            .clk                      (clk_i),
            .rst_n                    (rst_ni),
            .en                       (sa_step),
            .mac_en_i                 (pe_mac_en),
            .clear_i                  (clear_i),
            .ctx_clear_i              (ctx_clear_i),
            .ctx_id_i                 (ctx_id_i),
            .output_ctx_id_i          (output_ctx_id_i),
            .output_en                (output_en_i),
            .activations              ((j == 0) ? act_in[i] : act_reg[i][j-1]),
            .weights                  ((i == 0) ? wgt_in[j] : weight_reg[i-1][j]),
            .shift_amt_i              (pe_shift_amt),
            .lbmac_mode_i             (pe_lbmac_mode),
            .input_output_compute_reg ((j == COLS-1) ? '0 : output_reg_compute[i][j+1]),
            .act_hold_i               ((j == 0) ? (act_window[i] && !bmpu_act_operand_valid_i[i]) : 1'b0),
            .wgt_hold_i               ((i == 0) ? (wgt_window[j] && !bmpu_wgt_operand_valid_i[j]) : 1'b0),
            .activation_out           (act_reg[i][j]),
            .weight_out               (weight_reg[i][j]),
            .output_out               (output_reg_compute[i][j]),
            .output_selected_o        (output_reg_selected[i][j])
        );
      end
    end
  endgenerate

  always_comb begin
    output_data_o = '{default:'0};
    for (int i = 0; i < ROWS; i++) begin
      output_data_o[i] = output_reg_selected[i][output_col_id_i];
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      dbg_valid_q <= 1'b0;
      dbg_step_q  <= 1'b0;
    end else begin
      if ((valid_i != dbg_valid_q) || (sa_step != dbg_step_q) || sa_done_o) begin
        $display("[%0t][SA_DBG] valid=%0b step=%0b done=%0b cyc=%0d/%0d ctx=%0d out_ctx=%0d out_col=%0d act_v=%b act_r=%b wgt_v=%b wgt_r=%b act_win=%b wgt_win=%b",
                 $time, valid_i, sa_step, sa_done_o, cycle_cnt, compute_last,
                 ctx_id_i, output_ctx_id_i, output_col_id_i,
                 bmpu_act_operand_valid_i, bmpu_act_operand_ready_o,
                 bmpu_wgt_operand_valid_i, bmpu_wgt_operand_ready_o,
                 act_window_eff, wgt_window_eff);
      end
      dbg_valid_q <= valid_i;
      dbg_step_q  <= sa_step;
    end
  end
`endif

endmodule
