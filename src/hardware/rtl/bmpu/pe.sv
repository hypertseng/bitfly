module pe #(
    parameter int unsigned BIT_ACT    = 8,
    parameter int unsigned BIT_WEIGHT = 1,
    parameter int unsigned CTX_MAX    = 8,
    localparam int unsigned CTX_IDX_W = (CTX_MAX <= 1) ? 1 : $clog2(CTX_MAX)
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  en,
    input  logic                  mac_en_i,
    input  logic                  clear_i,
    input  logic                  ctx_clear_i,
    input  logic [CTX_IDX_W-1:0]  ctx_id_i,
    input  logic [CTX_IDX_W-1:0]  output_ctx_id_i,
    input  logic                  output_en,
    input  logic [63:0]           activations,
    input  logic [63:0]           weights,
    input  logic [2:0]            shift_amt_i,
    input  logic [1:0]            lbmac_mode_i,
    input  logic [127:0]          input_output_compute_reg,
    input  logic                  act_hold_i,
    input  logic                  wgt_hold_i,
    output logic [63:0]           weight_out,
    output logic [63:0]           activation_out,
    output logic [127:0]          output_out,
    output logic [127:0]          output_selected_o
);

  logic [63:0] activation_reg;
  logic [63:0] weight_reg;
  logic [63:0] activation_compute;
  logic [63:0] weight_compute;
  logic signed [20:0] partial_sum_reg [CTX_MAX-1:0][8];
  logic [127:0]       output_reg_compute;
  logic signed [11:0] pe_outputs[8];

  function automatic logic signed [15:0] sat21_to_s16(input logic signed [20:0] value);
    if (value > 21'sd32767) begin
      sat21_to_s16 = 16'sh7FFF;
    end else if (value < -21'sd32768) begin
      sat21_to_s16 = 16'sh8000;
    end else begin
      sat21_to_s16 = value[15:0];
    end
  endfunction

  function automatic logic signed [15:0] sat17_to_s16(input logic signed [16:0] value);
    if (value > 17'sd32767) begin
      sat17_to_s16 = 16'sh7FFF;
    end else if (value < -17'sd32768) begin
      sat17_to_s16 = 16'sh8000;
    end else begin
      sat17_to_s16 = value[15:0];
    end
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      activation_reg      <= '0;
      weight_reg          <= '0;
      output_reg_compute  <= '0;
      foreach (partial_sum_reg[c, i]) partial_sum_reg[c][i] <= '0;
    end else if (clear_i) begin
      activation_reg      <= '0;
      weight_reg          <= '0;
      output_reg_compute  <= '0;
      foreach (partial_sum_reg[c, i]) partial_sum_reg[c][i] <= '0;
    end else begin
      if (ctx_clear_i) begin
        for (int i = 0; i < 8; i++) begin
          partial_sum_reg[ctx_id_i][i] <= '0;
        end
      end

      if (en) begin
        // Keep the last propagated operand once this PE has finished its local
        // MACs; downstream neighbors may still need that value for one more hop.
        if (mac_en_i && !act_hold_i) activation_reg <= activations;
        if (mac_en_i && !wgt_hold_i) weight_reg     <= weights;

        if (mac_en_i) begin
          for (int i = 0; i < 8; i++) begin
            automatic logic signed [20:0] accum_base;
            automatic logic signed [20:0] next_sum;
            automatic logic signed [15:0] local_sat;

            accum_base = ctx_clear_i ? '0 : partial_sum_reg[ctx_id_i][i];
            next_sum   = accum_base + ($signed(pe_outputs[i]) <<< shift_amt_i);
            local_sat  = sat21_to_s16(next_sum);

            partial_sum_reg[ctx_id_i][i] <= next_sum;
            output_reg_compute[i*16+:16] <= local_sat;
          end
        end
      end
    end
  end

  always_comb begin
    output_selected_o = '0;
    for (int i = 0; i < 8; i++) begin
      output_selected_o[i*16+:16] = sat21_to_s16(partial_sum_reg[output_ctx_id_i][i]);
    end
  end

  assign activation_compute = act_hold_i ? activation_reg : activations;
  assign weight_compute     = wgt_hold_i ? weight_reg : weights;

  assign weight_out      = weight_reg;
  assign activation_out  = activation_reg;
  assign output_out      = output_reg_compute;

  generate
    for (genvar i = 0; i < 8; i++) begin : pe_array
      lbmac #(
          .BIT_ACT   (BIT_ACT),
          .BIT_WEIGHT(BIT_WEIGHT)
      ) u_lbmac (
          .mode_i     (lbmac_mode_i),
          .weights    (weight_compute[i*8+:8]),
          .activations(activation_compute),
          .result     (pe_outputs[i])
      );
    end
  endgenerate
endmodule
