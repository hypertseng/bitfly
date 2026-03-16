module pe #(
    parameter int unsigned BIT_ACT    = 8,
    parameter int unsigned BIT_WEIGHT = 1,
    parameter int unsigned CTX_MAX    = 8,
    localparam int unsigned CTX_IDX_W = (CTX_MAX <= 1) ? 1 : $clog2(CTX_MAX)
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  en,
    input  logic                  clear_i,
    input  logic                  ctx_clear_i,
    input  logic [CTX_IDX_W-1:0]  ctx_id_i,
    input  logic [CTX_IDX_W-1:0]  output_ctx_id_i,
    input  logic                  output_en,
    input  logic [63:0]           activations,
    input  logic [63:0]           weights,
    input  logic [2:0]            shift_amt_i,
    input  logic [1:0]            lbmac_mode_i,
    input  logic [127:0]          input_output_reg,
    output logic [63:0]           weight_out,
    output logic [63:0]           activation_out,
    output logic [127:0]          output_out,
    output logic [127:0]          output_selected_o
);

  logic [63:0] activation_reg;
  logic [63:0] weight_reg;
  logic signed [20:0] partial_sum_reg [CTX_MAX-1:0][8];
  logic [127:0]       output_reg      [CTX_MAX-1:0];
  logic signed [11:0] pe_outputs[8];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      activation_reg <= '0;
      weight_reg     <= '0;
      foreach (output_reg[c]) output_reg[c] <= '0;
      foreach (partial_sum_reg[c, i]) partial_sum_reg[c][i] <= '0;
    end else if (clear_i) begin
      activation_reg <= '0;
      weight_reg     <= '0;
      foreach (output_reg[c]) output_reg[c] <= '0;
      foreach (partial_sum_reg[c, i]) partial_sum_reg[c][i] <= '0;
    end else begin
      if (ctx_clear_i) begin
        output_reg[ctx_id_i] <= '0;
        foreach (partial_sum_reg[ctx_id_i, i]) partial_sum_reg[ctx_id_i][i] <= '0;
      end

      if (en) begin
        activation_reg <= activations;
        weight_reg     <= weights;

        foreach (partial_sum_reg[ctx_id_i, i]) begin
          automatic logic signed [20:0] accum_base;
          automatic logic signed [20:0] next_sum;
          automatic logic signed [15:0] local_sat;
          automatic logic signed [15:0] casc_in;
          automatic logic signed [16:0] casc_sum;

          accum_base = ctx_clear_i ? '0 : partial_sum_reg[ctx_id_i][i];
          next_sum   = accum_base + ($signed(pe_outputs[i]) <<< shift_amt_i);
          partial_sum_reg[ctx_id_i][i] <= next_sum;

          if (next_sum > 21'sd32767) begin
            local_sat = 16'sh7FFF;
          end else if (next_sum < -21'sd32768) begin
            local_sat = 16'sh8000;
          end else begin
            local_sat = next_sum[15:0];
          end

          casc_in = $signed(input_output_reg[i*16+:16]);
          casc_sum = output_en ? $signed(local_sat) : ($signed(local_sat) + casc_in);
          if (casc_sum > 17'sd32767) begin
            output_reg[ctx_id_i][i*16+:16] <= 16'sh7FFF;
          end else if (casc_sum < -17'sd32768) begin
            output_reg[ctx_id_i][i*16+:16] <= 16'sh8000;
          end else begin
            output_reg[ctx_id_i][i*16+:16] <= casc_sum[15:0];
          end
        end
      end
    end
  end

  assign weight_out         = weight_reg;
  assign activation_out     = activation_reg;
  assign output_out         = output_reg[ctx_id_i];
  assign output_selected_o  = output_reg[output_ctx_id_i];

  generate
    for (genvar i = 0; i < 8; i++) begin : pe_array
      lbmac #(
          .BIT_ACT   (BIT_ACT),
          .BIT_WEIGHT(BIT_WEIGHT)
      ) u_lbmac (
          .mode_i     (lbmac_mode_i),
          .weights    (weights[i*8+:8]),
          .activations(activations),
          .result     (pe_outputs[i])
      );
    end
  endgenerate
endmodule
