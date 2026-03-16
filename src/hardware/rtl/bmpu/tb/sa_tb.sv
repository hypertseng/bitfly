`timescale 1ns/1ps

package ara_pkg;
  typedef logic [63:0] elen_t;
endpackage

package rvv_pkg;
endpackage

module sa_tb;
  import ara_pkg::*;
  import rvv_pkg::*;

  localparam int ROWS = 2;
  localparam int COLS = 2;

  logic clk_i;
  logic rst_ni;
  logic valid_i;
  logic output_en_i;
  logic [ROWS-1:0][63:0] bmpu_act_operand_i;
  logic [COLS-1:0][63:0] bmpu_wgt_operand_i;
  logic [16:0] k_dim_i;
  logic [2:0] prec_i;
  logic [127:0] output_data_o [ROWS-1:0];
  logic sa_done_o;

  sa #(
    .ROWS(ROWS),
    .COLS(COLS),
    .BIT_ACT(8),
    .BIT_WEIGHT(1)
  ) dut (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .valid_i(valid_i),
    .output_en_i(output_en_i),
    .bmpu_act_operand_i(bmpu_act_operand_i),
    .bmpu_wgt_operand_i(bmpu_wgt_operand_i),
    .k_dim_i(k_dim_i),
    .prec_i(prec_i),
    .output_data_o(output_data_o),
    .sa_done_o(sa_done_o)
  );

  always #5 clk_i = ~clk_i;

  function automatic int expected_cycles(input logic [2:0] p, input int kdim);
    int planes;
    begin
      case (p)
        3'd0: planes = 1; // binary
        3'd1: planes = 2; // ternary
        3'd2: planes = 2; // int2
        3'd3: planes = 4; // int4
        default: planes = 1;
      endcase
      expected_cycles = (ROWS - 1) + ((kdim / 8) * planes) + COLS;
    end
  endfunction

  task automatic drive_zero_operands();
    for (int i = 0; i < ROWS; i++) bmpu_act_operand_i[i] = '0;
    for (int j = 0; j < COLS; j++) bmpu_wgt_operand_i[j] = '0;
  endtask

  task automatic run_one_case(input logic [2:0] p, input int kdim);
    int cyc;
    int exp;
    begin
      exp = expected_cycles(p, kdim);
      k_dim_i = kdim;
      prec_i = p;
      output_en_i = 1'b0;
      valid_i = 1'b1;
      cyc = 0;

      while (!sa_done_o && cyc < (exp + 10)) begin
        @(posedge clk_i);
        cyc++;
        if ($isunknown(sa_done_o)) begin
          $fatal(1, "[SA_TB] sa_done_o has X at cycle %0d", cyc);
        end
      end

      if (!sa_done_o) begin
        $fatal(1, "[SA_TB] timeout waiting sa_done_o, prec=%0d k=%0d exp=%0d", p, kdim, exp);
      end

      if (cyc != exp) begin
        $fatal(1, "[SA_TB] sa_done_o cycle mismatch, prec=%0d got=%0d exp=%0d", p, cyc, exp);
      end

      output_en_i = 1'b1;
      @(posedge clk_i);
      for (int r = 0; r < ROWS; r++) begin
        if ($isunknown(output_data_o[r])) begin
          $fatal(1, "[SA_TB] output_data_o[%0d] has X", r);
        end
        if (output_data_o[r] !== '0) begin
          $fatal(1, "[SA_TB] zero-input case output not zero row=%0d data=%h", r, output_data_o[r]);
        end
      end

      valid_i = 1'b0;
      output_en_i = 1'b0;
      repeat (2) @(posedge clk_i);
      $display("[SA_TB] PASS prec=%0d k=%0d cycles=%0d", p, kdim, cyc);
    end
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b0;
    valid_i = 1'b0;
    output_en_i = 1'b0;
    k_dim_i = '0;
    prec_i = 3'd0;
    drive_zero_operands();

    repeat (5) @(posedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(posedge clk_i);

    run_one_case(3'd0, 64); // binary
    run_one_case(3'd1, 64); // ternary
    run_one_case(3'd2, 64); // int2
    run_one_case(3'd3, 64); // int4

    $display("[SA_TB] ALL TESTS PASSED");
    $finish;
  end
endmodule
