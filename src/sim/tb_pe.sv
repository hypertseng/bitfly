`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 15:40:44
// Design Name: 
// Module Name: tb_pe
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tb_pe;

  // Inputs
  logic [ 7:0] weights;  // 8-bit weights (1: keep value, 0: negate)
  logic [63:0] activations;  // 64-bit activations (8x int8)

  // Outputs
  logic [11:0] result;  // Final int16 result

  // Instantiate the module
  pe uut (
      .weights(weights),
      .activations(activations),
      .result(result)
  );

  // Testbench logic
  initial begin
    // Test case 1: All weights are 1
    weights = 8'b11111111;
    activations = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};  // 8x int8
    #10;
    $display("Test 1: Result = %d", $signed(result));  // Expected: 36

    // Test case 2: All weights are 0 (negate activations)
    weights = 8'b00000000;
    activations = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};  // 8x int8
    #10;
    $display("Test 2: Result = %d", $signed(result));  // Expected: -36

    // Test case 3: Mixed weights
    weights = 8'b10101010;
    activations = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};  // 8x int8
    #10;
    $display("Test 3: Result = %d", $signed(result));  // Expected: 4

    // Test case 4: Large activations (to test int16 range)
    weights = 8'b11111111;
    activations = {8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127};  // 8x 127
    #10;
    $display("Test 4: Result = %d", $signed(result));  // Expected: 1016

    // Test case 5: Mixed activations with large values
    weights = 8'b10101010;
    activations = {8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127};  // 8x 127
    #10;
    $display("Test 5: Result = %d", $signed(result));  // Expected: 0

    // Finish simulation
    $finish;
  end

  // Dump waveform for power analysis
  initial begin
    $dumpfile("waveform_int.vcd");  // Updated waveform file name
    $dumpvars(0, tb_pe);
  end

endmodule
