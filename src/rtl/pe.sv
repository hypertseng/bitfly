`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 15:39:54
// Design Name: 
// Module Name: pe
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


module pe (
    input  logic [ 7:0] weights,      // 8-bit weights (1: keep value, 0: negate)
    input  logic [63:0] activations,  // 64-bit activations (8x int8)
    output logic [15:0] result        // Final int16 result
);

  // Weighted activations
  logic signed [7:0][7:0] weighted_activations;

  // Weight processing and dot product
  always_comb begin
    for (int i = 0; i < 8; i++) begin
      // If weight is 0 (representing -1), negate the activation value
      weighted_activations[i] = weights[i] ? activations[(i*8)+:8] : ~activations[(i*8)+:8] + 1;
    end
  end

  // Accumulate results
  always_comb begin
    // Use a wider temporary variable to avoid overflow
    logic signed [10:0] temp_sum;  // 11-bit temporary sum to handle overflow
    temp_sum = $signed(weighted_activations[0]) + $signed(weighted_activations[1]) +
        $signed(weighted_activations[2]) + $signed(weighted_activations[3]) +
        $signed(weighted_activations[4]) + $signed(weighted_activations[5]) +
        $signed(weighted_activations[6]) + $signed(weighted_activations[7]);

    // Output the result as int16 (no need for saturation)
    result = $signed(temp_sum);
  end

endmodule
