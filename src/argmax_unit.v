`default_nettype none
`timescale 1ns/1ps

//==============================================================================
//
//  Module:      argmax_unit
//  Project:     SEQ_MAC_INF_16H - Sequential MAC with Inference
//  Description: Winner-Take-All argmax with signed 21-bit comparison
//
//==============================================================================
//
//  FUNCTION
//  --------
//  Tracks the maximum class score and its index during Layer 2 inference.
//  Uses signed comparison since class scores can be negative with signed weights.
//
//  ALGORITHM (Winner Take All)
//  ---------------------------
//  For each class (0-9):
//    1. Compute class score via MAC (signed 21-bit)
//    2. Compare with current maximum (signed comparison)
//    3. If score > max_value, update max_value and best_class
//  
//  After class 9: best_class contains the predicted digit (0-9)
//
//  DATA FORMAT
//  -----------
//  • score:      21-bit signed (from accumulator)
//  • max_value:  21-bit signed (stored maximum)
//  • class_idx:  4-bit unsigned (0-9)
//  • best_class: 4-bit unsigned (winning class index)
//
//  TIMING
//  ------
//  • clear:      Resets max_value to minimum signed value
//  • compare_en: Single-cycle pulse triggers comparison
//  • Result:     best_class valid after 10 compare cycles
//
//==============================================================================

module argmax_unit (
    //==========================================================================
    // Clock and Reset
    //==========================================================================
    input  wire        clk,          // System clock (50 MHz)
    input  wire        rst_n,        // Async active-low reset
    input  wire        ena,          // Clock enable
    
    //==========================================================================
    // Control Inputs
    //==========================================================================
    input  wire        clear,        // Clear for new inference
    input  wire        compare_en,   // Trigger comparison
    
    //==========================================================================
    // Data Inputs
    //==========================================================================
    input  wire signed [24:0] score, // Current class score (signed, 25-bit)
    input  wire [3:0]  class_idx,    // Current class index (0-9)
    
    //==========================================================================
    // Data Outputs
    //==========================================================================
    output reg  [3:0]  best_class    // Winning class index
);

    //==========================================================================
    // Internal Registers
    //==========================================================================
    
    // Maximum value seen so far (signed)
    // Initialize to most negative value so first score always wins
    reg signed [24:0] max_value;
    
    // Minimum signed 25-bit value: -2^24 = -16,777,216
    localparam signed [24:0] MIN_VALUE = 25'sh1000000;  // -16777216 in two's complement

    //==========================================================================
    // Comparison Logic
    //==========================================================================
    //
    // Winner-Take-All: Keep track of highest score and its index
    // Uses signed comparison since scores can be negative
    //
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset to minimum value so first comparison always updates
            max_value  <= MIN_VALUE;
            best_class <= 4'd0;
        end else if (ena) begin
            if (clear) begin
                // Clear for new inference
                max_value  <= MIN_VALUE;
                best_class <= 4'd0;
            end else if (compare_en) begin
                // Signed comparison: if current score > max, update
                if (score > max_value) begin
                    max_value  <= score;
                    best_class <= class_idx;
                end
            end
        end
    end

endmodule

`default_nettype wire
