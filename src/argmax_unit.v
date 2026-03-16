`default_nettype none
`timescale 1ns/1ps

//==============================================================================
//
//  Module:      argmax_unit
//  Project:     SEQ_MAC_INF_16H - Sequential MAC with Inference
//  Description: Winner-Take-All argmax with signed 26-bit comparison
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
    input  wire signed [25:0] score, // Current class score (signed, 26-bit)
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
    reg signed [25:0] max_value;
    
    // Minimum signed 26-bit value: -2^25 = -33,554,432
    localparam signed [25:0] MIN_VALUE = 26'sh2000000;

    //==========================================================================
    // Comparison Logic
    //==========================================================================
    
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