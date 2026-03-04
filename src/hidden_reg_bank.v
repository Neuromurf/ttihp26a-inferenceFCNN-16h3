`default_nettype none
`timescale 1ns/1ps

//==============================================================================
//
//  Module:      hidden_reg_bank
//  Project:     SEQ_MAC_INF_16H - Sequential MAC with Inference
//  Description: 16×8-bit register bank for hidden layer activations
//
//==============================================================================

module hidden_reg_bank (
    //==========================================================================
    // Clock and Reset
    //==========================================================================
    input  wire        clk,          // System clock (50 MHz)
    input  wire        rst_n,        // Async active-low reset
    input  wire        ena,          // Clock enable
    
    //==========================================================================
    // Write Port (from ReLU output during Layer 1)
    //==========================================================================
    input  wire        wr_en,        // Write enable (single cycle pulse)
    input  wire [3:0]  wr_addr,      // Write address (0-15)
    input  wire [7:0]  wr_data,      // Write data (ReLU-activated value)
    
    //==========================================================================
    // Read Port (to MAC input mux during Layer 2)
    //==========================================================================
    input  wire [3:0]  rd_addr,      // Read address (0-15)
    output wire [7:0]  rd_data       // Read data (combinational)
);

    //==========================================================================
    // Register Bank Storage
    //==========================================================================

    reg [7:0] hidden_regs [0:15];
    
    //==========================================================================
    // Write Logic
    //==========================================================================

    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers to zero
            for (i = 0; i < 16; i = i + 1) begin
                hidden_regs[i] <= 8'd0;
            end
        end else if (ena && wr_en) begin
            // Write to addressed register
            hidden_regs[wr_addr] <= wr_data;
        end
    end
    
    //==========================================================================
    // Read Logic
    //==========================================================================

    assign rd_data = hidden_regs[rd_addr];

endmodule

`default_nettype wire
