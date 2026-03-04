`default_nettype none
`timescale 1ns/1ps

//==============================================================================

module tt_um_neuromurf_seq_mac_inf (
    //==========================================================================
    // TinyTapeout Standard Interface
    //==========================================================================
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // Bidirectional inputs
    output wire [7:0] uio_out,  // Bidirectional outputs
    output wire [7:0] uio_oe,   // Bidirectional output enables (active high)
    input  wire       ena,      // Always 1 when design is powered
    input  wire       clk,      // Clock
    input  wire       rst_n     // Active-low reset
);

    //==========================================================================
    // Core Design Instance
    //==========================================================================
       
    seq_mac_inf_top u_core (
        // Clock and Reset
        .clk     (clk),
        .rst_n   (rst_n),
        .ena     (ena),
        
        // Data Interface
        .ui_in   (ui_in),
        .uo_out  (uo_out),
        
        // Control Interface
        .uio_in  (uio_in),
        .uio_out (uio_out),
        .uio_oe  (uio_oe)
    );

endmodule

`default_nettype wire
