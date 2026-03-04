`default_nettype none
`timescale 1ns/1ps

//==============================================================================

module seq_mac_datapath (
    //==========================================================================
    // Clock and Reset
    //==========================================================================
    input  wire        clk,           // System clock (50 MHz)
    input  wire        rst_n,         // Async active-low reset
    input  wire        ena,           // Clock enable

    //==========================================================================
    // Data Inputs
    //==========================================================================
    input  wire [7:0]  data_in,       // Weight (signed) or Input (unsigned)

    //==========================================================================
    // Control Inputs (from FSM)
    //==========================================================================
    input  wire        weight_load,   // Load weight register
    input  wire        input_load,    // Load input register
    input  wire        mult_start,    // Start multiplication
    input  wire        acc_clear,     // Clear accumulator
    input  wire        acc_add,       // Add product to accumulator
    input  wire        result_load,   // Load result shift register
    input  wire        result_shift,  // Shift result register

    //==========================================================================
    // Status Outputs
    //==========================================================================
    output wire        mult_done,     // Multiplication complete

    //==========================================================================
    // Data Outputs
    //==========================================================================
    output wire [24:0] acc_out,       // Signed accumulator for ReLU/argmax
    output wire [7:0]  result_byte    // Current result byte (MAC_ONLY mode)
);

    //==========================================================================
    // Registers
    //==========================================================================
    
    // Weight register (signed 8-bit, stored as-is)
    reg [7:0] weight_reg;
    
    // Input register (unsigned 8-bit)
    reg [7:0] input_reg;
    
    // Serial multiplier registers
    reg [15:0] ser_a_shift;    // Multiplicand (unsigned input, shifts left)
    reg [7:0]  ser_b_shift;    // Multiplier (|weight|, shifts right)
    reg [15:0] ser_prod;       // Accumulated product (unsigned during mult)
    reg [3:0]  mult_cnt;       // Multiply cycle counter
    reg        mult_active;    // Multiplication in progress
    reg        weight_neg;     // Weight was negative (for final sign correction)
    reg        mult_done_reg;  // Multiplication complete flag
    
    // Final signed product after sign correction
    reg [16:0] final_prod;     // 17-bit signed product
    
    // Accumulator (25-bit signed)
    reg signed [24:0] acc_reg;
    
    // Output shift register (for MAC_ONLY 3-byte readout)
    reg [23:0] out_shift_reg;

    //==========================================================================
    // Weight and Input Registers
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 8'd0;
            input_reg  <= 8'd0;
        end else if (ena) begin
            if (weight_load) begin
                weight_reg <= data_in;  // Signed weight (two's complement)
            end
            if (input_load) begin
                input_reg <= data_in;   // Unsigned input
            end
        end
    end

    //==========================================================================
    // Serial Signed × Unsigned Multiplier
    //==========================================================================

    // Absolute value of weight
    wire [7:0] weight_abs = weight_reg[7] ? (~weight_reg + 8'd1) : weight_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ser_a_shift  <= 16'd0;
            ser_b_shift  <= 8'd0;
            ser_prod     <= 16'd0;
            mult_cnt     <= 4'd0;
            mult_active  <= 1'b0;
            weight_neg   <= 1'b0;
            mult_done_reg<= 1'b0;
            final_prod   <= 17'd0;
        end else if (ena) begin
            // Default: clear done flag after one cycle
            mult_done_reg <= 1'b0;
            
            if (mult_start) begin
                // Initialize multiplication
                ser_a_shift  <= {8'd0, input_reg};  // Zero-extend input to 16 bits
                ser_b_shift  <= weight_abs;          // Use |weight|
                ser_prod     <= 16'd0;
                mult_cnt     <= 4'd0;
                mult_active  <= 1'b1;
                weight_neg   <= weight_reg[7];       // Remember original sign
                mult_done_reg<= 1'b0;
            end else if (mult_active) begin
                if (mult_cnt < 4'd8) begin
                    // Shift-add step
                    if (ser_b_shift[0]) begin
                        ser_prod <= ser_prod + ser_a_shift;
                    end
                    ser_a_shift <= ser_a_shift << 1;
                    ser_b_shift <= ser_b_shift >> 1;
                    mult_cnt    <= mult_cnt + 4'd1;
                end else begin
                    // Multiplication complete - apply sign correction
                    if (weight_neg) begin
                        // Negate: two's complement of 16-bit result, sign-extend to 17 bits
                        final_prod <= {1'b1, ~ser_prod + 16'd1};
                    end else begin
                        // Positive: zero-extend to 17 bits
                        final_prod <= {1'b0, ser_prod};
                    end
                    mult_active   <= 1'b0;
                    mult_done_reg <= 1'b1;
                end
            end
        end
    end
    
    assign mult_done = mult_done_reg;

    //==========================================================================
    // Signed Accumulator (25-bit)
    //==========================================================================
    
    // Sign-extend 17-bit product to 25 bits
    wire signed [24:0] product_sext = {{8{final_prod[16]}}, final_prod};
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 25'sd0;
        end else if (ena) begin
            if (acc_clear) begin
                acc_reg <= 25'sd0;
            end else if (acc_add) begin
                acc_reg <= acc_reg + product_sext;
            end
        end
    end
    
    // Expose accumulator for ReLU/truncation and argmax
    assign acc_out = acc_reg;

    //==========================================================================
    // Output Shift Register (MAC_ONLY mode)
    //==========================================================================
     
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_shift_reg <= 24'd0;
        end else if (ena) begin
            if (result_load) begin
                // Load accumulator value (lower 20 bits)
                out_shift_reg <= {4'd0, acc_reg[19:0]};
            end else if (result_shift) begin
                // Shift out LSB first
                out_shift_reg <= {8'd0, out_shift_reg[23:8]};
            end
        end
    end
    
    assign result_byte = out_shift_reg[7:0];

endmodule

`default_nettype wire
