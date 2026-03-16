`default_nettype none
`timescale 1ns/1ps

//==============================================================================
//
//  Module:      seq_mac_inf_top
//  Project:     SEQ_MAC_INF_16H - Sequential MAC with Inference
//  Description: Top-level integration with CDC, ReLU, muxes, and submodules
//
//==============================================================================

module seq_mac_inf_top (
    //==========================================================================
    // Clock and Reset
    //==========================================================================
    input  wire        clk,           // System clock (50 MHz)
    input  wire        rst_n,         // Async active-low reset
    input  wire        ena,           // Clock enable

    //==========================================================================
    // Data Interface
    //==========================================================================
    input  wire [7:0]  ui_in,         // Data input (weight/input)
    output wire [7:0]  uo_out,        // Data output (result/status)

    //==========================================================================
    // Control Interface
    //==========================================================================
    input  wire [7:0]  uio_in,        // Control inputs
    output wire [7:0]  uio_out,       // Control outputs (directly from uio_in)
    output wire [7:0]  uio_oe         // Output enable (directly from uio_in direction)
);

    //==========================================================================
    // Pin Decoding
    //==========================================================================
    wire       data_toggle_raw = uio_in[0];
    wire       data_type_raw   = uio_in[1];
    wire       start_raw       = uio_in[2];
    wire       mode0_raw       = uio_in[3];
    wire       next_byte_raw   = uio_in[4];
    wire       soft_rst        = uio_in[5];  // Level-sensitive, no sync needed
    wire       mode1_raw       = uio_in[6];
    wire       status_sel      = uio_in[7];  // Level-sensitive, no sync needed

    //==========================================================================
    // CDC Synchronizers (2-FF)
    //==========================================================================
    
    reg [1:0] toggle_sync;
    reg [1:0] start_sync;
    reg [1:0] mode0_sync;
    reg [1:0] mode1_sync;
    reg [1:0] next_sync;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            toggle_sync <= 2'b00;
            start_sync  <= 2'b00;
            mode0_sync  <= 2'b00;
            mode1_sync  <= 2'b00;
            next_sync   <= 2'b00;
        end else if (ena) begin
            toggle_sync <= {toggle_sync[0], data_toggle_raw};
            start_sync  <= {start_sync[0],  start_raw};
            mode0_sync  <= {mode0_sync[0],  mode0_raw};
            mode1_sync  <= {mode1_sync[0],  mode1_raw};
            next_sync   <= {next_sync[0],   next_byte_raw};
        end
    end
    
    // Synchronized signals
    wire       data_toggle = toggle_sync[1];
    wire       start       = start_sync[1];
    wire [1:0] mode        = {mode1_sync[1], mode0_sync[1]};
    wire       next_byte   = next_sync[1];

    //==========================================================================
    // Data Capture on RAW Toggle Edge (CDC FIX)
    //==========================================================================
    
    reg toggle_raw_prev;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            toggle_raw_prev <= 1'b0;
        else if (ena)
            toggle_raw_prev <= data_toggle_raw;
    end
    
    wire toggle_edge_raw = (data_toggle_raw != toggle_raw_prev);
    
    reg [7:0] data_captured;
    reg       type_captured;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_captured <= 8'd0;
            type_captured <= 1'b0;
        end else if (ena) begin
            if (toggle_edge_raw) begin
                data_captured <= ui_in;
                type_captured <= data_type_raw;
            end
        end
    end

    //==========================================================================
    // Internal Wires
    //==========================================================================
    
    // FSM to Datapath
    wire        weight_load;
    wire        input_load;
    wire        mult_start;
    wire        acc_clear;
    wire        acc_add;
    wire        result_load;
    wire        result_shift;
    wire        mult_done;
    
    // FSM to Hidden Bank
    wire        hidden_wr_en;
    wire [3:0]  hidden_wr_addr;
    wire [3:0]  hidden_rd_addr;
    wire        use_hidden;
    
    // FSM to Argmax
    wire        argmax_clear;
    wire        argmax_compare;
    wire [3:0]  argmax_class;
    
    // Status signals
    wire        busy;
    wire        done;
    wire        ready;
    wire        byte_valid;
    wire        inf_done;
    wire        layer;
    wire        err_flag;
    
    // Datapath outputs (26-bit accumulator)
    wire [25:0] acc_value;
    wire [7:0]  result_byte;
    
    // Hidden bank
    wire [7:0]  hidden_rd_data;
    
    // Argmax output
    wire [3:0]  best_class;

    //==========================================================================
    // ReLU Activation with Rounding (26-bit accumulator)
    //==========================================================================
    // Applies ReLU to signed accumulator:
    //   if (acc < 0) output = 0
    //   else         output = round(acc / 256) with saturation
    //
    
    wire acc_negative = acc_value[25];  // Sign bit (26-bit accumulator)
    wire acc_overflow = |acc_value[24:16];  // Overflow if upper bits set (positive case)
    
    // Rounding logic: add 0.5 (bit 7) before shifting
    wire [8:0] acc_rounded = {1'b0, acc_value[15:8]} + {8'b0, acc_value[7]};
    
    wire [7:0] hidden_wr_data = acc_negative ? 8'h00 :                     // ReLU: negative → 0
                                (acc_overflow || acc_rounded[8]) ? 8'hFF : // Saturate to 255
                                acc_rounded[7:0];                          // Rounded value

    //==========================================================================
    // Input Data Mux
    //==========================================================================
    
    wire [7:0] mac_input_data = (input_load && use_hidden) ? hidden_rd_data : data_captured;

    //==========================================================================
    // Datapath Instance
    //==========================================================================
    seq_mac_datapath u_datapath (
        .clk          (clk),
        .rst_n        (rst_n),
        .ena          (ena),
        .data_in      (mac_input_data),
        .weight_load  (weight_load),
        .input_load   (input_load),
        .mult_start   (mult_start),
        .acc_clear    (acc_clear),
        .acc_add      (acc_add),
        .result_load  (result_load),
        .result_shift (result_shift),
        .mult_done    (mult_done),
        .acc_out      (acc_value),
        .result_byte  (result_byte)
    );

    //==========================================================================
    // FSM Instance
    //==========================================================================
    seq_mac_inf_fsm u_fsm (
        .clk           (clk),
        .rst_n         (rst_n),
        .soft_rst      (soft_rst),
        .ena           (ena),
        .mode          (mode),
        .start         (start),
        .data_toggle   (data_toggle),
        .data_type     (type_captured),
        .next_byte     (next_byte),
        .mult_done     (mult_done),
        .weight_load   (weight_load),
        .input_load    (input_load),
        .mult_start    (mult_start),
        .acc_clear     (acc_clear),
        .acc_add       (acc_add),
        .result_load   (result_load),
        .result_shift  (result_shift),
        .hidden_wr_en  (hidden_wr_en),
        .hidden_wr_addr(hidden_wr_addr),
        .hidden_rd_addr(hidden_rd_addr),
        .use_hidden    (use_hidden),
        .argmax_clear  (argmax_clear),
        .argmax_compare(argmax_compare),
        .argmax_class  (argmax_class),
        .busy          (busy),
        .done          (done),
        .ready         (ready),
        .byte_valid    (byte_valid),
        .inf_done      (inf_done),
        .layer         (layer),
        .err_flag      (err_flag)
    );

    //==========================================================================
    // Hidden Register Bank Instance
    //==========================================================================
    hidden_reg_bank u_hidden (
        .clk     (clk),
        .rst_n   (rst_n),
        .ena     (ena),
        .wr_en   (hidden_wr_en),
        .wr_addr (hidden_wr_addr),
        .wr_data (hidden_wr_data),
        .rd_addr (hidden_rd_addr),
        .rd_data (hidden_rd_data)
    );

    //==========================================================================
    // Argmax Unit Instance
    //==========================================================================
    argmax_unit u_argmax (
        .clk        (clk),
        .rst_n      (rst_n),
        .ena        (ena),
        .clear      (argmax_clear),
        .compare_en (argmax_compare),
        .score      (acc_value),
        .class_idx  (argmax_class),
        .best_class (best_class)
    );

    //==========================================================================
    // Output Mux
    //==========================================================================
    
    wire [7:0] status_byte = {
        1'b0,       // [7] reserved
        err_flag,   // [6] protocol error
        layer,      // [5] current layer
        inf_done,   // [4] inference complete
        byte_valid, // [3] valid byte available
        ready,      // [2] ready for data
        done,       // [1] MAC_ONLY output phase
        busy        // [0] operation in progress
    };
    
    wire [7:0] data_byte = inf_done ? {4'b0000, best_class} : result_byte;
    
    assign uo_out = status_sel ? status_byte : data_byte;

    //==========================================================================
    // Bidirectional Pin Control
    //==========================================================================
    assign uio_oe  = 8'b00000000;  // All pins are inputs
    assign uio_out = 8'b00000000;  // No output drive

endmodule

`default_nettype wire