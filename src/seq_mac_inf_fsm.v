`default_nettype none
`timescale 1ns/1ps

//==============================================================================
//
//  Module:      seq_mac_inf_fsm
//  Project:     SEQ_MAC_INF_16H - Sequential MAC with Inference
//  Description: Control FSM for MAC_ONLY and INFERENCE modes
//
//==============================================================================
//
//  OVERVIEW
//  --------
//  Controls the sequential MAC datapath for:
//  • MAC_ONLY mode:  9-tap dot product (backward compatible)
//  • INFERENCE mode: 785→16→10 neural network with folded bias
//
//  OPERATING MODES
//  ---------------
//  mode[1:0]:
//    00 = IDLE      - Wait for start
//    01 = MAC_ONLY  - 9-tap MAC, 20-bit result readout
//    10 = INFERENCE - Full MNIST inference (785→16→10)
//    11 = RESERVED  - Behaves as IDLE
//
//  INFERENCE MODE FLOW
//  -------------------
//  Layer 1 (785 → 16):
//    - 16 hidden neurons, each with 785 signed weights
//    - 785 = 784 pixels + 1 folded bias input (constant 255)
//    - 88 passes: 87 × 9 taps + 1 × 2 taps
//    - Result: ReLU(accumulator) → 8-bit unsigned → hidden bank
//
//  Layer 2 (16 → 10):
//    - 10 output classes, each with 16 signed weights
//    - Hidden values fetched internally as unsigned inputs
//    - After each class: argmax comparison (signed)
//    - After class 9: best_class = predicted digit
//
//  PROTOCOL
//  --------
//  • data_toggle: Any edge triggers data transfer
//  • data_type:   0=weight, 1=input (Layer 1 only)
//  • Layer 2:     Only weights sent, hidden values fetched internally
//
//==============================================================================

module seq_mac_inf_fsm (
    //==========================================================================
    // Clock and Reset
    //==========================================================================
    input  wire        clk,           // System clock (50 MHz)
    input  wire        rst_n,         // Async active-low reset
    input  wire        soft_rst,      // Synchronous soft reset
    input  wire        ena,           // Clock enable

    //==========================================================================
    // Mode and Control Inputs (directly from synchronized inputs)
    //==========================================================================
    input  wire [1:0]  mode,          // Operating mode
    input  wire        start,         // Start pulse
    input  wire        data_toggle,   // Data toggle (edge-sensitive)
    input  wire        data_type,     // 0=weight, 1=input
    input  wire        next_byte,     // Next result byte (MAC_ONLY)

    //==========================================================================
    // Datapath Interface
    //==========================================================================
    input  wire        mult_done,     // Multiplication complete
    output reg         weight_load,   // Load weight register
    output reg         input_load,    // Load input register
    output reg         mult_start,    // Start multiplication
    output reg         acc_clear,     // Clear accumulator
    output reg         acc_add,       // Add product to accumulator
    output reg         result_load,   // Load result shift register
    output reg         result_shift,  // Shift result register

    //==========================================================================
    // Hidden Register Bank Interface
    //==========================================================================
    output reg         hidden_wr_en,  // Write to hidden bank
    output reg  [3:0]  hidden_wr_addr,// Write address (0-15)
    output wire [3:0]  hidden_rd_addr,// Read address (0-15)
    output reg         use_hidden,    // Mux select: 1=use hidden value as input

    //==========================================================================
    // Argmax Interface
    //==========================================================================
    output reg         argmax_clear,  // Clear argmax for new inference
    output reg         argmax_compare,// Trigger comparison
    output wire [3:0]  argmax_class,  // Current class index for argmax

    //==========================================================================
    // Status Outputs
    //==========================================================================
    output reg         busy,          // Operation in progress
    output wire        done,          // MAC_ONLY output phase
    output wire        ready,         // Ready to accept data
    output reg         byte_valid,    // Valid result byte available
    output reg         inf_done,      // Inference complete
    output reg         layer,         // Current layer: 0=L1, 1=L2
    output reg         err_flag       // Protocol error
);

    //==========================================================================
    // Mode Encoding
    //==========================================================================
    localparam [1:0]
        MODE_IDLE      = 2'b00,
        MODE_MAC_ONLY  = 2'b01,
        MODE_INFERENCE = 2'b10;
        // MODE_RESERVED (2'b11) not used - handled as default case

    //==========================================================================
    // State Encoding
    //==========================================================================
    localparam [4:0]
        // Common states
        S_IDLE          = 5'd0,
        S_CLEAR         = 5'd1,
        S_WAIT_WEIGHT   = 5'd2,
        S_WAIT_INPUT    = 5'd3,
        S_START_MULT    = 5'd4,
        S_MULTIPLY      = 5'd5,
        S_ACCUM         = 5'd6,
        S_NEXT_TAP      = 5'd7,
        // MAC_ONLY output states
        S_FINISH        = 5'd8,
        S_OUTPUT        = 5'd9,
        S_BYTE_ACK      = 5'd10,
        // INFERENCE Layer 1 states
        S_L1_NEXT_PASS  = 5'd11,
        S_L1_STORE      = 5'd12,
        S_L1_NEXT_NEURON= 5'd13,
        // INFERENCE Layer 2 states
        S_L2_START      = 5'd14,
        S_L2_WAIT_WEIGHT= 5'd15,
        S_L2_LOAD_HIDDEN= 5'd16,
        S_L2_START_MULT = 5'd17,
        S_L2_MULTIPLY   = 5'd18,
        S_L2_ACCUM      = 5'd19,
        S_L2_NEXT_HIDDEN= 5'd20,
        S_L2_ARGMAX     = 5'd21,
        S_L2_NEXT_CLASS = 5'd22,
        S_INF_DONE      = 5'd23;

    //==========================================================================
    // Registers
    //==========================================================================
    reg [4:0]  state;
    reg [3:0]  tap_cnt;        // Taps within 9-tap pass (0-8)
    reg [6:0]  pass_cnt;       // Passes within neuron (0-87)
    reg [3:0]  neuron_cnt;     // Hidden neuron index (0-15)
    reg [3:0]  hidden_idx;     // Hidden read index for L2 (0-15)
    reg [3:0]  class_cnt;      // Output class index (0-9)
    reg [1:0]  byte_cnt;       // Output byte counter (0-2)
    reg        toggle_last;    // Previous toggle value
    reg        next_last;      // Previous next_byte value
    reg [1:0]  mode_latched;   // Latched mode at start

    //==========================================================================
    // Constants
    //==========================================================================
    // Layer 1: 785 inputs / 9 taps = 87.2 → 88 passes
    // Pass 0-86: 9 taps each (783 values)
    // Pass 87: 2 taps (pixel 784 + bias input 255)
    localparam [6:0] L1_LAST_PASS = 7'd87;
    localparam [3:0] L1_LAST_TAP_NORMAL = 4'd8;   // Taps 0-8 for passes 0-86
    localparam [3:0] L1_LAST_TAP_FINAL = 4'd1;    // Taps 0-1 for pass 87
    
    // Layer 2: 16 hidden values per class
    localparam [3:0] L2_LAST_HIDDEN = 4'd15;
    localparam [3:0] LAST_CLASS = 4'd9;
    localparam [3:0] LAST_NEURON = 4'd15;

    //==========================================================================
    // Edge Detection
    //==========================================================================
    wire toggle_event = (data_toggle != toggle_last);
    wire next_event   = (next_byte != next_last);
    
    // Determine last tap for current pass
    wire [3:0] l1_last_tap = (pass_cnt == L1_LAST_PASS) ? L1_LAST_TAP_FINAL : L1_LAST_TAP_NORMAL;

    //==========================================================================
    // Output Assignments
    //==========================================================================
    assign hidden_rd_addr = hidden_idx;
    assign argmax_class   = class_cnt;
    assign done           = (state == S_OUTPUT) || (state == S_BYTE_ACK);
    assign ready          = (state == S_WAIT_WEIGHT) || (state == S_WAIT_INPUT) ||
                            (state == S_L2_WAIT_WEIGHT);

    //==========================================================================
    // Main FSM
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Async reset
            state         <= S_IDLE;
            tap_cnt       <= 4'd0;
            pass_cnt      <= 7'd0;
            neuron_cnt    <= 4'd0;
            hidden_idx    <= 4'd0;
            class_cnt     <= 4'd0;
            byte_cnt      <= 2'd0;
            toggle_last   <= 1'b0;
            next_last     <= 1'b0;
            mode_latched  <= MODE_IDLE;
            busy          <= 1'b0;
            byte_valid    <= 1'b0;
            inf_done      <= 1'b0;
            layer         <= 1'b0;
            err_flag      <= 1'b0;
            weight_load   <= 1'b0;
            input_load    <= 1'b0;
            mult_start    <= 1'b0;
            acc_clear     <= 1'b0;
            acc_add       <= 1'b0;
            result_load   <= 1'b0;
            result_shift  <= 1'b0;
            hidden_wr_en  <= 1'b0;
            hidden_wr_addr<= 4'd0;
            use_hidden    <= 1'b0;
            argmax_clear  <= 1'b0;
            argmax_compare<= 1'b0;
        end else if (ena) begin
            // Default: clear single-cycle pulses
            weight_load    <= 1'b0;
            input_load     <= 1'b0;
            mult_start     <= 1'b0;
            acc_clear      <= 1'b0;
            acc_add        <= 1'b0;
            result_load    <= 1'b0;
            result_shift   <= 1'b0;
            hidden_wr_en   <= 1'b0;
            argmax_clear   <= 1'b0;
            argmax_compare <= 1'b0;
            
            // Soft reset handling
            if (soft_rst) begin
                state         <= S_IDLE;
                tap_cnt       <= 4'd0;
                pass_cnt      <= 7'd0;
                neuron_cnt    <= 4'd0;
                hidden_idx    <= 4'd0;
                class_cnt     <= 4'd0;
                byte_cnt      <= 2'd0;
                toggle_last   <= data_toggle;
                busy          <= 1'b0;
                byte_valid    <= 1'b0;
                inf_done      <= 1'b0;
                layer         <= 1'b0;
                err_flag      <= 1'b0;
                use_hidden    <= 1'b0;
            end else begin
                // State machine
                case (state)
                    //==========================================================
                    // S_IDLE: Wait for start
                    //==========================================================
                    S_IDLE: begin
                        toggle_last <= data_toggle;
                        next_last   <= next_byte;
                        
                        if (start && (mode == MODE_MAC_ONLY || mode == MODE_INFERENCE)) begin
                            busy         <= 1'b1;
                            inf_done     <= 1'b0;
                            err_flag     <= 1'b0;
                            tap_cnt      <= 4'd0;
                            pass_cnt     <= 7'd0;
                            neuron_cnt   <= 4'd0;
                            hidden_idx   <= 4'd0;
                            class_cnt    <= 4'd0;
                            mode_latched <= mode;
                            state        <= S_CLEAR;
                            
                            if (mode == MODE_INFERENCE) begin
                                argmax_clear <= 1'b1;
                            end
                        end
                    end

                    //==========================================================
                    // S_CLEAR: Clear accumulator
                    //==========================================================
                    S_CLEAR: begin
                        acc_clear   <= 1'b1;
                        toggle_last <= data_toggle;
                        state       <= S_WAIT_WEIGHT;
                    end

                    //==========================================================
                    // S_WAIT_WEIGHT: Wait for weight (toggle edge)
                    //==========================================================
                    S_WAIT_WEIGHT: begin
                        if (toggle_event) begin
                            if (data_type == 1'b0) begin
                                weight_load <= 1'b1;
                                toggle_last <= data_toggle;
                                state       <= S_WAIT_INPUT;
                            end else begin
                                err_flag    <= 1'b1;
                                toggle_last <= data_toggle;
                            end
                        end
                    end

                    //==========================================================
                    // S_WAIT_INPUT: Wait for input (toggle edge)
                    //==========================================================
                    S_WAIT_INPUT: begin
                        if (toggle_event) begin
                            if (data_type == 1'b1) begin
                                input_load  <= 1'b1;
                                toggle_last <= data_toggle;
                                state       <= S_START_MULT;
                            end else begin
                                err_flag    <= 1'b1;
                                toggle_last <= data_toggle;
                            end
                        end
                    end

                    //==========================================================
                    // S_START_MULT: Start multiplication
                    //==========================================================
                    S_START_MULT: begin
                        mult_start <= 1'b1;
                        state      <= S_MULTIPLY;
                    end

                    //==========================================================
                    // S_MULTIPLY: Wait for multiplication complete
                    //==========================================================
                    S_MULTIPLY: begin
                        if (mult_done) begin
                            state <= S_ACCUM;
                        end
                    end

                    //==========================================================
                    // S_ACCUM: Add product to accumulator
                    //==========================================================
                    S_ACCUM: begin
                        acc_add <= 1'b1;
                        state   <= S_NEXT_TAP;
                    end

                    //==========================================================
                    // S_NEXT_TAP: Decide next action based on tap count
                    //==========================================================
                    S_NEXT_TAP: begin
                        if (mode_latched == MODE_MAC_ONLY) begin
                            // MAC_ONLY: 9 taps total
                            if (tap_cnt == 4'd8) begin
                                state <= S_FINISH;
                            end else begin
                                tap_cnt <= tap_cnt + 4'd1;
                                state   <= S_WAIT_WEIGHT;
                            end
                        end else begin
                            // INFERENCE mode
                            if (tap_cnt == l1_last_tap) begin
                                state <= S_L1_NEXT_PASS;
                            end else begin
                                tap_cnt <= tap_cnt + 4'd1;
                                state   <= S_WAIT_WEIGHT;
                            end
                        end
                    end

                    //==========================================================
                    // MAC_ONLY Output States
                    //==========================================================
                    S_FINISH: begin
                        result_load <= 1'b1;
                        byte_cnt    <= 2'd0;
                        next_last   <= next_byte;  // Capture current state before S_OUTPUT
                        state       <= S_OUTPUT;
                    end

                    S_OUTPUT: begin
                        byte_valid <= 1'b1;
                        // Only update next_last when event detected (not every cycle)
                        
                        if (next_event) begin
                            next_last <= next_byte;
                            if (byte_cnt == 2'd2) begin
                                busy       <= 1'b0;
                                byte_valid <= 1'b0;
                                state      <= S_IDLE;
                            end else begin
                                result_shift <= 1'b1;
                                byte_cnt     <= byte_cnt + 2'd1;
                                state        <= S_BYTE_ACK;
                            end
                        end
                    end

                    S_BYTE_ACK: begin
                        next_last <= next_byte;  // Resync before returning to S_OUTPUT
                        state <= S_OUTPUT;
                    end

                    //==========================================================
                    // INFERENCE Layer 1 States
                    //==========================================================
                    S_L1_NEXT_PASS: begin
                        if (pass_cnt == L1_LAST_PASS) begin
                            // All 785 inputs done, store hidden value
                            state <= S_L1_STORE;
                        end else begin
                            // More passes needed
                            pass_cnt <= pass_cnt + 7'd1;
                            tap_cnt  <= 4'd0;
                            state    <= S_WAIT_WEIGHT;
                        end
                    end

                    S_L1_STORE: begin
                        // Write ReLU-activated value to hidden bank
                        hidden_wr_en   <= 1'b1;
                        hidden_wr_addr <= neuron_cnt;
                        state          <= S_L1_NEXT_NEURON;
                    end

                    S_L1_NEXT_NEURON: begin
                        if (neuron_cnt == LAST_NEURON) begin
                            // All 16 hidden neurons done, start Layer 2
                            state <= S_L2_START;
                        end else begin
                            // Next neuron
                            neuron_cnt <= neuron_cnt + 4'd1;
                            pass_cnt   <= 7'd0;
                            tap_cnt    <= 4'd0;
                            acc_clear  <= 1'b1;
                            state      <= S_WAIT_WEIGHT;
                        end
                    end

                    //==========================================================
                    // INFERENCE Layer 2 States
                    //==========================================================
                    S_L2_START: begin
                        layer       <= 1'b1;
                        use_hidden  <= 1'b1;
                        hidden_idx  <= 4'd0;
                        class_cnt   <= 4'd0;
                        acc_clear   <= 1'b1;
                        toggle_last <= data_toggle;
                        state       <= S_L2_WAIT_WEIGHT;
                    end

                    S_L2_WAIT_WEIGHT: begin
                        if (toggle_event) begin
                            weight_load <= 1'b1;
                            toggle_last <= data_toggle;
                            state       <= S_L2_LOAD_HIDDEN;
                        end
                    end

                    S_L2_LOAD_HIDDEN: begin
                        // Hidden value is already on rd_data (combinational)
                        // Load it as input
                        input_load <= 1'b1;
                        state      <= S_L2_START_MULT;
                    end

                    S_L2_START_MULT: begin
                        mult_start <= 1'b1;
                        state      <= S_L2_MULTIPLY;
                    end

                    S_L2_MULTIPLY: begin
                        if (mult_done) begin
                            state <= S_L2_ACCUM;
                        end
                    end

                    S_L2_ACCUM: begin
                        acc_add <= 1'b1;
                        state   <= S_L2_NEXT_HIDDEN;
                    end

                    S_L2_NEXT_HIDDEN: begin
                        if (hidden_idx == L2_LAST_HIDDEN) begin
                            // All 16 hidden values done for this class
                            state <= S_L2_ARGMAX;
                        end else begin
                            hidden_idx <= hidden_idx + 4'd1;
                            state      <= S_L2_WAIT_WEIGHT;
                        end
                    end

                    S_L2_ARGMAX: begin
                        argmax_compare <= 1'b1;
                        state          <= S_L2_NEXT_CLASS;
                    end

                    S_L2_NEXT_CLASS: begin
                        if (class_cnt == LAST_CLASS) begin
                            // All 10 classes done
                            state <= S_INF_DONE;
                        end else begin
                            class_cnt  <= class_cnt + 4'd1;
                            hidden_idx <= 4'd0;
                            acc_clear  <= 1'b1;
                            state      <= S_L2_WAIT_WEIGHT;
                        end
                    end

                    S_INF_DONE: begin
                        inf_done   <= 1'b1;
                        busy       <= 1'b0;
                        use_hidden <= 1'b0;
                        layer      <= 1'b0;
                        state      <= S_IDLE;
                    end

                    default: begin
                        state <= S_IDLE;
                    end
                endcase
            end
        end
    end

endmodule

`default_nettype wire
