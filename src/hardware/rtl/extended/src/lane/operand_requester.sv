// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
// Description:
// This stage is responsible for requesting individual elements from the vector
// register file, in order, and sending them to the corresponding operand
// queues. This stage also includes the VRF arbiter.

`include "bitfly_debug.svh"

module operand_requester
  import ara_pkg::*;
  import rvv_pkg::*;
#(
    parameter int unsigned NrLanes = 0,
    parameter int unsigned VLEN = 0,
    parameter int unsigned NrBanks = 0,  // Number of banks in the vector register file
    parameter type vaddr_t = logic,  // Type used to address vector register file elements
    parameter type operand_request_cmd_t = logic,
    parameter type operand_queue_cmd_t = logic,
    // Dependant parameters. DO NOT CHANGE!
    localparam type strb_t = logic [$bits(elen_t)/8-1:0],
    localparam type vlen_t = logic [$clog2(VLEN+1)-1:0],
    localparam int unsigned NrBmpuResultQueues = 4
) (
    input logic clk_i,
    input logic rst_ni,
    input logic [cf_math_pkg::idx_width(NrLanes)-1:0] lane_id_i,
    // Interface with the main sequencer
    input logic [NrVInsn-1:0][NrVInsn-1:0] global_hazard_table_i,
    // Interface with the lane sequencer
    input operand_request_cmd_t [NrOperandQueues-1:0] operand_request_i,
    input logic [NrOperandQueues-1:0] operand_request_valid_i,
    output logic [NrOperandQueues-1:0] operand_request_ready_o,
    // Support for store exception flush
    input logic lsu_ex_flush_i,
    output logic lsu_ex_flush_o,
    // Interface with the VRF
    output logic [NrBanks-1:0] vrf_req_o,
    output vaddr_t [NrBanks-1:0] vrf_addr_o,
    output logic [NrBanks-1:0] vrf_wen_o,
    output elen_t [NrBanks-1:0] vrf_wdata_o,
    output strb_t [NrBanks-1:0] vrf_be_o,
    output opqueue_e [NrBanks-1:0] vrf_tgt_opqueue_o,
    // Interface with the operand queues
    input logic [NrOperandQueues-1:0] operand_queue_ready_i,
    output logic [NrOperandQueues-1:0] operand_issued_o,
    output operand_queue_cmd_t [NrOperandQueues-1:0] operand_queue_cmd_o,
    output logic [NrOperandQueues-1:0] operand_queue_cmd_valid_o,
    // Interface with the VFUs
    // ALU
    input logic alu_result_req_i,
    input vid_t alu_result_id_i,
    input vaddr_t alu_result_addr_i,
    input elen_t alu_result_wdata_i,
    input strb_t alu_result_be_i,
    output logic alu_result_gnt_o,
    // Multiplier/FPU
    input logic mfpu_result_req_i,
    input vid_t mfpu_result_id_i,
    input vaddr_t mfpu_result_addr_i,
    input elen_t mfpu_result_wdata_i,
    input strb_t mfpu_result_be_i,
    output logic mfpu_result_gnt_o,
    // Mask unit
    input logic masku_result_req_i,
    input vid_t masku_result_id_i,
    input vaddr_t masku_result_addr_i,
    input elen_t masku_result_wdata_i,
    input strb_t masku_result_be_i,
    output logic masku_result_gnt_o,
    output logic masku_result_final_gnt_o,
    // Slide unit
    input logic sldu_result_req_i,
    input vid_t sldu_result_id_i,
    input vaddr_t sldu_result_addr_i,
    input elen_t sldu_result_wdata_i,
    input strb_t sldu_result_be_i,
    output logic sldu_result_gnt_o,
    output logic sldu_result_final_gnt_o,
    // Load unit
    input logic ldu_result_req_i,
    input vid_t ldu_result_id_i,
    input vaddr_t ldu_result_addr_i,
    input elen_t ldu_result_wdata_i,
    input strb_t ldu_result_be_i,
    output logic ldu_result_gnt_o,
    output logic ldu_result_final_gnt_o,
    // BMPU
    input logic bmpu_result_req_i,
    input vid_t [NrBmpuResultQueues-1:0] bmpu_result_id_i,
    input vaddr_t [NrBmpuResultQueues-1:0] bmpu_result_addr_i,
    input elen_t [NrBmpuResultQueues-1:0] bmpu_result_wdata_i,
    input strb_t [NrBmpuResultQueues-1:0] bmpu_result_be_i,
    output logic bmpu_result_gnt_o,

    input logic               bmpu_en_i,
    input logic               bmpu_output_en_i,
    input logic               is_bmpu_store_i,
    input logic               is_bmpu_load_i,
    input logic [NrVInsn-1:0] is_bmpu_load_vinsn_i,
    input logic [NrVInsn-1:0] bmpu_load_slot_valid_i,
    input logic [NrVInsn-1:0][1:0] bmpu_load_slot_i,
    input logic [NrVInsn-1:0] bmpu_load_is_weight_i
);

  import cf_math_pkg::idx_width;

  ////////////////////////
  //  Stream registers  //
  ////////////////////////

  typedef struct packed {
    vid_t   id;
    vaddr_t addr;
    elen_t  wdata;
    strb_t  be;
  } stream_register_payload_t;

  // Load unit
  vid_t   ldu_result_id;
  vaddr_t ldu_result_addr;
  elen_t  ldu_result_wdata;
  strb_t  ldu_result_be;
  logic   ldu_result_req;
  logic   ldu_result_gnt;
  stream_register #(
      .T(stream_register_payload_t)
  ) i_ldu_stream_register (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .clr_i     (1'b0),
      .testmode_i(1'b0),
      .data_i    ({ldu_result_id_i, ldu_result_addr_i, ldu_result_wdata_i, ldu_result_be_i}),
      .valid_i   (ldu_result_req_i),
      .ready_o   (ldu_result_gnt_o),
      .data_o    ({ldu_result_id, ldu_result_addr, ldu_result_wdata, ldu_result_be}),
      .valid_o   (ldu_result_req),
      .ready_i   (ldu_result_gnt)
  );

  // Slide unit
  vid_t   sldu_result_id;
  vaddr_t sldu_result_addr;
  elen_t  sldu_result_wdata;
  strb_t  sldu_result_be;
  logic   sldu_result_req;
  logic   sldu_result_gnt;
  stream_register #(
      .T(stream_register_payload_t)
  ) i_sldu_stream_register (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .clr_i     (1'b0),
      .testmode_i(1'b0),
      .data_i    ({sldu_result_id_i, sldu_result_addr_i, sldu_result_wdata_i, sldu_result_be_i}),
      .valid_i   (sldu_result_req_i),
      .ready_o   (sldu_result_gnt_o),
      .data_o    ({sldu_result_id, sldu_result_addr, sldu_result_wdata, sldu_result_be}),
      .valid_o   (sldu_result_req),
      .ready_i   (sldu_result_gnt)
  );

  // Mask unit
  vid_t   masku_result_id;
  vaddr_t masku_result_addr;
  elen_t  masku_result_wdata;
  strb_t  masku_result_be;
  logic   masku_result_req;
  logic   masku_result_gnt;
  stream_register #(
      .T(stream_register_payload_t)
  ) i_masku_stream_register (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .clr_i(1'b0),
      .testmode_i(1'b0),
      .data_i({masku_result_id_i, masku_result_addr_i, masku_result_wdata_i, masku_result_be_i}),
      .valid_i(masku_result_req_i),
      .ready_o(masku_result_gnt_o),
      .data_o({masku_result_id, masku_result_addr, masku_result_wdata, masku_result_be}),
      .valid_o(masku_result_req),
      .ready_i(masku_result_gnt)
  );

  // The very last grant must happen when the instruction actually write in the VRF
  // Otherwise the dependency is freed in advance
  always_ff @(posedge clk_i or negedge rst_ni) begin : p_final_gnts
    if (!rst_ni) begin
      ldu_result_final_gnt_o   <= 1'b0;
      sldu_result_final_gnt_o  <= 1'b0;
      masku_result_final_gnt_o <= 1'b0;
    end else begin
      ldu_result_final_gnt_o   <= ldu_result_gnt;
      sldu_result_final_gnt_o  <= sldu_result_gnt;
      masku_result_final_gnt_o <= masku_result_gnt;
    end
  end

  ///////////////////////
  //  Stall mechanism  //
  ///////////////////////

  // To handle any type of stall between vector instructions, we ensure
  // that operands of a second instruction that has a hazard on a first
  // instruction are read at the same rate the results of the second
  // instruction are written. Therefore, the second instruction can never
  // overtake the first one.

  // Instruction wrote a result
  logic [NrVInsn-1:0] vinsn_result_written_d, vinsn_result_written_q;
  logic [NrBmpuResultQueues-1:0] bmpu_result_granted_d, bmpu_result_granted_q;

  always_comb begin
    vinsn_result_written_d = '0;

    // Which vector instructions are writing something?
    vinsn_result_written_d[alu_result_id_i] |= alu_result_gnt_o;
    vinsn_result_written_d[mfpu_result_id_i] |= mfpu_result_gnt_o;
    vinsn_result_written_d[masku_result_id] |= masku_result_gnt;
    vinsn_result_written_d[ldu_result_id] |= ldu_result_gnt;
    vinsn_result_written_d[sldu_result_id] |= sldu_result_gnt;
    for (int i = 0; i < NrBmpuResultQueues; i++) begin
      vinsn_result_written_d[bmpu_result_id_i[i]] |= bmpu_result_gnt_o;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin : p_vinsn_result_written_ff
    if (!rst_ni) begin
      vinsn_result_written_q <= '0;
      lsu_ex_flush_o <= 1'b0;
      bmpu_result_granted_q <= '0;
    end else begin
      vinsn_result_written_q <= vinsn_result_written_d;
      lsu_ex_flush_o <= lsu_ex_flush_i;
      bmpu_result_granted_q <= bmpu_result_granted_d;
    end
  end

  ///////////////////////
  //  Operand request  //
  ///////////////////////

  // There is an operand requester_index for each operand queue. Each one
  // can be in one of the following two states.
  typedef enum logic {
    IDLE,
    REQUESTING
  } state_t;

  // A set bit indicates that the the master q is requesting access to the bank b
  // Masters 0 to NrOperandQueues-1 correspond to the operand queues.
  // The remaining four masters correspond to the ALU, the MFPU, the MASKU, the VLDU, and the SLDU.
  localparam NrGlobalMasters = 5;
  localparam NrBmpuOutQueues = NrBmpuResultQueues;
  localparam NrMasters = NrOperandQueues + NrGlobalMasters + NrBmpuOutQueues;

  typedef struct packed {
    vaddr_t addr;
    logic wen;
    elen_t wdata;
    strb_t be;
    opqueue_e opqueue;
  } payload_t;

  logic [NrBanks-1:0][NrOperandQueues-1:0] lane_operand_req;
  logic [NrOperandQueues-1:0][NrBanks-1:0] lane_operand_req_transposed;
  logic [NrBanks-1:0][NrGlobalMasters+NrBmpuOutQueues-1:0] ext_operand_req;
  logic [NrBanks-1:0][NrMasters-1:0] operand_gnt;
  payload_t [NrMasters-1:0] operand_payload;
  logic [NrBmpuResultQueues-1:0] bmpu_result_gnt_vec;

  // Metadata required to request all elements of this vector operand
  typedef struct packed {
    // ID of the instruction for this requester_index
    vid_t   id;
    // Address of the next element to be read
    vaddr_t addr;
    // How many elements remain to be read
    vlen_t  len;
    // Element width
    vew_e   vew;
    logic   bmpu_replay_en;
    logic [3:0] bmpu_self_blocks;
    logic [3:0] bmpu_repeat_blocks;
    vlen_t      bmpu_block_words;
    logic [2:0] bmpu_self_idx;
    logic [2:0] bmpu_repeat_idx;
    vlen_t      bmpu_word_idx;
    vaddr_t     bmpu_base_addr;

    // Hazards between vector instructions
    logic [NrVInsn-1:0] hazard;

    // Widening instructions produces two writes of every read
    // In case of a WAW with a previous instruction,
    // read once every two writes of the previous instruction
    logic is_widening;
    // One-bit counters
    logic [NrVInsn-1:0] waw_hazard_counter;
  } requester_metadata_t;

  for (genvar b = 0; b < NrBanks; b++) begin
    for (genvar r = 0; r < NrOperandQueues; r++) begin
      assign lane_operand_req[b][r] = lane_operand_req_transposed[r][b];
    end
  end

  logic bmpu_output_en_q;
  logic   [idx_width(NrBanks)-1:0] ldu_result_bank_mapped;
  vaddr_t                          ldu_result_addr_mapped;

  always_comb begin : p_ldu_bmpu_bank_map
    ldu_result_addr_mapped = ldu_result_addr;
    ldu_result_bank_mapped = ldu_result_addr[idx_width(NrBanks)-1:0];

    if (is_bmpu_load_vinsn_i[ldu_result_id] && bmpu_load_slot_valid_i[ldu_result_id]) begin
      automatic logic [idx_width(NrBanks)-1:0] slot_base;
      automatic logic [idx_width(NrBanks)-1:0] raw_bank;
      automatic logic [idx_width(NrBanks)-1:0] bank_in_slot;
      slot_base = {bmpu_load_slot_i[ldu_result_id], 1'b0};
      raw_bank = ldu_result_addr[idx_width(NrBanks)-1:0];
      bank_in_slot = raw_bank[0];
      ldu_result_bank_mapped = slot_base + bank_in_slot;
      ldu_result_addr_mapped[idx_width(NrBanks)-1:0] = ldu_result_bank_mapped;
`ifndef SYNTHESIS
      if (`BITFLY_OPREQ_DEBUG && (lane_id_i == '0)) begin
        $display("[%0t][OPREQ_MAP][lane%0d] id=%0d is_w=%0b slotv=%0b slot=%0d raw_bank=%0d bank_in_slot=%0d mapped_bank=%0d addr=%0d",
                 $time, lane_id_i, ldu_result_id, bmpu_load_is_weight_i[ldu_result_id],
                 bmpu_load_slot_valid_i[ldu_result_id], bmpu_load_slot_i[ldu_result_id],
                 raw_bank, bank_in_slot, ldu_result_bank_mapped, ldu_result_addr_mapped);
      end
`endif
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin : p_bmpu_output_en_ff
    if (!rst_ni) begin
      bmpu_output_en_q <= 1'b0;
    end else begin
      bmpu_output_en_q <= bmpu_output_en_i;
    end
  end

  for (
      genvar requester_index = 0; requester_index < NrOperandQueues; requester_index++
  ) begin : gen_operand_requester
    // State of this operand requester_index
    state_t state_d, state_q;

    requester_metadata_t requester_metadata_d, requester_metadata_q;
    // Is there a hazard during this cycle?
    logic stall;
    assign stall = (
      |(requester_metadata_q.hazard & ~(
        vinsn_result_written_q &
        (~{NrVInsn{requester_metadata_q.is_widening}} | requester_metadata_q.waw_hazard_counter)
      ))
      ) || (
        (requester_index == 15) && ((bmpu_output_en_q === 1'b1)||(bmpu_output_en_i === 1'b1))
      );

    // Did we get a grant?
    logic [NrBanks-1:0] operand_requester_gnt;
    for (genvar bank = 0; bank < NrBanks; bank++) begin : gen_operand_requester_gnt
      assign operand_requester_gnt[bank] = operand_gnt[bank][requester_index];
    end

    // Did we issue a word to this operand queue?
    assign operand_issued_o[requester_index] = |(operand_requester_gnt);

    logic [6:0] request_cnt_d, request_cnt_q;

    always_comb begin : operand_requester
      // Helper local variables
      automatic operand_queue_cmd_t  operand_queue_cmd_tmp;
      automatic requester_metadata_t requester_metadata_tmp;
      automatic vlen_t               effective_vector_body_length;
      automatic vaddr_t              vrf_addr;

      automatic elen_t               vl_byte;
      automatic elen_t               vstart_byte;
      automatic elen_t               vector_body_len_byte;
      automatic elen_t               scaled_vector_len_elements;
      automatic vaddr_t              bmpu_bank_addr_init;
      automatic logic                bmpu_word_ahead_hold;
      automatic int unsigned         bmpu_peer_index;
      automatic int unsigned         bmpu_progress_idx;
      automatic int unsigned         bmpu_peer_progress_idx;
      automatic int unsigned         bmpu_allowed_lead;

      // Bank we are currently requesting
      automatic int                  bank = requester_metadata_q.addr[idx_width(NrBanks)-1:0];

      // Maintain state
      state_d                          = state_q;
      requester_metadata_d             = requester_metadata_q;
      request_cnt_d                    = request_cnt_q;

      // Make no requests to the VRF
      operand_payload[requester_index] = '0;
      for (int b = 0; b < NrBanks; b++) lane_operand_req_transposed[requester_index][b] = 1'b0;

      // Do not acknowledge any operand requester_index commands
      operand_request_ready_o[requester_index] = 1'b0;

      // Do not send any operand conversion commands
      operand_queue_cmd_o[requester_index] = '0;
      operand_queue_cmd_valid_o[requester_index] = 1'b0;

      // Count the number of packets to fetch if we need to deshuffle.
      // Slide operations use the vstart signal, which does NOT correspond to the architectural
      // vstart, only when computing the fetch address. Ara supports architectural vstart > 0
      // only for memory operations.
      vl_byte     = operand_request_i[requester_index].vl     << operand_request_i[requester_index].vtype.vsew;
      vstart_byte = operand_request_i[requester_index].is_slide
                  ? 0
                  : operand_request_i[requester_index].vstart << operand_request_i[requester_index].vtype.vsew;
      vector_body_len_byte = vl_byte - vstart_byte + (vstart_byte % 8);
      scaled_vector_len_elements = vector_body_len_byte >> operand_request_i[requester_index].eew;
      if (scaled_vector_len_elements << operand_request_i[requester_index].eew < vector_body_len_byte)
        scaled_vector_len_elements += 1;

      // Final computed length
      effective_vector_body_length = (operand_request_i[requester_index].scale_vl)
                                   ? scaled_vector_len_elements
                                   : operand_request_i[requester_index].vl;

      // Address of the vstart element of the vector in the VRF
      // This vstart is NOT the architectural one and was modified in the lane
      // sequencer to provide the correct start address
      vrf_addr = vaddr(
          operand_request_i[requester_index].vs, NrLanes, VLEN) +
          (operand_request_i[requester_index].vstart >>
           (unsigned'(EW64) - unsigned'(operand_request_i[requester_index].eew)));

      unique case (requester_index)
        BMPUAct0, BMPUWgt0: bmpu_bank_addr_init = {operand_request_i[requester_index].bmpu_slot, 1'b0};
        BMPUAct1, BMPUWgt1: bmpu_bank_addr_init = {operand_request_i[requester_index].bmpu_slot, 1'b0} + 1'b1;
        default:  bmpu_bank_addr_init = requester_index - 5;
      endcase

      // Init helper variables
      requester_metadata_tmp = '{
          id                : operand_request_i[requester_index].id,
          addr              : ((requester_index >= BMPUAct0) && (requester_index <= BMPUWgt1)) ? bmpu_bank_addr_init :
              vrf_addr,
          len               : effective_vector_body_length,
          vew               : operand_request_i[requester_index].eew,
          bmpu_replay_en    : operand_request_i[requester_index].bmpu_replay_en,
          bmpu_self_blocks  : operand_request_i[requester_index].bmpu_self_blocks,
          bmpu_repeat_blocks: operand_request_i[requester_index].bmpu_repeat_blocks,
          bmpu_block_words  : operand_request_i[requester_index].bmpu_block_words,
          bmpu_self_idx     : '0,
          bmpu_repeat_idx   : '0,
          bmpu_word_idx     : '0,
          bmpu_base_addr    : ((requester_index >= BMPUAct0) && (requester_index <= BMPUWgt1)) ? bmpu_bank_addr_init : vrf_addr,
          hazard            : operand_request_i[requester_index].hazard,
          is_widening       : operand_request_i[requester_index].cvt_resize == CVT_WIDE,
          default: '0
      };
      operand_queue_cmd_tmp = '{
          eew       : operand_request_i[requester_index].eew,
          elem_count: effective_vector_body_length,
          conv      : operand_request_i[requester_index].conv,
          ntr_red   : operand_request_i[requester_index].cvt_resize,
          target_fu : operand_request_i[requester_index].target_fu,
          is_reduct : operand_request_i[requester_index].is_reduct
      };

      case (state_q)
        IDLE: begin : state_q_IDLE
          // Accept a new instruction
          if (operand_request_valid_i[requester_index]) begin : op_req_valid
            state_d                                    = REQUESTING;
            // Acknowledge the request
            operand_request_ready_o[requester_index]   = 1'b1;

            // Send a command to the operand queue
            operand_queue_cmd_o[requester_index]       = operand_queue_cmd_tmp;
            operand_queue_cmd_valid_o[requester_index] = 1'b1;

            // The length should be at least one after the rescaling
            if (operand_queue_cmd_o[requester_index].elem_count == '0) begin : cmd_zero_rescaled_vl
              operand_queue_cmd_o[requester_index].elem_count = 1;
            end : cmd_zero_rescaled_vl

            // Store the request
            requester_metadata_d = requester_metadata_tmp;

            // The length should be at least one after the rescaling
            if (requester_metadata_d.len == '0) begin : req_zero_rescaled_vl
              requester_metadata_d.len = 1;
            end : req_zero_rescaled_vl


            // Mute the requisition if the vl is zero
            if (operand_request_i[requester_index].vl == '0) begin : zero_vl
              state_d                                    = IDLE;
              operand_queue_cmd_valid_o[requester_index] = 1'b0;
            end : zero_vl

            request_cnt_d = '0;

`ifndef SYNTHESIS
            if (`BITFLY_OPREQ_DEBUG && (lane_id_i == '0) && (requester_index >= BMPUAct0) && (requester_index <= BMPUWgt1)) begin
              $display("[%0t][OPREQ][lane%0d] accept q=%0d id=%0d len=%0d addr=%0d vew=%0d", $time,
                       lane_id_i, requester_index, operand_request_i[requester_index].id,
                       requester_metadata_tmp.len, requester_metadata_tmp.addr,
                       operand_request_i[requester_index].eew);
            end
`endif
          end : op_req_valid
        end : state_q_IDLE

        REQUESTING: begin
          // Update waw counters
          for (int b = 0; b < NrVInsn; b++) begin : waw_counters_update
            if (vinsn_result_written_d[b]) begin : result_valid
              requester_metadata_d.waw_hazard_counter[b] = ~requester_metadata_q.waw_hazard_counter[b];
            end : result_valid
          end : waw_counters_update

          if (operand_queue_ready_i[requester_index]) begin
            automatic vlen_t num_elements;
            automatic int unsigned startup_delay;
            automatic logic bmpu_startup_hold;
            startup_delay = 0;
            bmpu_startup_hold = 1'b0;
            bmpu_word_ahead_hold  = 1'b0;
            bmpu_peer_index = requester_index;
            bmpu_progress_idx = 0;
            bmpu_peer_progress_idx = 0;
            bmpu_allowed_lead = 0;

            // Operand request
            lane_operand_req_transposed[requester_index][bank] = !stall && !bmpu_startup_hold && !bmpu_word_ahead_hold;
            operand_payload[requester_index] = '{
                addr   : requester_metadata_q.addr >> $clog2(NrBanks),
                opqueue: opqueue_e'(requester_index),
                default: '0  // this is a read operation
            };

            if (bmpu_startup_hold) begin
              request_cnt_d = request_cnt_q + 1'b1;
            end

            // Received a grant.
            if (|operand_requester_gnt) begin : op_req_grant
`ifndef SYNTHESIS
              if (`BITFLY_OPREQ_DEBUG && (requester_index >= BMPUAct0) && (requester_index <= BMPUWgt1)) begin
                $display("[%0t][OPREQ][lane%0d] grant q=%0d bank=%0d addr=%0d len_before=%0d",
                         $time, lane_id_i, requester_index, bank, requester_metadata_q.addr,
                         requester_metadata_q.len);
              end
`endif
`ifndef SYNTHESIS
              if (`BITFLY_OPREQ_DEBUG && (lane_id_i == '0) && (requester_index >= BMPUAct0) && (requester_index <= BMPUWgt1) &&
                  requester_metadata_q.bmpu_replay_en &&
                  (requester_metadata_q.bmpu_word_idx == '0)) begin
                $display("[%0t][OPREQ_BMPU_BLK][lane%0d] q=%0d bank=%0d row=%0d addr=%0d req_cnt=%0d len=%0d self=%0d repeat=%0d word=%0d base=%0d",
                         $time, lane_id_i, requester_index, bank,
                         requester_metadata_q.addr >> $clog2(NrBanks), requester_metadata_q.addr,
                         request_cnt_q, requester_metadata_q.len,
                         requester_metadata_q.bmpu_self_idx, requester_metadata_q.bmpu_repeat_idx,
                         requester_metadata_q.bmpu_word_idx, requester_metadata_q.bmpu_base_addr);
              end
              if (`BITFLY_OPREQ_DEBUG && `BITFLY_OPREQ_WGT_DEBUG && (lane_id_i == '0) &&
                  ((requester_index == BMPUWgt0) || (requester_index == BMPUWgt1)) &&
                  requester_metadata_q.bmpu_replay_en &&
                  (requester_metadata_q.bmpu_word_idx == '0)) begin
                $display("[%0t][OPREQ_WGT_BLK][lane%0d] q=%0d bank=%0d addr=%0d self=%0d repeat=%0d word=%0d block_words=%0d self_blocks=%0d repeat_blocks=%0d",
                         $time, lane_id_i, requester_index, bank, requester_metadata_q.addr,
                         requester_metadata_q.bmpu_self_idx, requester_metadata_q.bmpu_repeat_idx,
                         requester_metadata_q.bmpu_word_idx, requester_metadata_q.bmpu_block_words,
                         requester_metadata_q.bmpu_self_blocks, requester_metadata_q.bmpu_repeat_blocks);
              end
`endif
              // Bump the address pointer
              if (requester_index >= BMPUAct0 && requester_index <= BMPUWgt1) begin
                if (requester_metadata_q.bmpu_replay_en) begin
                  automatic int unsigned next_self_idx;
                  automatic int unsigned next_repeat_idx;
                  automatic int unsigned next_word_idx;
                  automatic int unsigned row_base_words;
                  next_self_idx   = requester_metadata_q.bmpu_self_idx;
                  next_repeat_idx = requester_metadata_q.bmpu_repeat_idx;
                  next_word_idx   = requester_metadata_q.bmpu_word_idx;

                  if ((requester_metadata_q.bmpu_word_idx + 1) < requester_metadata_q.bmpu_block_words) begin
                    next_word_idx = requester_metadata_q.bmpu_word_idx + 1;
                  end else begin
                    next_word_idx = 0;
                    if ((requester_index == BMPUAct0) || (requester_index == BMPUAct1)) begin
                      if ((requester_metadata_q.bmpu_repeat_idx + 1) < requester_metadata_q.bmpu_repeat_blocks) begin
                        next_repeat_idx = requester_metadata_q.bmpu_repeat_idx + 1;
                      end else begin
                        next_repeat_idx = 0;
                        if ((requester_metadata_q.bmpu_self_idx + 1) < requester_metadata_q.bmpu_self_blocks) begin
                          next_self_idx = requester_metadata_q.bmpu_self_idx + 1;
                        end
                      end
                    end else begin
                      if ((requester_metadata_q.bmpu_self_idx + 1) < requester_metadata_q.bmpu_self_blocks) begin
                        next_self_idx = requester_metadata_q.bmpu_self_idx + 1;
                      end else begin
                        next_self_idx = 0;
                        if ((requester_metadata_q.bmpu_repeat_idx + 1) < requester_metadata_q.bmpu_repeat_blocks) begin
                          next_repeat_idx = requester_metadata_q.bmpu_repeat_idx + 1;
                        end
                      end
                    end
                  end

                  requester_metadata_d.bmpu_self_idx   = next_self_idx[2:0];
                  requester_metadata_d.bmpu_repeat_idx = next_repeat_idx[2:0];
                  requester_metadata_d.bmpu_word_idx   = vlen_t'(next_word_idx);
                  row_base_words = next_self_idx * requester_metadata_q.bmpu_block_words;
                  requester_metadata_d.addr = requester_metadata_q.bmpu_base_addr +
                                              vaddr_t'((row_base_words + next_word_idx) << 3);
                end else begin
                  requester_metadata_d.addr = requester_metadata_q.addr + 4'b1000;
                end
              end else begin
                requester_metadata_d.addr = requester_metadata_q.addr + 1'b1;
              end

              // We read less than 64 bits worth of elements
              num_elements = (1 << (unsigned'(EW64) - unsigned'(requester_metadata_q.vew)));
              if (requester_metadata_q.len < num_elements) begin
                requester_metadata_d.len = 0;
              end else begin
                requester_metadata_d.len = requester_metadata_q.len - num_elements;
              end

              if (requester_metadata_q.bmpu_replay_en &&
                  ((requester_metadata_q.bmpu_word_idx + 1) >= requester_metadata_q.bmpu_block_words)) begin
                request_cnt_d = 0;
              end else begin
                request_cnt_d += 1;
              end
            end : op_req_grant

            // Finished requesting all the elements

            if (requester_metadata_d.len == '0) begin
              state_d = IDLE;
              request_cnt_d = '0;

              // Accept a new instruction
              if (operand_request_valid_i[requester_index]) begin
                state_d                                    = REQUESTING;
                // Acknowledge the request
                operand_request_ready_o[requester_index]   = 1'b1;

                // Send a command to the operand queue
                operand_queue_cmd_o[requester_index]       = operand_queue_cmd_tmp;
                operand_queue_cmd_valid_o[requester_index] = 1'b1;

                // The length should be at least one after the rescaling
                if (operand_queue_cmd_o[requester_index].elem_count == '0) begin : cmd_zero_rescaled_vl
                  operand_queue_cmd_o[requester_index].elem_count = 1;
                end : cmd_zero_rescaled_vl

                // Store the request
                requester_metadata_d = requester_metadata_tmp;

                // The length should be at least one after the rescaling
                if (requester_metadata_d.len == '0) begin : req_zero_rescaled_vl
                  requester_metadata_d.len = 1;
                end : req_zero_rescaled_vl

                // Mute the requisition if the vl is zero
                if (operand_request_i[requester_index].vl == '0) begin
                  state_d                                    = IDLE;
                  operand_queue_cmd_valid_o[requester_index] = 1'b0;
                end
              end
            end
          end
        end
      endcase
`ifndef SYNTHESIS
      if (`BITFLY_OPREQ_DEBUG && (lane_id_i == 0) && ((requester_index == BMPUAct0) || (requester_index == BMPUAct1) ||
                                (requester_index == BMPUWgt0) || (requester_index == BMPUWgt1)) &&
          (state_q == REQUESTING) && (requester_metadata_q.len <= 192) && (requester_metadata_q.len != 0)) begin
        $display("[%0t][OPREQ_TAIL][lane%0d] q=%0d addr=%0d len=%0d req_cnt=%0d self=%0d repeat=%0d word=%0d stall=%0b oq_ready=%0b issued=%0b req=%b",
                 $time, lane_id_i, requester_index, requester_metadata_q.addr, requester_metadata_q.len,
                 request_cnt_q, requester_metadata_q.bmpu_self_idx, requester_metadata_q.bmpu_repeat_idx,
                 requester_metadata_q.bmpu_word_idx, stall, operand_queue_ready_i[requester_index], operand_issued_o[requester_index], lane_operand_req_transposed[requester_index]);
      end
`endif

      // Always keep the hazard bits up to date with the global hazard table
      requester_metadata_d.hazard &= global_hazard_table_i[requester_metadata_d.id];

      // Kill all store-unit, idx, and mem-masked requests in case of exceptions
      if (lsu_ex_flush_o && (requester_index == StA || requester_index == SlideAddrGenA || requester_index == MaskM)) begin : vlsu_exception_idle
        // Reset state
        state_d = IDLE;
        // Don't wake up the store queue (redundant, as it will be flushed anyway)
        operand_queue_cmd_valid_o[requester_index] = 1'b0;
        // Clear metadata
        requester_metadata_d = '0;
        // Flush this request
        lane_operand_req_transposed[requester_index][bank] = '0;

        request_cnt_d = '0;
      end : vlsu_exception_idle
    end : operand_requester

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        state_q              <= IDLE;
        requester_metadata_q <= '0;
        request_cnt_q        <= '0;
      end else begin
        state_q              <= state_d;
        requester_metadata_q <= requester_metadata_d;
        request_cnt_q        <= request_cnt_d;
      end
    end
  end : gen_operand_requester

  ////////////////
  //  Arbiters  //
  ////////////////

  // Remember whether the VFUs are trying to write something to the VRF
  always_comb begin
    bmpu_result_granted_d = bmpu_result_granted_q;

    // Default assignment
    for (int bank = 0; bank < NrBanks; bank++) begin
      ext_operand_req[bank][VFU_Alu]       = 1'b0;
      ext_operand_req[bank][VFU_MFpu]      = 1'b0;
      ext_operand_req[bank][VFU_MaskUnit]  = 1'b0;
      ext_operand_req[bank][VFU_SlideUnit] = 1'b0;
      ext_operand_req[bank][VFU_LoadUnit]  = 1'b0;
      for (int i = 0; i < NrBmpuResultQueues; i++) begin
        ext_operand_req[bank][i+NrGlobalMasters] = 1'b0;
      end
    end

    // Generate the payloads for write back operations
    operand_payload[NrOperandQueues+VFU_Alu] = '{
        addr   : alu_result_addr_i >> $clog2(NrBanks),
        wen    : 1'b1,
        wdata  : alu_result_wdata_i,
        be     : alu_result_be_i,
        opqueue: AluA,
        default: '0
    };
    operand_payload[NrOperandQueues+VFU_MFpu] = '{
        addr   : mfpu_result_addr_i >> $clog2(NrBanks),
        wen    : 1'b1,
        wdata  : mfpu_result_wdata_i,
        be     : mfpu_result_be_i,
        opqueue: AluA,
        default: '0
    };
    operand_payload[NrOperandQueues+VFU_MaskUnit] = '{
        addr   : masku_result_addr >> $clog2(NrBanks),
        wen    : 1'b1,
        wdata  : masku_result_wdata,
        be     : masku_result_be,
        opqueue: AluA,
        default: '0
    };
    operand_payload[NrOperandQueues+VFU_SlideUnit] = '{
        addr   : sldu_result_addr >> $clog2(NrBanks),
        wen    : 1'b1,
        wdata  : sldu_result_wdata,
        be     : sldu_result_be,
        opqueue: AluA,
        default: '0
    };
    operand_payload[NrOperandQueues+VFU_LoadUnit] = '{
        addr   : ldu_result_addr_mapped >> $clog2(NrBanks),
        wen    : 1'b1,
        wdata  : ldu_result_wdata,
        be     : ldu_result_be,
        opqueue: AluA,
        default: '0
    };
    if (bmpu_output_en_q) begin
      for (int i = 0; i < NrBmpuResultQueues; i++) begin
        operand_payload[NrOperandQueues+NrGlobalMasters+i].addr = bmpu_result_addr_i[i] >>
            $clog2(NrBanks);
        operand_payload[NrOperandQueues+NrGlobalMasters+i].wen = 1'b1;
        operand_payload[NrOperandQueues+NrGlobalMasters+i].wdata = bmpu_result_wdata_i[i];
        operand_payload[NrOperandQueues+NrGlobalMasters+i].be = bmpu_result_be_i[i];
        if (i < NrBmpuResultQueues / 2) begin
          operand_payload[NrOperandQueues+NrGlobalMasters+i].opqueue = opqueue_e'(BMPUAct0 + i);
        end else begin
          operand_payload[NrOperandQueues + NrGlobalMasters + i].opqueue = opqueue_e'(BMPUWgt0 + (i - NrBmpuResultQueues / 2));
        end
      end
    end

    // Store their request value
    ext_operand_req[alu_result_addr_i[idx_width(NrBanks)-1:0]][VFU_Alu] = alu_result_req_i;
    ext_operand_req[mfpu_result_addr_i[idx_width(NrBanks)-1:0]][VFU_MFpu] = mfpu_result_req_i;
    ext_operand_req[masku_result_addr[idx_width(NrBanks)-1:0]][VFU_MaskUnit] = masku_result_req;
    ext_operand_req[sldu_result_addr[idx_width(NrBanks)-1:0]][VFU_SlideUnit] = sldu_result_req;
    ext_operand_req[ldu_result_bank_mapped][VFU_LoadUnit] = ldu_result_req;
    for (int i = 0; i < NrBmpuResultQueues; i++) begin
      ext_operand_req[bmpu_result_addr_i[i][idx_width(NrBanks)-1:0]][i+NrGlobalMasters] =
          bmpu_result_req_i && !bmpu_result_granted_q[i];
    end

    // Generate the grant signals
    alu_result_gnt_o = 1'b0;
    mfpu_result_gnt_o = 1'b0;
    masku_result_gnt = 1'b0;
    sldu_result_gnt = 1'b0;
    ldu_result_gnt = 1'b0;
    bmpu_result_gnt_o = 1'b0;
    bmpu_result_gnt_vec = '0;
    for (int bank = 0; bank < NrBanks; bank++) begin
      alu_result_gnt_o  = alu_result_gnt_o | operand_gnt[bank][NrOperandQueues + VFU_Alu];
      mfpu_result_gnt_o = mfpu_result_gnt_o | operand_gnt[bank][NrOperandQueues + VFU_MFpu];
      masku_result_gnt  = masku_result_gnt | operand_gnt[bank][NrOperandQueues + VFU_MaskUnit];
      sldu_result_gnt   = sldu_result_gnt | operand_gnt[bank][NrOperandQueues + VFU_SlideUnit];
      ldu_result_gnt    = ldu_result_gnt | operand_gnt[bank][NrOperandQueues + VFU_LoadUnit];
      for (int i = 0; i < NrBmpuResultQueues; i++) begin
        bmpu_result_gnt_vec[i] |= operand_gnt[bank][NrOperandQueues+NrGlobalMasters+i];
      end
    end
`ifndef SYNTHESIS
    if (`BITFLY_OPREQ_DEBUG && (lane_id_i == '0) && ldu_result_req && is_bmpu_load_vinsn_i[ldu_result_id] &&
        ((ldu_result_addr >> $clog2(NrBanks)) < 16)) begin
      $display("[%0t][LDU2BMPU][lane%0d] id=%0d raw_addr=%0d raw_bank=%0d raw_row=%0d map_bank=%0d map_row=%0d slotv=%0b slot=%0d is_w=%0b gnt=%0b be=%h data=%h",
               $time, lane_id_i, ldu_result_id,
               ldu_result_addr,
               ldu_result_addr[idx_width(NrBanks)-1:0],
               ldu_result_addr >> $clog2(NrBanks),
               ldu_result_bank_mapped,
               ldu_result_addr_mapped >> $clog2(NrBanks),
               bmpu_load_slot_valid_i[ldu_result_id],
               bmpu_load_slot_i[ldu_result_id],
               bmpu_load_is_weight_i[ldu_result_id],
               ldu_result_gnt,
               ldu_result_be,
               ldu_result_wdata);
    end
`endif
    if (bmpu_result_req_i) begin
      automatic logic [NrBmpuResultQueues-1:0] bmpu_granted_acc;
      bmpu_granted_acc  = bmpu_result_granted_q | bmpu_result_gnt_vec;
      bmpu_result_gnt_o = &bmpu_granted_acc;
      if (bmpu_result_gnt_o) bmpu_result_granted_d = '0;
      else bmpu_result_granted_d = bmpu_granted_acc;
    end else begin
      bmpu_result_gnt_o = 1'b0;
      bmpu_result_granted_d = '0;
    end
  end

  // Instantiate a RR arbiter per bank
  for (genvar bank = 0; bank < NrBanks; bank++) begin : gen_vrf_arbiters
    // High-priority requests
    payload_t payload_hp;
    logic payload_hp_req;
    logic payload_hp_gnt;
    rr_arb_tree #(
        .NumIn    (unsigned'(BMPUWgt1) - unsigned'(AluA) + 1 + unsigned'(VFU_MFpu) - unsigned'(VFU_Alu) + 1),
        .DataWidth($bits(payload_t)),
        .AxiVldRdy(1'b0)
    ) i_hp_vrf_arbiter (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .flush_i(1'b0),
        .rr_i('0),
        .data_i({
          operand_payload[BMPUWgt1:AluA],
          operand_payload[NrOperandQueues+VFU_MFpu:NrOperandQueues+VFU_Alu]
        }),
        .req_i({lane_operand_req[bank][BMPUWgt1:AluA], ext_operand_req[bank][VFU_MFpu:VFU_Alu]}),
        .gnt_o({
          operand_gnt[bank][BMPUWgt1:AluA],
          operand_gnt[bank][NrOperandQueues+VFU_MFpu:NrOperandQueues+VFU_Alu]
        }),
        .data_o(payload_hp),
        .idx_o(  /* Unused */),
        .req_o(payload_hp_req),
        .gnt_i(payload_hp_gnt)
    );

    // Low-priority requests
    payload_t payload_lp;
    logic payload_lp_req;
    logic payload_lp_gnt;
    rr_arb_tree #(
        .NumIn(unsigned'(SlideAddrGenA)- unsigned'(MaskB) + 1 + unsigned'(VFU_LoadUnit) - unsigned'(VFU_SlideUnit) + 1 + NrBmpuOutQueues),
        .DataWidth($bits(payload_t)),
        .AxiVldRdy(1'b0)
    ) i_lp_vrf_arbiter (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .flush_i(1'b0),
        .rr_i('0),
        .data_i({
          operand_payload[SlideAddrGenA:MaskB],
          operand_payload[NrOperandQueues+VFU_LoadUnit:NrOperandQueues+VFU_SlideUnit],
          operand_payload[NrOperandQueues+NrGlobalMasters+NrBmpuOutQueues-1:NrOperandQueues+NrGlobalMasters]
        }),
        .req_i({
          lane_operand_req[bank][SlideAddrGenA:MaskB],
          ext_operand_req[bank][VFU_LoadUnit:VFU_SlideUnit],
          ext_operand_req[bank][NrGlobalMasters+NrBmpuOutQueues-1:NrGlobalMasters]
        }),
        .gnt_o({
          operand_gnt[bank][SlideAddrGenA:MaskB],
          operand_gnt[bank][NrOperandQueues+VFU_LoadUnit:NrOperandQueues+VFU_SlideUnit],
          operand_gnt[bank][NrOperandQueues+NrGlobalMasters+NrBmpuOutQueues-1:NrOperandQueues+NrGlobalMasters]
        }),
        .data_o(payload_lp),
        .idx_o(  /* Unused */),
        .req_o(payload_lp_req),
        .gnt_i(payload_lp_gnt)
    );

    // High-priority requests always mask low-priority requests
    rr_arb_tree #(
        .NumIn    (2),
        .DataWidth($bits(payload_t)),
        .AxiVldRdy(1'b0),
        .ExtPrio  (1'b1)
    ) i_vrf_arbiter (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .flush_i(1'b0),
        .rr_i(1'b0),
        .data_i({payload_lp, payload_hp}),
        .req_i({payload_lp_req, payload_hp_req}),
        .gnt_o({payload_lp_gnt, payload_hp_gnt}),
        .data_o({
          vrf_addr_o[bank],
          vrf_wen_o[bank],
          vrf_wdata_o[bank],
          vrf_be_o[bank],
          vrf_tgt_opqueue_o[bank]
        }),
        .idx_o(  /* Unused */),
        .req_o(vrf_req_o[bank]),
        .gnt_i(vrf_req_o[bank])  // Acknowledge it directly
    );

`ifndef SYNTHESIS
    if ((bank == 4) || (bank == 5)) begin : gen_bmpu_bank_dbg
      always_ff @(posedge clk_i) begin
        if (`BITFLY_OPREQ_DEBUG && rst_ni && (lane_id_i == '0) && vrf_req_o[bank] && vrf_wen_o[bank]) begin
          $display("[%0t][OPREQ_VRF_WR][lane%0d] bank=%0d addr=%0d data=%h be=%h hp_req=%0b lp_req=%0b alu=%0b mfpu=%0b load=%0b slide=%0b mask=%0b bmpu_out=%b%b%b%b",
                   $time, lane_id_i, bank, vrf_addr_o[bank], vrf_wdata_o[bank], vrf_be_o[bank],
                   payload_hp_req, payload_lp_req,
                   ext_operand_req[bank][VFU_Alu],
                   ext_operand_req[bank][VFU_MFpu],
                   ext_operand_req[bank][VFU_LoadUnit],
                   ext_operand_req[bank][VFU_SlideUnit],
                   ext_operand_req[bank][VFU_MaskUnit],
                   ext_operand_req[bank][NrGlobalMasters+3],
                   ext_operand_req[bank][NrGlobalMasters+2],
                   ext_operand_req[bank][NrGlobalMasters+1],
                   ext_operand_req[bank][NrGlobalMasters+0]);
        end
      end
    end
`endif
  end : gen_vrf_arbiters

endmodule : operand_requester
