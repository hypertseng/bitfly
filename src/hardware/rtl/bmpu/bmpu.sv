module bmpu
  import ara_pkg::*;
  import rvv_pkg::*;
  import cf_math_pkg::idx_width;
#(
    parameter  int unsigned NrLanes         = 0,
    parameter  int unsigned VLEN            = 0,
    // Type used to address vector register file elements
    parameter  type         vaddr_t         = logic,
    parameter  type         vfu_operation_t = logic,
    // Dependant parameters. DO NOT CHANGE!
    localparam int unsigned DataWidth       = $bits(elen_t),
    localparam int unsigned StrbWidth       = DataWidth / 8,
    localparam type         strb_t          = logic         [     StrbWidth-1:0],
    localparam type         vlen_t          = logic         [$clog2(VLEN+1)-1:0],
    localparam int unsigned NrInputQueues  = 2,
    localparam int unsigned NrResultQueues  = 4
) (
    input  logic                                    clk_i,
    input  logic                                    rst_ni,
    input  logic           [idx_width(NrLanes)-1:0] lane_id_i,
    // Interface with the lane sequencer
    input  vfu_operation_t                          vfu_operation_i,
    input  logic                                    vfu_operation_valid_i,
    input  logic                                    bmpu_prefetch_ready_i,
    input  logic                                    bmpu_weight_load_i,
    output logic                                    bmpu_ready_o,
    output logic           [           NrVInsn-1:0] bmpu_insn_done_o,
    // Interface with the operand queues
    input  elen_t          [   NrInputQueues - 1:0] bmpu_act_operand_i,
    input  logic           [   NrInputQueues - 1:0] bmpu_act_operand_valid_i,
    output logic           [   NrInputQueues - 1:0] bmpu_act_operand_ready_o,
    input  elen_t          [   NrInputQueues - 1:0] bmpu_wgt_operand_i,
    input  logic           [   NrInputQueues - 1:0] bmpu_wgt_operand_valid_i,
    output logic           [   NrInputQueues - 1:0] bmpu_wgt_operand_ready_o,
    output elen_t                                   bmpu_store_data_o,
    output logic                                    bmpu_store_valid_o,
    input  logic                                    bmpu_store_ready_i,
    // Interface with the vector register file
    output logic                                    bmpu_result_req_o,
    output vid_t           [                   NrResultQueues - 1:0] bmpu_result_id_o,
    output vaddr_t         [                   NrResultQueues - 1:0] bmpu_result_addr_o,
    output elen_t          [                   NrResultQueues - 1:0] bmpu_result_wdata_o,
    output strb_t          [                   NrResultQueues - 1:0] bmpu_result_be_o,
    input  logic                                    bmpu_result_gnt_i,
    output logic                                    bmpu_compute_busy_o,
    output logic                                    bmpu_output_en_o
);

  import cf_math_pkg::idx_width;
  `include "common_cells/registers.svh"

  logic unused_bmpu_side_inputs;
  assign unused_bmpu_side_inputs = bmpu_prefetch_ready_i ^ bmpu_weight_load_i;

  /////////////
  // Lane ID //
  /////////////

  // Lane 0 has different logic than Lanes != 0
  // A parameter would be perfect to save HW, but our hierarchical
  // synth/pnr flow needs that all lanes are the same

  ////////////////////////////////
  //  Vector instruction queue  //
  ////////////////////////////////

  // We store a certain number of in-flight vector instructions
  localparam VInsnQueueDepth = BmpuInsnQueueDepth;

  struct packed {
    vfu_operation_t [VInsnQueueDepth-1:0] vinsn;

    // Each instruction can be in one of the three execution phases.
    // - Being accepted (i.e., it is being stored for future execution in this
    //   vector functional unit).
    // - Being issued (i.e., its micro-operations are currently being issued
    //   to the corresponding functional units).
    // - Being committed (i.e., its results are being written to the vector
    //   register file).
    // We need pointers to index which instruction is at each execution phase
    // between the VInsnQueueDepth instructions in memory.
    logic [idx_width(VInsnQueueDepth)-1:0] accept_pnt;
    logic [idx_width(VInsnQueueDepth)-1:0] issue_pnt;
    logic [idx_width(VInsnQueueDepth)-1:0] commit_pnt;

    // We also need to count how many instructions are queueing to be
    // issued/committed, to avoid accepting more instructions than
    // we can handle.
    logic [idx_width(VInsnQueueDepth):0] issue_cnt;
    logic [idx_width(VInsnQueueDepth):0] commit_cnt;
  }
      vinsn_queue_d, vinsn_queue_q;

  // Is the vector instruction queue full?
  logic vinsn_queue_full;
  assign vinsn_queue_full = (vinsn_queue_q.commit_cnt == VInsnQueueDepth);

  // Do we have a vector instruction ready to be issued?
  vfu_operation_t vinsn_issue_d, vinsn_issue_q;
  logic vinsn_issue_valid;
  assign vinsn_issue_d     = vinsn_queue_d.vinsn[vinsn_queue_d.issue_pnt];
  assign vinsn_issue_valid = (vinsn_queue_q.issue_cnt != '0);

  // Do we have a vector instruction with results being committed?
  vfu_operation_t vinsn_commit;
  logic           vinsn_commit_valid;
  assign vinsn_commit       = vinsn_queue_q.vinsn[vinsn_queue_q.commit_pnt];
  assign vinsn_commit_valid = (vinsn_queue_q.commit_cnt != '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      vinsn_queue_q <= '0;
      vinsn_issue_q <= '0;
    end else begin
      vinsn_queue_q <= vinsn_queue_d;
      vinsn_issue_q <= vinsn_issue_d;
    end
  end

  ////////////////////
  //  Result queue  //
  ////////////////////

  localparam int unsigned ResultQueueDepth = 2;

  // There is a result queue per VFU, holding the results that were not
  // yet accepted by the corresponding lane.
  typedef struct packed {
    vid_t   id;
    vaddr_t addr;
    elen_t  wdata;
    strb_t  be;
  } payload_t;

  // Result queue
  payload_t [ResultQueueDepth-1:0][NrResultQueues-1:0] result_queue_d, result_queue_q;
  logic [ResultQueueDepth-1:0] result_queue_valid_d, result_queue_valid_q;
  // We need two pointers in the result queue. One pointer to indicate with `payload_t` we are
  // currently writing into (write_pnt), and one pointer to indicate which `payload_t` we are
  // currently reading from and writing into the lanes (read_pnt).
  logic [idx_width(ResultQueueDepth)-1:0] result_queue_write_pnt_d, result_queue_write_pnt_q;
  logic [idx_width(ResultQueueDepth)-1:0] result_queue_read_pnt_d, result_queue_read_pnt_q;
  // We need to count how many valid elements are there in this result queue.
  logic [idx_width(ResultQueueDepth):0] result_queue_cnt_d, result_queue_cnt_q;
  logic [$clog2(NrResultQueues)-1:0] store_word_pnt_d, store_word_pnt_q;

  // Is the result queue full?
  logic result_queue_full;
  assign result_queue_full = (result_queue_cnt_q == ResultQueueDepth);

  always_ff @(posedge clk_i or negedge rst_ni) begin : p_result_queue_ff
    if (!rst_ni) begin
      result_queue_q           <= '0;
      result_queue_valid_q     <= '0;
      result_queue_write_pnt_q <= '0;
      result_queue_read_pnt_q  <= '0;
      result_queue_cnt_q       <= '0;
      store_word_pnt_q         <= '0;
    end else begin
      result_queue_q           <= result_queue_d;
      result_queue_valid_q     <= result_queue_valid_d;
      result_queue_write_pnt_q <= result_queue_write_pnt_d;
      result_queue_read_pnt_q  <= result_queue_read_pnt_d;
      result_queue_cnt_q       <= result_queue_cnt_d;
      store_word_pnt_q         <= store_word_pnt_d;
    end
  end

  localparam int unsigned MaxBmpuContexts   = 8;
  localparam int unsigned MaxBmpuGroupElems = 1024;
  localparam int unsigned CtxIdxWidth       = (MaxBmpuContexts <= 1) ? 1 : $clog2(MaxBmpuContexts);

  logic bmpu_valid;
  logic bmpu_clear;
  logic sa_ctx_clear;
  logic compute_active_d, compute_active_q;
  logic epoch_active_d, epoch_active_q;
  logic first_k_round_d, first_k_round_q;
  logic [127:0] sa_result [NrInputQueues-1:0];
  logic [127:0] bmpu_result [NrInputQueues-1:0];
  logic [NrInputQueues-1:0] sa_act_operand_ready;
  logic [NrInputQueues-1:0] sa_wgt_operand_ready;
  logic [16:0] k_dim_d, k_dim_q;
  logic [2:0] prec_d, prec_q;
  logic [2:0] gm_d, gm_q, gn_d, gn_q;
  logic [3:0] ctx_count_d, ctx_count_q;
  logic [2:0] m_block_count_d, m_block_count_q, n_block_count_d, n_block_count_q;
  logic [2:0] compute_ai_d, compute_ai_q, compute_wi_d, compute_wi_q;
  logic [2:0] compute_mblock_d, compute_mblock_q, compute_nblock_d, compute_nblock_q;
  logic [2:0] store_ai_d, store_ai_q, store_wi_d, store_wi_q;
  logic [2:0] store_mblock_d, store_mblock_q, store_nblock_d, store_nblock_q;
  logic [0:0] store_col_d, store_col_q;
  logic [CtxIdxWidth-1:0] sa_compute_ctx_id, sa_output_ctx_id;
  logic sa_done_d, sa_done_q;
`ifndef SYNTHESIS
  logic dbg_compute_active_q;
  logic dbg_epoch_active_q;
  logic dbg_bmpu_valid_q;
`endif

  assign sa_compute_ctx_id = (compute_mblock_q * n_block_count_q) + compute_nblock_q;
  assign sa_output_ctx_id  = (store_mblock_q * n_block_count_q) + store_nblock_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sa_done_q        <= 1'b0;
      compute_active_q <= 1'b0;
      epoch_active_q   <= 1'b0;
      first_k_round_q  <= 1'b1;
      k_dim_q          <= '0;
      prec_q           <= '0;
      gm_q             <= 3'd1;
      gn_q             <= 3'd1;
      ctx_count_q      <= 4'd1;
      m_block_count_q  <= 3'd1;
      n_block_count_q  <= 3'd1;
      compute_ai_q     <= '0;
      compute_wi_q     <= '0;
      compute_mblock_q <= '0;
      compute_nblock_q <= '0;
      store_ai_q       <= '0;
      store_wi_q       <= '0;
      store_mblock_q   <= '0;
      store_nblock_q   <= '0;
      store_col_q      <= '0;
    end else begin
      sa_done_q        <= sa_done_d;
      compute_active_q <= compute_active_d;
      epoch_active_q   <= epoch_active_d;
      first_k_round_q  <= first_k_round_d;
      k_dim_q          <= k_dim_d;
      prec_q           <= prec_d;
      gm_q             <= gm_d;
      gn_q             <= gn_d;
      ctx_count_q      <= ctx_count_d;
      m_block_count_q  <= m_block_count_d;
      n_block_count_q  <= n_block_count_d;
      compute_ai_q     <= compute_ai_d;
      compute_wi_q     <= compute_wi_d;
      compute_mblock_q <= compute_mblock_d;
      compute_nblock_q <= compute_nblock_d;
      store_ai_q       <= store_ai_d;
      store_wi_q       <= store_wi_d;
      store_mblock_q   <= store_mblock_d;
      store_nblock_q   <= store_nblock_d;
      store_col_q      <= store_col_d;
    end
  end

  sa #(
      .ROWS      (NrInputQueues),
      .COLS      (NrInputQueues),
      .BIT_ACT   (8),
      .BIT_WEIGHT(1),
      .CTX_MAX   (MaxBmpuContexts)
  ) u_sa (
      .clk_i             (clk_i),
      .rst_ni            (rst_ni),
      .valid_i           (bmpu_valid),
      .clear_i           (bmpu_clear),
      .ctx_clear_i       (sa_ctx_clear),
      .ctx_id_i          (sa_compute_ctx_id),
      .output_ctx_id_i   (sa_output_ctx_id),
      .output_col_id_i   (store_col_q),
      .output_en_i       (bmpu_output_en_o),
      .bmpu_act_operand_i(bmpu_act_operand_i),
      .bmpu_act_operand_valid_i(bmpu_act_operand_valid_i),
      .bmpu_wgt_operand_i(bmpu_wgt_operand_i),
      .bmpu_wgt_operand_valid_i(bmpu_wgt_operand_valid_i),
      .k_dim_i           (k_dim_q),
      .prec_i            (prec_q),
      .output_data_o     (sa_result),
      .bmpu_act_operand_ready_o(sa_act_operand_ready),
      .bmpu_wgt_operand_ready_o(sa_wgt_operand_ready),
      .sa_done_o         (sa_done_d)
  );

  ///////////////
  //  Control  //
  ///////////////

  // Remaining elements of the current instruction in the issue phase
  vlen_t issue_cnt_d, issue_cnt_q;
  // Remaining elements of the current instruction in the commit phase
  vlen_t commit_cnt_d, commit_cnt_q;

  // How many elements are issued/committed
  logic [5:0] element_cnt_buf_issue, element_cnt_buf_commit;
  logic [1:0] issue_effective_eew, commit_effective_eew;
  logic [8:0] element_cnt_issue;
  logic [8:0] element_cnt_commit;

  always_comb begin : p_bmpu
    // Maintain state
    vinsn_queue_d            = vinsn_queue_q;
    issue_cnt_d              = issue_cnt_q;
    commit_cnt_d             = commit_cnt_q;
    k_dim_d                  = k_dim_q;
    prec_d                   = prec_q;
    gm_d                     = gm_q;
    gn_d                     = gn_q;
    ctx_count_d              = ctx_count_q;
    m_block_count_d          = m_block_count_q;
    n_block_count_d          = n_block_count_q;
    compute_ai_d             = compute_ai_q;
    compute_wi_d             = compute_wi_q;
    compute_mblock_d         = compute_mblock_q;
    compute_nblock_d         = compute_nblock_q;
    store_ai_d               = store_ai_q;
    store_wi_d               = store_wi_q;
    store_mblock_d           = store_mblock_q;
    store_nblock_d           = store_nblock_q;
    store_col_d              = store_col_q;
    epoch_active_d           = epoch_active_q;
    first_k_round_d          = first_k_round_q;

    result_queue_d           = result_queue_q;
    result_queue_valid_d     = result_queue_valid_q;
    result_queue_read_pnt_d  = result_queue_read_pnt_q;
    result_queue_write_pnt_d = result_queue_write_pnt_q;
    result_queue_cnt_d       = result_queue_cnt_q;
    store_word_pnt_d         = store_word_pnt_q;
    // Inform our status to the lane controller
    bmpu_ready_o              = !vinsn_queue_full &&
                                (!vfu_operation_valid_i || !vfu_operation_i.bmpu_en || bmpu_prefetch_ready_i);

    bmpu_insn_done_o          = '0;
    bmpu_valid                = 1'b0;
    bmpu_clear                = 1'b0;
    sa_ctx_clear              = 1'b0;
    bmpu_output_en_o          = 1'b0;
    compute_active_d          = compute_active_q;

    // Do not acknowledge any operands
    bmpu_act_operand_ready_o  = '0;
    bmpu_wgt_operand_ready_o  = '0;

    // How many elements are we processing this cycle?
    issue_effective_eew      = unsigned'(vinsn_issue_q.vtype.vsew[1:0]);
    element_cnt_buf_issue    = NrResultQueues * (1 << (unsigned'(EW64) - issue_effective_eew));
    element_cnt_issue        = {2'b0, element_cnt_buf_issue};

    commit_effective_eew     = unsigned'(vinsn_commit.vtype.vsew[1:0]);
    element_cnt_buf_commit   = NrResultQueues * (1 << (unsigned'(EW64) - commit_effective_eew));
    element_cnt_commit       = {2'b0, element_cnt_buf_commit};

    if ((vinsn_issue_valid && (vinsn_issue_q.op == BMPSE)) ||
        (vinsn_commit_valid && (vinsn_commit.op == BMPSE))) begin
      bmpu_output_en_o = 1'b1;
    end


    for (int unsigned i = 0; i < NrInputQueues; i++) begin
      bmpu_result[i] = sa_result[i];
    end

    if (sa_done_q) begin  // sa computation done
      automatic logic last_compute_block;
      last_compute_block = (compute_nblock_q + 3'd1 >= n_block_count_q)
                        && (compute_mblock_q + 3'd1 >= m_block_count_q);
      if (1'b0 && lane_id_i == 0) begin
        if (1'b0) $display("[%0t][BMPU_SA] sa_done c=(ai=%0d,wi=%0d,mb=%0d,nb=%0d) s=(ai=%0d,wi=%0d,mb=%0d,nb=%0d,col=%0d) gm=%0d gn=%0d mbc=%0d nbc=%0d last=%0b issue=%0d commit=%0d sa0=%h sa1=%h",
                 $time, compute_ai_q, compute_wi_q, compute_mblock_q, compute_nblock_q,
                 store_ai_q, store_wi_q, store_mblock_q, store_nblock_q, store_col_q,
                 gm_q, gn_q, m_block_count_q, n_block_count_q, last_compute_block, issue_cnt_q, commit_cnt_q,
                 sa_result[0], sa_result[1]);
      end
      if (last_compute_block) begin
        // Finished all physical blocks of this BMPMM instruction.
        vinsn_queue_d.issue_cnt -= 1;
        if (vinsn_queue_q.issue_pnt == VInsnQueueDepth - 1) vinsn_queue_d.issue_pnt = '0;
        else vinsn_queue_d.issue_pnt = vinsn_queue_q.issue_pnt + 1;

        if (vinsn_queue_d.issue_cnt != 0)
          issue_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.issue_pnt].vl;

        vinsn_queue_d.commit_cnt -= 1;
        if (vinsn_queue_d.commit_pnt == VInsnQueueDepth - 1) vinsn_queue_d.commit_pnt = '0;
        else vinsn_queue_d.commit_pnt = vinsn_queue_q.commit_pnt + 1;

        if (vinsn_queue_d.commit_cnt != '0)
          commit_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.commit_pnt].vl;

        bmpu_insn_done_o[vinsn_commit.id] = 1'b1;
      end
    end

    ////////////////////////////////////////
    //  Write data into the result queue  //
    ////////////////////////////////////////
    if (!sa_done_q && vinsn_issue_valid) begin
      // Do not accept operands if the result queue is full!
      if (!result_queue_full) begin
        if (vinsn_issue_q.op == BMPSE) begin
          bmpu_valid = 1'b1;
        end else begin
          if ((|bmpu_act_operand_valid_i) && (|bmpu_wgt_operand_valid_i)) begin
            if (1'b0 && (lane_id_i == 0) && !compute_active_q) begin
              if (1'b0) $display("[%0t][BMPU_GO] c=(ai=%0d,wi=%0d,mb=%0d,nb=%0d) actv=%b wgtv=%b first_k=%0b epoch=%0b",
                       $time, compute_ai_q, compute_wi_q, compute_mblock_q, compute_nblock_q,
                       bmpu_act_operand_valid_i, bmpu_wgt_operand_valid_i, first_k_round_q, epoch_active_q);
            end
            if (!epoch_active_q) begin
              epoch_active_d  = 1'b1;
              first_k_round_d = 1'b1;
              compute_ai_d     = '0;
              compute_wi_d     = '0;
              compute_mblock_d = '0;
              compute_nblock_d = '0;
              store_ai_d       = '0;
              store_wi_d       = '0;
              store_mblock_d   = '0;
              store_nblock_d   = '0;
              store_col_d      = '0;
            end
            sa_ctx_clear = !compute_active_q && (!epoch_active_q || first_k_round_q);
            compute_active_d = 1'b1;

            // Only consume operands when the SA actually injects a new k-slice.
            bmpu_act_operand_ready_o = sa_act_operand_ready;
            bmpu_wgt_operand_ready_o = sa_wgt_operand_ready;
          end

          if (compute_active_q || ((|bmpu_act_operand_valid_i) && (|bmpu_wgt_operand_valid_i))) begin
            bmpu_valid = 1'b1;
          end
        end
        if (bmpu_valid && bmpu_output_en_o) begin
          // How many elements are we committing with this word?
          automatic logic [8:0] element_cnt = element_cnt_issue;

          if (element_cnt > issue_cnt_q) element_cnt = issue_cnt_q;
          // Pack each 64b lane word as two columns x two local rows:
          // {col1,row1, col1,row0, col0,row1, col0,row0}.
          // The VSTU BMPU-specific byte mapper then reorders lane data into
          // global column-major row order across all lanes.
          result_queue_d[result_queue_write_pnt_q][0].wdata = {
              bmpu_result[1][16 +: 16], bmpu_result[0][16 +: 16],
              bmpu_result[1][ 0 +: 16], bmpu_result[0][ 0 +: 16]};
          result_queue_d[result_queue_write_pnt_q][1].wdata = {
              bmpu_result[1][48 +: 16], bmpu_result[0][48 +: 16],
              bmpu_result[1][32 +: 16], bmpu_result[0][32 +: 16]};
          result_queue_d[result_queue_write_pnt_q][2].wdata = {
              bmpu_result[1][80 +: 16], bmpu_result[0][80 +: 16],
              bmpu_result[1][64 +: 16], bmpu_result[0][64 +: 16]};
          result_queue_d[result_queue_write_pnt_q][3].wdata = {
              bmpu_result[1][112 +: 16], bmpu_result[0][112 +: 16],
              bmpu_result[1][96 +: 16],  bmpu_result[0][96 +: 16]};
          for (int unsigned i = 0; i < NrResultQueues; i++) begin
            result_queue_d[result_queue_write_pnt_q][i].addr  = (k_dim_q * (4'b1000 << EW8) / DataWidth) * NrVRFBanksPerLane + ((vinsn_issue_q.vl - issue_cnt_q) >> (unsigned'(EW64) - unsigned'(vinsn_issue_q.vtype.vsew))) + i;
            result_queue_d[result_queue_write_pnt_q][i].id = vinsn_issue_q.id;
            result_queue_d[result_queue_write_pnt_q][i].be = '1;
          end

          if (vinsn_issue_q.op == BMPSE) begin
            if (store_col_q + 1'b1 < NrInputQueues) begin
              store_col_d = store_col_q + 1'b1;
            end else begin
              store_col_d = '0;
              if (store_nblock_q + 3'd1 < n_block_count_q) begin
                store_nblock_d = store_nblock_q + 3'd1;
              end else begin
                store_nblock_d = '0;
                if (store_mblock_q + 3'd1 < m_block_count_q) begin
                  store_mblock_d = store_mblock_q + 3'd1;
                end else begin
                  store_mblock_d = '0;
                end
              end
            end
          end
          // Bump pointers and counters of the result queue
          result_queue_valid_d[result_queue_write_pnt_q] = 1'b1;
          result_queue_cnt_d += 1;
          if (result_queue_write_pnt_q == ResultQueueDepth - 1) result_queue_write_pnt_d = 0;
          else result_queue_write_pnt_d = result_queue_write_pnt_q + 1;
          issue_cnt_d = issue_cnt_q - element_cnt;

          // Finished issuing the micro-operations of this vector instruction
          if (vinsn_issue_valid && issue_cnt_d == '0) begin
            // Bump issue counter and pointers
            vinsn_queue_d.issue_cnt -= 1;
            if (vinsn_queue_q.issue_pnt == VInsnQueueDepth - 1) vinsn_queue_d.issue_pnt = '0;
            else vinsn_queue_d.issue_pnt = vinsn_queue_q.issue_pnt + 1;

            // Assign vector length for next instruction in the instruction queue
            if (vinsn_queue_d.issue_cnt != 0)
              issue_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.issue_pnt].vl;
          end
        end
      end
    end

    ////////////////////////////////////////
    //  Stream results into the Store Unit  //
    ////////////////////////////////////////

    bmpu_result_req_o = 1'b0;
    for (int unsigned i = 0; i < NrResultQueues / 2; i++) begin
      bmpu_result_wdata_o[i] = '0;
      bmpu_result_wdata_o[i + (NrResultQueues / 2)] = '0;
      bmpu_result_addr_o[i] = '0;
      bmpu_result_addr_o[i + (NrResultQueues / 2)] = '0;
      bmpu_result_id_o[i] = '0;
      bmpu_result_id_o[i + (NrResultQueues / 2)] = '0;
      bmpu_result_be_o[i] = '0;
      bmpu_result_be_o[i + (NrResultQueues / 2)] = '0;
    end

    bmpu_store_valid_o = result_queue_valid_q[result_queue_read_pnt_q] &&
                         vinsn_commit_valid && (vinsn_commit.op == BMPSE);
    bmpu_store_data_o  = result_queue_q[result_queue_read_pnt_q][store_word_pnt_q].wdata;

    if (bmpu_store_valid_o && bmpu_store_ready_i) begin
      automatic logic [8:0] element_cnt = element_cnt_commit;
      if (store_word_pnt_q == NrResultQueues - 1) begin
        result_queue_valid_d[result_queue_read_pnt_q] = 1'b0;
        result_queue_d[result_queue_read_pnt_q]       = '0;
        store_word_pnt_d = '0;

        if (result_queue_read_pnt_q == ResultQueueDepth - 1) result_queue_read_pnt_d = 0;
        else result_queue_read_pnt_d = result_queue_read_pnt_q + 1;

        result_queue_cnt_d -= 1;

        if (commit_cnt_q < element_cnt) begin
          commit_cnt_d = '0;
        end else begin
          commit_cnt_d = commit_cnt_q - element_cnt;
        end
      end else begin
        store_word_pnt_d = store_word_pnt_q + 1'b1;
      end
    end

    // Finished committing the results of a vector instruction
    if (vinsn_commit_valid && (commit_cnt_d == '0)) begin
      // Update the commit counters and pointers
      vinsn_queue_d.commit_cnt -= 1;
      if (vinsn_queue_d.commit_pnt == VInsnQueueDepth - 1) vinsn_queue_d.commit_pnt = '0;
      else vinsn_queue_d.commit_pnt += 1;

      // Update the commit counter for the next instruction
      if (vinsn_queue_d.commit_cnt != '0)
        commit_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.commit_pnt].vl;

      if (vinsn_commit.op == BMPSE) begin
        if ((store_col_d == '0) && (store_nblock_d == '0) && (store_mblock_d == '0)) begin
          store_ai_d      = '0;
          store_wi_d      = '0;
          epoch_active_d  = 1'b0;
          first_k_round_d = 1'b1;
          compute_ai_d    = '0;
          compute_wi_d    = '0;
        end
      end

      bmpu_insn_done_o[vinsn_commit.id] = 1'b1;
    end

    //////////////////////////////
    //  Accept new instruction  //
    //////////////////////////////

    if (!vinsn_queue_full && vfu_operation_valid_i &&
        ((vfu_operation_i.vfu == BMPU) || vfu_operation_i.op == BMPSE) &&
        ((vfu_operation_i.vfu != BMPU) || bmpu_prefetch_ready_i)) begin
      vinsn_queue_d.vinsn[vinsn_queue_q.accept_pnt] = vfu_operation_i;

      // Initialize counters and alu state if the instruction queue was empty
      if (vinsn_queue_d.issue_cnt == '0) begin
        issue_cnt_d = vfu_operation_i.vl;
      end
      if (vinsn_queue_d.commit_cnt == '0) commit_cnt_d = vfu_operation_i.vl;

      k_dim_d     = vfu_operation_i.k_dim;
      prec_d      = vfu_operation_i.prec;
      gm_d        = vfu_operation_i.gm;
      gn_d        = vfu_operation_i.gn;
      m_block_count_d = (vfu_operation_i.mtile[5:3] == '0) ? 3'd1 : vfu_operation_i.mtile[5:3];
      n_block_count_d = (vfu_operation_i.ntile[6:4] == '0) ? 3'd1 : vfu_operation_i.ntile[6:4];
      ctx_count_d = vfu_operation_i.group_g;

      // Bump pointers and counters of the vector instruction queue
      vinsn_queue_d.accept_pnt += 1;
      vinsn_queue_d.issue_cnt += 1;
      vinsn_queue_d.commit_cnt += 1;

      if ((vfu_operation_i.vfu == BMPU) && (vfu_operation_i.op != BMPSE)) begin
        bmpu_output_en_o = 1'b0;
        bmpu_valid = 1'b0;
      end
    end

    if (sa_done_q) begin
      compute_active_d = 1'b0;
      bmpu_valid = 1'b0;

      if (compute_nblock_q + 3'd1 < n_block_count_q) begin
        compute_nblock_d = compute_nblock_q + 3'd1;
      end else begin
        compute_nblock_d = '0;
        if (compute_mblock_q + 3'd1 < m_block_count_q) begin
          compute_mblock_d = compute_mblock_q + 3'd1;
        end else begin
          compute_mblock_d = '0;
          compute_ai_d = '0;
          compute_wi_d = '0;
          if (first_k_round_q) first_k_round_d = 1'b0;
        end
      end
    end


  end : p_bmpu

  assign bmpu_compute_busy_o = bmpu_valid | (vinsn_queue_q.issue_cnt != '0) | (vinsn_queue_q.commit_cnt != '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      issue_cnt_q  <= '0;
      commit_cnt_q <= '0;
    end else begin
      issue_cnt_q  <= issue_cnt_d;
      commit_cnt_q <= commit_cnt_d;
    end
  end


endmodule
