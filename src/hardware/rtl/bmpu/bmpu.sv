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
    localparam int unsigned NrInputQueues  = 4,
    localparam int unsigned NrResultQueues  = 8
) (
    input  logic                                    clk_i,
    input  logic                                    rst_ni,
    input  logic           [idx_width(NrLanes)-1:0] lane_id_i,
    // Interface with the lane sequencer
    input  vfu_operation_t                          vfu_operation_i,
    input  logic                                    vfu_operation_valid_i,
    output logic                                    bmpu_ready_o,
    output logic           [           NrVInsn-1:0] bmpu_insn_done_o,
    // Interface with the operand queues
    input  elen_t          [   NrInputQueues - 1:0] bmpu_act_operand_i,
    input  logic           [   NrInputQueues - 1:0] bmpu_act_operand_valid_i,
    output logic           [   NrInputQueues - 1:0] bmpu_act_operand_ready_o,
    input  elen_t          [   NrInputQueues - 1:0] bmpu_wgt_operand_i,
    input  logic           [   NrInputQueues - 1:0] bmpu_wgt_operand_valid_i,
    output logic           [   NrInputQueues - 1:0] bmpu_wgt_operand_ready_o,
    // Interface with the vector register file
    output logic                                    bmpu_result_req_o,
    output vid_t           [                   NrResultQueues - 1:0] bmpu_result_id_o,
    output vaddr_t         [                   NrResultQueues - 1:0] bmpu_result_addr_o,
    output elen_t          [                   NrResultQueues - 1:0] bmpu_result_wdata_o,
    output strb_t          [                   NrResultQueues - 1:0] bmpu_result_be_o,
    input  logic                                    bmpu_result_gnt_i,
    output logic                                    bmpu_output_en_o
);

  import cf_math_pkg::idx_width;
  `include "common_cells/registers.svh"

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
  payload_t [ResultQueueDepth-1:0][7:0] result_queue_d, result_queue_q;
  logic [ResultQueueDepth-1:0][7:0] result_queue_valid_d, result_queue_valid_q;
  // We need two pointers in the result queue. One pointer to indicate with `payload_t` we are
  // currently writing into (write_pnt), and one pointer to indicate which `payload_t` we are
  // currently reading from and writing into the lanes (read_pnt).
  logic [idx_width(ResultQueueDepth)-1:0][7:0] result_queue_write_pnt_d, result_queue_write_pnt_q;
  logic [idx_width(ResultQueueDepth)-1:0][7:0] result_queue_read_pnt_d, result_queue_read_pnt_q;
  // We need to count how many valid elements are there in this result queue.
  logic [idx_width(ResultQueueDepth):0] result_queue_cnt_d, result_queue_cnt_q;

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
    end else begin
      result_queue_q           <= result_queue_d;
      result_queue_valid_q     <= result_queue_valid_d;
      result_queue_write_pnt_q <= result_queue_write_pnt_d;
      result_queue_read_pnt_q  <= result_queue_read_pnt_d;
      result_queue_cnt_q       <= result_queue_cnt_d;
    end
  end

  logic bmpu_valid;
  logic [127:0] bmpu_result [3:0];
  logic [16:0] k_dim;
  logic [2:0] prec;
  logic sa_done_d, sa_done_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sa_done_q <= 1'b0;
    end else begin
      sa_done_q <= sa_done_d;
    end
  end

  sa #(
      .ROWS      (4),
      .COLS      (4),
      .BIT_ACT   (8),
      .BIT_WEIGHT(1)
  ) u_sa (
      .clk_i            (clk_i),
      .rst_ni           (rst_ni),
      .valid_i          (bmpu_valid),
      .output_en_i      (bmpu_output_en_o),
      .bmpu_act_operand_i(bmpu_act_operand_i),
      .bmpu_wgt_operand_i(bmpu_wgt_operand_i),
      .k_dim_i          (k_dim),
      .prec_i           (prec),
      .output_data_o    (bmpu_result),
      .sa_done_o        (sa_done_d)
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

    result_queue_d           = result_queue_q;
    result_queue_valid_d     = result_queue_valid_q;
    result_queue_read_pnt_d  = result_queue_read_pnt_q;
    result_queue_write_pnt_d = result_queue_write_pnt_q;
    result_queue_cnt_d       = result_queue_cnt_q;

    // Inform our status to the lane controller
    bmpu_ready_o              = !vinsn_queue_full;
    bmpu_insn_done_o          = '0;

    // Do not acknowledge any operands
    bmpu_act_operand_ready_o  = '0;
    bmpu_wgt_operand_ready_o  = '0;

    // How many elements are we processing this cycle?
    issue_effective_eew      = unsigned'(vinsn_issue_q.vtype.vsew[1:0]);
    element_cnt_buf_issue    = 8 * (1 << (unsigned'(EW64) - issue_effective_eew));
    element_cnt_issue        = {2'b0, element_cnt_buf_issue};

    commit_effective_eew     = unsigned'(vinsn_commit.vtype.vsew[1:0]);
    element_cnt_buf_commit   = 8 * (1 << (unsigned'(EW64) - commit_effective_eew));
    element_cnt_commit       = {2'b0, element_cnt_buf_commit};

    if (vinsn_issue_q.op == BMPSE) begin
      bmpu_output_en_o = 1'b1;
    end

    if (sa_done_q) begin  // sa computation done
      // We have finished the computation of the micro-operations of this vector instruction
      // issue
      vinsn_queue_d.issue_cnt -= 1;
      if (vinsn_queue_q.issue_pnt == VInsnQueueDepth - 1) vinsn_queue_d.issue_pnt = '0;
      else vinsn_queue_d.issue_pnt = vinsn_queue_q.issue_pnt + 1;

      // Assign vector length for next instruction in the instruction queue
      if (vinsn_queue_d.issue_cnt != 0)
        issue_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.issue_pnt].vl;

      // commit
      vinsn_queue_d.commit_cnt -= 1;
      if (vinsn_queue_d.commit_pnt == VInsnQueueDepth - 1) vinsn_queue_d.commit_pnt = '0;
      else vinsn_queue_d.commit_pnt += 1;

      // Update the commit counter for the next instruction
      if (vinsn_queue_d.commit_cnt != '0)
        commit_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.commit_pnt].vl;
    end

    ////////////////////////////////////////
    //  Write data into the result queue  //
    ////////////////////////////////////////
    if (vinsn_issue_valid) begin
      // Do not accept operands if the result queue is full!
      if (!result_queue_full) begin
        // Do we have all the operands necessary for this instruction?
        if ((|bmpu_act_operand_valid_i) && (|bmpu_wgt_operand_valid_i)) begin
          // Issue the operation
          bmpu_valid = 1'b1;

          // Acknowledge the operands of this instruction
          bmpu_act_operand_ready_o = '1;
          bmpu_wgt_operand_ready_o = '1;
        end
        if (bmpu_output_en_o) begin
          // How many elements are we committing with this word?
          automatic logic [8:0] element_cnt = element_cnt_issue;

          if (element_cnt > issue_cnt_q) element_cnt = issue_cnt_q;
          // Store the result in the result queue
          for (int unsigned i = 0; i < NrResultQueues / 2; i++) begin
            result_queue_d[result_queue_write_pnt_q][i].wdata = bmpu_result[i][63:0];
            result_queue_d[result_queue_write_pnt_q][i + (NrResultQueues / 2)].wdata = bmpu_result[i][127:64];
            result_queue_d[result_queue_write_pnt_q][i].addr  = (k_dim * (4'b1000 << EW8) / DataWidth) * NrVRFBanksPerLane + ((vinsn_issue_q.vl - issue_cnt_q) >> (unsigned'(EW64) - unsigned'(vinsn_issue_q.vtype.vsew))) + i;
            result_queue_d[result_queue_write_pnt_q][i + (NrResultQueues / 2)].addr  = (k_dim * (4'b1000 << EW8) / DataWidth) * NrVRFBanksPerLane + ((vinsn_issue_q.vl - issue_cnt_q) >> (unsigned'(EW64) - unsigned'(vinsn_issue_q.vtype.vsew))) + i + (NrResultQueues / 2);
            result_queue_d[result_queue_write_pnt_q][i].id = vinsn_issue_q.id;
            result_queue_d[result_queue_write_pnt_q][i + (NrResultQueues / 2)].id = vinsn_issue_q.id;
            result_queue_d[result_queue_write_pnt_q][i].be = '1;
            result_queue_d[result_queue_write_pnt_q][i + (NrResultQueues / 2)].be = '1;
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

    //////////////////////////////////
    //  Write results into the VRF  //
    //////////////////////////////////

    bmpu_result_req_o = result_queue_valid_q[result_queue_read_pnt_q] && bmpu_output_en_o;
    for (int unsigned i = 0; i < NrResultQueues / 2; i++) begin
      bmpu_result_wdata_o[i] = result_queue_q[result_queue_read_pnt_q][i].wdata;
      bmpu_result_wdata_o[i + (NrResultQueues / 2)] = result_queue_q[result_queue_read_pnt_q][i + (NrResultQueues / 2)].wdata;
      bmpu_result_addr_o[i] = result_queue_q[result_queue_read_pnt_q][i].addr;
      bmpu_result_addr_o[i + (NrResultQueues / 2)] = result_queue_q[result_queue_read_pnt_q][i + (NrResultQueues / 2)].addr;
      bmpu_result_id_o[i] = result_queue_q[result_queue_read_pnt_q][i].id;
      bmpu_result_id_o[i + (NrResultQueues / 2)] = result_queue_q[result_queue_read_pnt_q][i + (NrResultQueues / 2)].id;
      bmpu_result_be_o[i] = result_queue_q[result_queue_read_pnt_q][i].be;
      bmpu_result_be_o[i + (NrResultQueues / 2)] = result_queue_q[result_queue_read_pnt_q][i + (NrResultQueues / 2)].be;
    end


    // Received a grant from the VRF.
    // Deactivate the request.
    if (bmpu_result_gnt_i) begin
      automatic logic [8:0] element_cnt = element_cnt_commit;
      result_queue_valid_d[result_queue_read_pnt_q] = 1'b0;
      result_queue_d[result_queue_read_pnt_q]       = '0;

      // Increment the read pointer
      if (result_queue_read_pnt_q == ResultQueueDepth - 1) result_queue_read_pnt_d = 0;
      else result_queue_read_pnt_d = result_queue_read_pnt_q + 1;

      // Decrement the counter of results waiting to be written
      result_queue_cnt_d -= 1;

      // Decrement the counter of remaining vector elements waiting to be written
      if (commit_cnt_q < element_cnt) begin
        commit_cnt_d = '0;
      end else begin
        commit_cnt_d = commit_cnt_q - element_cnt;
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

      bmpu_insn_done_o[vinsn_commit.id] = 1'b1;
    end

    if (commit_cnt_q == '0) begin
      bmpu_output_en_o = 1'b0;
    end

    //////////////////////////////
    //  Accept new instruction  //
    //////////////////////////////

    if (!vinsn_queue_full && vfu_operation_valid_i && (vfu_operation_i.vfu == BMPU || vfu_operation_i.op == BMPSE)) begin
      vinsn_queue_d.vinsn[vinsn_queue_q.accept_pnt] = vfu_operation_i;

      // Initialize counters and alu state if the instruction queue was empty
      if (vinsn_queue_d.issue_cnt == '0) begin
        issue_cnt_d = vfu_operation_i.vl;
      end
      if (vinsn_queue_d.commit_cnt == '0) commit_cnt_d = vfu_operation_i.vl;

      k_dim = vfu_operation_i.k_dim;
      prec = vfu_operation_i.prec;

      // Bump pointers and counters of the vector instruction queue
      vinsn_queue_d.accept_pnt += 1;
      vinsn_queue_d.issue_cnt += 1;
      vinsn_queue_d.commit_cnt += 1;

      if (vfu_operation_i.vfu == BMPU) begin
        bmpu_output_en_o = 1'b0;
        bmpu_valid = 1'b0;
      end
    end

    if (sa_done_q) begin
      // Mark the vector instruction as being done
      bmpu_insn_done_o[vinsn_commit.id] = 1'b1;
      bmpu_valid = 1'b0;
    end


  end : p_bmpu

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
