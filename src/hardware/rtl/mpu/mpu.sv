module mpu import ara_pkg::*; import rvv_pkg::*; import cf_math_pkg::idx_width; #(
    parameter  int    unsigned NrLanes         = 0,
    parameter  int    unsigned VLEN            = 0,
    // Type used to address vector register file elements
    parameter  type            vaddr_t         = logic,
    parameter  type            vfu_operation_t = logic,
    // Dependant parameters. DO NOT CHANGE!
    localparam int    unsigned DataWidth    = $bits(elen_t),
    localparam int    unsigned StrbWidth    = DataWidth/8,
    localparam type            strb_t       = logic [StrbWidth-1:0],
    localparam type            vlen_t       = logic[$clog2(VLEN+1)-1:0]
  ) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic[idx_width(NrLanes)-1:0] lane_id_i,
    // Interface with the lane sequencer
    input  vfu_operation_t               vfu_operation_i,
    input  logic                         vfu_operation_valid_i,
    output logic                         mpu_ready_o,
    output logic           [NrVInsn-1:0] mpu_vinsn_done_o,
    // Interface with the operand queues
    input  elen_t          [3:0]         mpu_act_operand_i,
    input  logic           [3:0]         mpu_act_operand_valid_i,
    output logic           [3:0]         mpu_act_operand_ready_o,
    input  elen_t          [3:0]         mpu_wgt_operand_i,
    input  logic           [3:0]         mpu_wgt_operand_valid_i,
    output logic           [3:0]         mpu_wgt_operand_ready_o,
    // Interface with the vector register file
    output logic                         mpu_result_req_o,
    output vid_t                         mpu_result_id_o,
    output vaddr_t                       mpu_result_addr_o,
    output elen_t                        mpu_result_wdata_o,
    output strb_t                        mpu_result_be_o,
    input  logic                         mpu_result_gnt_i
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
  localparam VInsnQueueDepth = MpuInsnQueueDepth;

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
  } vinsn_queue_d, vinsn_queue_q;

  // Is the vector instruction queue full?
  logic vinsn_queue_full;
  assign vinsn_queue_full = (vinsn_queue_q.commit_cnt == VInsnQueueDepth);

  // Do we have a vector instruction ready to be issued?
  vfu_operation_t vinsn_issue_d, vinsn_issue_q;
  logic           vinsn_issue_valid;
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
    vid_t id;
    vaddr_t addr;
    elen_t wdata;
    strb_t be;
    logic mask;
  } payload_t;

  // Result queue
  payload_t [ResultQueueDepth-1:0]            result_queue_d[3:0], result_queue_q[3:0];
  logic     [ResultQueueDepth-1:0]            result_queue_valid_d[3:0], result_queue_valid_q[3:0];
  // We need two pointers in the result queue. One pointer to indicate with `payload_t` we are
  // currently writing into (write_pnt), and one pointer to indicate which `payload_t` we are
  // currently reading from and writing into the lanes (read_pnt).
  logic     [idx_width(ResultQueueDepth)-1:0] result_queue_write_pnt_d[3:0], result_queue_write_pnt_q[3:0];
  logic     [idx_width(ResultQueueDepth)-1:0] result_queue_read_pnt_d[3:0], result_queue_read_pnt_q[3:0];
  // We need to count how many valid elements are there in this result queue.
  logic     [idx_width(ResultQueueDepth):0]   result_queue_cnt_d[3:0], result_queue_cnt_q[3:0];

  // Is the result queue full?
  logic result_queue_full[3:0];
  assign result_queue_full = (result_queue_cnt_q == ResultQueueDepth);

  always_ff @(posedge clk_i or negedge rst_ni) begin: p_result_queue_ff
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

  // 实例化 SA 模块
  sa #(
    .ROWS      (4),
    .COLS      (4),
    .BIT_ACT   (8),
    .BIT_WEIGHT(1)
  ) u_sa (
    .clk            (clk_i),
    .rst_n          (rst_ni),
    .en             (vfu_operation_i.mpu_en),
    .output_en      (vfu_operation_i.mpu_output_en),
    .act_in         (mpu_act_operand),
    .weight_in      (mpu_wgt_operand),
    .k_dim          (vfu_operation_i.k_dim),
    .output_data    (mpu_output_data),
    .mpu_insn_done_o (mpu_insn_done)
    );

  always_comb begin: p_mpu
    // Maintain state
    vinsn_queue_d = vinsn_queue_q;
    issue_cnt_d   = issue_cnt_q;
    commit_cnt_d  = commit_cnt_q;

    result_queue_d           = result_queue_q;
    result_queue_valid_d     = result_queue_valid_q;
    result_queue_read_pnt_d  = result_queue_read_pnt_q;
    result_queue_write_pnt_d = result_queue_write_pnt_q;
    result_queue_cnt_d       = result_queue_cnt_q;

    first_op_d              = first_op_q;

    // Do not issue any operations
    valu_valid  = 1'b0;
    alu_state_d = alu_state_q;

    // Inform our status to the lane controller
    alu_ready_o      = !vinsn_queue_full;
    alu_vinsn_done_o = '0;

    // Do not acknowledge any operands
    alu_operand_ready_o = '0;
    mask_ready_o        = '0;

    // Don't prevent commit by default
    prevent_commit = 1'b0;



    //////////////////////////////////
    //  Write results into the VRF  //
    //////////////////////////////////

    alu_result_wdata_o = result_queue_q[result_queue_read_pnt_q].wdata;
    if (alu_state_q == NO_REDUCTION || (alu_state_q == SIMD_REDUCTION && simd_red_cnt_q == simd_red_cnt_max_q)) begin
      alu_result_req_o = result_queue_valid_q[result_queue_read_pnt_q] & ((alu_state_q == SIMD_REDUCTION) || !result_queue_q[result_queue_read_pnt_q].mask);
    end else begin
      alu_result_req_o = 1'b0;
    end
    alu_result_addr_o = result_queue_q[result_queue_read_pnt_q].addr;
    alu_result_id_o   = result_queue_q[result_queue_read_pnt_q].id;
    alu_result_be_o   = result_queue_q[result_queue_read_pnt_q].be;


    // Received a grant from the VRF or MASKU.
    // Deactivate the request.
    if (alu_result_gnt_i || mask_operand_gnt) begin
      result_queue_valid_d[result_queue_read_pnt_q] = 1'b0;
      result_queue_d[result_queue_read_pnt_q]       = '0;

      // Increment the read pointer
      if (result_queue_read_pnt_q == ResultQueueDepth-1) result_queue_read_pnt_d = 0;
      else result_queue_read_pnt_d = result_queue_read_pnt_q + 1;

      // Decrement the counter of results waiting to be written
      result_queue_cnt_d -= 1;

      // Decrement the counter of remaining vector elements waiting to be written
      // Don't do it in case of a reduction
      if (!is_reduction(vinsn_commit.op)) begin
        automatic logic [6:0] element_cnt = element_cnt_commit;
          commit_cnt_d = commit_cnt_q - element_cnt;
        if (commit_cnt_q < element_cnt) commit_cnt_d = '0;
      end
    end

    // Finished committing the results of a vector instruction
    if (vinsn_commit_valid && (commit_cnt_d == '0) && !prevent_commit) begin
      // Mark the vector instruction as being done
      mpu_vinsn_done_o[vinsn_commit.id] = 1'b1;

      // Update the commit counters and pointers
      vinsn_queue_d.commit_cnt -= 1;
      if (vinsn_queue_d.commit_pnt == VInsnQueueDepth-1) vinsn_queue_d.commit_pnt = '0;
      else vinsn_queue_d.commit_pnt += 1;

      // Update the commit counter for the next instruction
      if (vinsn_queue_d.commit_cnt != '0)
        commit_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.commit_pnt].vl;

      // If this was a reduction, clean the Lane SLDU/ADDRGEN arbiter
      if (is_reduction(vinsn_commit.op)) alu_red_complete_d = 1'b1;

      // Initialize counters and alu state if needed by the next instruction
      // After a reduction, the next instructions starts after the reduction commits
      if (is_reduction(vinsn_queue_q.vinsn[vinsn_queue_d.issue_pnt].op) && (vinsn_queue_d.issue_cnt != '0)) begin
        // Initialize reduction-related sequential elements
        first_op_d              = 1'b1;
        reduction_rx_cnt_d      = reduction_rx_cnt_init(NrLanes, lane_id_i);
        sldu_transactions_cnt_d = $clog2(NrLanes) + 1;

        alu_state_d = INTRA_LANE_REDUCTION;
      end else begin
        alu_state_d = NO_REDUCTION;
      end
    end

    //////////////////////////////
    //  Accept new instruction  //
    //////////////////////////////

    if (!vinsn_queue_full && vfu_operation_valid_i && vfu_operation_i.vfu == MPU) begin
      vinsn_queue_d.vinsn[vinsn_queue_q.accept_pnt] = vfu_operation_i;

      // Initialize counters and alu state if the instruction queue was empty
      // if ((vinsn_queue_d.issue_cnt == '0) && !prevent_commit) begin
      //   first_op_d              = 1'b1;
      //   issue_cnt_d = vfu_operation_i.vl;
      // end
      // if (vinsn_queue_d.commit_cnt == '0)
      //   commit_cnt_d = vfu_operation_i.vl;

      // Bump pointers and counters of the vector instruction queue
      vinsn_queue_d.accept_pnt += 1;
      vinsn_queue_d.issue_cnt += 1;
      vinsn_queue_d.commit_cnt += 1;
    end
  end: mpu

  endmodule