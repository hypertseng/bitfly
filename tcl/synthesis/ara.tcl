# This script was generated automatically by bender.
set ROOT "/path/to/your/root"
set search_path_initial $search_path

set search_path $search_path_initial

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/tech_cells_generic/src/rtl/tc_sram.sv" \
        "$ROOT/deps/tech_cells_generic/src/rtl/tc_sram_impl.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/tech_cells_generic/src/rtl/tc_clk.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/tech_cells_generic/src/deprecated/pulp_clock_gating_async.sv" \
        "$ROOT/deps/tech_cells_generic/src/deprecated/cluster_clk_cells.sv" \
        "$ROOT/deps/tech_cells_generic/src/deprecated/pulp_clk_cells.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/common_cells/src/binary_to_gray.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/common_cells/src/cb_filter_pkg.sv" \
        "$ROOT/deps/common_cells/src/cc_onehot.sv" \
        "$ROOT/deps/common_cells/src/cdc_reset_ctrlr_pkg.sv" \
        "$ROOT/deps/common_cells/src/cf_math_pkg.sv" \
        "$ROOT/deps/common_cells/src/clk_int_div.sv" \
        "$ROOT/deps/common_cells/src/credit_counter.sv" \
        "$ROOT/deps/common_cells/src/delta_counter.sv" \
        "$ROOT/deps/common_cells/src/ecc_pkg.sv" \
        "$ROOT/deps/common_cells/src/edge_propagator_tx.sv" \
        "$ROOT/deps/common_cells/src/exp_backoff.sv" \
        "$ROOT/deps/common_cells/src/fifo_v3.sv" \
        "$ROOT/deps/common_cells/src/gray_to_binary.sv" \
        "$ROOT/deps/common_cells/src/isochronous_4phase_handshake.sv" \
        "$ROOT/deps/common_cells/src/isochronous_spill_register.sv" \
        "$ROOT/deps/common_cells/src/lfsr.sv" \
        "$ROOT/deps/common_cells/src/lfsr_16bit.sv" \
        "$ROOT/deps/common_cells/src/lfsr_8bit.sv" \
        "$ROOT/deps/common_cells/src/lossy_valid_to_stream.sv" \
        "$ROOT/deps/common_cells/src/mv_filter.sv" \
        "$ROOT/deps/common_cells/src/onehot_to_bin.sv" \
        "$ROOT/deps/common_cells/src/plru_tree.sv" \
        "$ROOT/deps/common_cells/src/passthrough_stream_fifo.sv" \
        "$ROOT/deps/common_cells/src/popcount.sv" \
        "$ROOT/deps/common_cells/src/rr_arb_tree.sv" \
        "$ROOT/deps/common_cells/src/rstgen_bypass.sv" \
        "$ROOT/deps/common_cells/src/serial_deglitch.sv" \
        "$ROOT/deps/common_cells/src/shift_reg.sv" \
        "$ROOT/deps/common_cells/src/shift_reg_gated.sv" \
        "$ROOT/deps/common_cells/src/spill_register_flushable.sv" \
        "$ROOT/deps/common_cells/src/stream_demux.sv" \
        "$ROOT/deps/common_cells/src/stream_filter.sv" \
        "$ROOT/deps/common_cells/src/stream_fork.sv" \
        "$ROOT/deps/common_cells/src/stream_intf.sv" \
        "$ROOT/deps/common_cells/src/stream_join_dynamic.sv" \
        "$ROOT/deps/common_cells/src/stream_mux.sv" \
        "$ROOT/deps/common_cells/src/stream_throttle.sv" \
        "$ROOT/deps/common_cells/src/sub_per_hash.sv" \
        "$ROOT/deps/common_cells/src/sync.sv" \
        "$ROOT/deps/common_cells/src/sync_wedge.sv" \
        "$ROOT/deps/common_cells/src/unread.sv" \
        "$ROOT/deps/common_cells/src/read.sv" \
        "$ROOT/deps/common_cells/src/addr_decode_dync.sv" \
        "$ROOT/deps/common_cells/src/cdc_2phase.sv" \
        "$ROOT/deps/common_cells/src/cdc_4phase.sv" \
        "$ROOT/deps/common_cells/src/clk_int_div_static.sv" \
        "$ROOT/deps/common_cells/src/addr_decode.sv" \
        "$ROOT/deps/common_cells/src/addr_decode_napot.sv" \
        "$ROOT/deps/common_cells/src/multiaddr_decode.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/common_cells/src/cb_filter.sv" \
        "$ROOT/deps/common_cells/src/cdc_fifo_2phase.sv" \
        "$ROOT/deps/common_cells/src/clk_mux_glitch_free.sv" \
        "$ROOT/deps/common_cells/src/counter.sv" \
        "$ROOT/deps/common_cells/src/ecc_decode.sv" \
        "$ROOT/deps/common_cells/src/ecc_encode.sv" \
        "$ROOT/deps/common_cells/src/edge_detect.sv" \
        "$ROOT/deps/common_cells/src/lzc.sv" \
        "$ROOT/deps/common_cells/src/max_counter.sv" \
        "$ROOT/deps/common_cells/src/rstgen.sv" \
        "$ROOT/deps/common_cells/src/spill_register.sv" \
        "$ROOT/deps/common_cells/src/stream_delay.sv" \
        "$ROOT/deps/common_cells/src/stream_fifo.sv" \
        "$ROOT/deps/common_cells/src/stream_fork_dynamic.sv" \
        "$ROOT/deps/common_cells/src/stream_join.sv" \
        "$ROOT/deps/common_cells/src/cdc_reset_ctrlr.sv" \
        "$ROOT/deps/common_cells/src/cdc_fifo_gray.sv" \
        "$ROOT/deps/common_cells/src/fall_through_register.sv" \
        "$ROOT/deps/common_cells/src/id_queue.sv" \
        "$ROOT/deps/common_cells/src/stream_to_mem.sv" \
        "$ROOT/deps/common_cells/src/stream_arbiter_flushable.sv" \
        "$ROOT/deps/common_cells/src/stream_fifo_optimal_wrap.sv" \
        "$ROOT/deps/common_cells/src/stream_register.sv" \
        "$ROOT/deps/common_cells/src/stream_xbar.sv" \
        "$ROOT/deps/common_cells/src/cdc_fifo_gray_clearable.sv" \
        "$ROOT/deps/common_cells/src/cdc_2phase_clearable.sv" \
        "$ROOT/deps/common_cells/src/mem_to_banks_detailed.sv" \
        "$ROOT/deps/common_cells/src/stream_arbiter.sv" \
        "$ROOT/deps/common_cells/src/stream_omega_net.sv" \
        "$ROOT/deps/common_cells/src/mem_to_banks.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/common_cells/src/deprecated/clock_divider_counter.sv" \
        "$ROOT/deps/common_cells/src/deprecated/clk_div.sv" \
        "$ROOT/deps/common_cells/src/deprecated/find_first_one.sv" \
        "$ROOT/deps/common_cells/src/deprecated/generic_LFSR_8bit.sv" \
        "$ROOT/deps/common_cells/src/deprecated/generic_fifo.sv" \
        "$ROOT/deps/common_cells/src/deprecated/prioarbiter.sv" \
        "$ROOT/deps/common_cells/src/deprecated/pulp_sync.sv" \
        "$ROOT/deps/common_cells/src/deprecated/pulp_sync_wedge.sv" \
        "$ROOT/deps/common_cells/src/deprecated/rrarbiter.sv" \
        "$ROOT/deps/common_cells/src/deprecated/clock_divider.sv" \
        "$ROOT/deps/common_cells/src/deprecated/fifo_v2.sv" \
        "$ROOT/deps/common_cells/src/deprecated/fifo_v1.sv" \
        "$ROOT/deps/common_cells/src/edge_propagator_ack.sv" \
        "$ROOT/deps/common_cells/src/edge_propagator.sv" \
        "$ROOT/deps/common_cells/src/edge_propagator_rx.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv" \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/iteration_div_sqrt_mvp.sv" \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/control_mvp.sv" \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/norm_div_sqrt_mvp.sv" \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/preprocess_mvp.sv" \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/nrbd_nrsc_mvp.sv" \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/div_sqrt_top_mvp.sv" \
        "$ROOT/deps/fpu_div_sqrt_mvp/hdl/div_sqrt_mvp_wrapper.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/axi/include"
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/axi/src/axi_pkg.sv" \
        "$ROOT/deps/axi/src/axi_intf.sv" \
        "$ROOT/deps/axi/src/axi_atop_filter.sv" \
        "$ROOT/deps/axi/src/axi_burst_splitter.sv" \
        "$ROOT/deps/axi/src/axi_bus_compare.sv" \
        "$ROOT/deps/axi/src/axi_cdc_dst.sv" \
        "$ROOT/deps/axi/src/axi_cdc_src.sv" \
        "$ROOT/deps/axi/src/axi_cut.sv" \
        "$ROOT/deps/axi/src/axi_delayer.sv" \
        "$ROOT/deps/axi/src/axi_demux_simple.sv" \
        "$ROOT/deps/axi/src/axi_dw_downsizer.sv" \
        "$ROOT/deps/axi/src/axi_dw_upsizer.sv" \
        "$ROOT/deps/axi/src/axi_fifo.sv" \
        "$ROOT/deps/axi/src/axi_id_remap.sv" \
        "$ROOT/deps/axi/src/axi_id_prepend.sv" \
        "$ROOT/deps/axi/src/axi_isolate.sv" \
        "$ROOT/deps/axi/src/axi_join.sv" \
        "$ROOT/deps/axi/src/axi_lite_demux.sv" \
        "$ROOT/deps/axi/src/axi_lite_dw_converter.sv" \
        "$ROOT/deps/axi/src/axi_lite_from_mem.sv" \
        "$ROOT/deps/axi/src/axi_lite_join.sv" \
        "$ROOT/deps/axi/src/axi_lite_lfsr.sv" \
        "$ROOT/deps/axi/src/axi_lite_mailbox.sv" \
        "$ROOT/deps/axi/src/axi_lite_mux.sv" \
        "$ROOT/deps/axi/src/axi_lite_regs.sv" \
        "$ROOT/deps/axi/src/axi_lite_to_apb.sv" \
        "$ROOT/deps/axi/src/axi_lite_to_axi.sv" \
        "$ROOT/deps/axi/src/axi_modify_address.sv" \
        "$ROOT/deps/axi/src/axi_mux.sv" \
        "$ROOT/deps/axi/src/axi_rw_join.sv" \
        "$ROOT/deps/axi/src/axi_rw_split.sv" \
        "$ROOT/deps/axi/src/axi_serializer.sv" \
        "$ROOT/deps/axi/src/axi_slave_compare.sv" \
        "$ROOT/deps/axi/src/axi_throttle.sv" \
        "$ROOT/deps/axi/src/axi_to_detailed_mem.sv" \
        "$ROOT/deps/axi/src/axi_cdc.sv" \
        "$ROOT/deps/axi/src/axi_demux.sv" \
        "$ROOT/deps/axi/src/axi_err_slv.sv" \
        "$ROOT/deps/axi/src/axi_dw_converter.sv" \
        "$ROOT/deps/axi/src/axi_from_mem.sv" \
        "$ROOT/deps/axi/src/axi_id_serialize.sv" \
        "$ROOT/deps/axi/src/axi_lfsr.sv" \
        "$ROOT/deps/axi/src/axi_multicut.sv" \
        "$ROOT/deps/axi/src/axi_to_axi_lite.sv" \
        "$ROOT/deps/axi/src/axi_to_mem.sv" \
        "$ROOT/deps/axi/src/axi_zero_mem.sv" \
        "$ROOT/deps/axi/src/axi_interleaved_xbar.sv" \
        "$ROOT/deps/axi/src/axi_iw_converter.sv" \
        "$ROOT/deps/axi/src/axi_lite_xbar.sv" \
        "$ROOT/deps/axi/src/axi_xbar_unmuxed.sv" \
        "$ROOT/deps/axi/src/axi_to_mem_banked.sv" \
        "$ROOT/deps/axi/src/axi_to_mem_interleaved.sv" \
        "$ROOT/deps/axi/src/axi_to_mem_split.sv" \
        "$ROOT/deps/axi/src/axi_xbar.sv" \
        "$ROOT/deps/axi/src/axi_xp.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/fpnew/src/fpnew_pkg.sv" \
        "$ROOT/deps/fpnew/src/fpnew_cast_multi.sv" \
        "$ROOT/deps/fpnew/src/fpnew_classifier.sv" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v" \
        "$ROOT/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ctrl.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_double.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ff1.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_pack.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_prepare.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_round.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_scalar_dp.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_bound_table.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_with_sqrt.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt.v" \
        "$ROOT/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_top.v" \
        "$ROOT/deps/fpnew/src/fpnew_divsqrt_th_32.sv" \
        "$ROOT/deps/fpnew/src/fpnew_divsqrt_th_64_multi.sv" \
        "$ROOT/deps/fpnew/src/fpnew_divsqrt_multi.sv" \
        "$ROOT/deps/fpnew/src/fpnew_fma.sv" \
        "$ROOT/deps/fpnew/src/fpnew_fma_multi.sv" \
        "$ROOT/deps/fpnew/src/fpnew_sdotp_multi.sv" \
        "$ROOT/deps/fpnew/src/fpnew_sdotp_multi_wrapper.sv" \
        "$ROOT/deps/fpnew/src/fpnew_noncomp.sv" \
        "$ROOT/deps/fpnew/src/fpnew_opgroup_block.sv" \
        "$ROOT/deps/fpnew/src/fpnew_opgroup_fmt_slice.sv" \
        "$ROOT/deps/fpnew/src/fpnew_opgroup_multifmt_slice.sv" \
        "$ROOT/deps/fpnew/src/fpnew_rounding.sv" \
        "$ROOT/deps/fpnew/src/lfsr_sr.sv" \
        "$ROOT/deps/fpnew/src/fpnew_top.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/apb/include"
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/apb/src/apb_pkg.sv" \
        "$ROOT/deps/apb/src/apb_intf.sv" \
        "$ROOT/deps/apb/src/apb_err_slv.sv" \
        "$ROOT/deps/apb/src/apb_regs.sv" \
        "$ROOT/deps/apb/src/apb_cdc.sv" \
        "$ROOT/deps/apb/src/apb_demux.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/cva6/core/include"
lappend search_path "$ROOT/deps/cva6/common/local/util"
lappend search_path "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/include"
lappend search_path "$ROOT/deps/axi/include"
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/cva6/core/include/config_pkg.sv" \
        "$ROOT/deps/cva6/core/include/cv64a6_imafdcv_sv39_config_pkg.sv" \
        "$ROOT/deps/cva6/core/include/riscv_pkg.sv" \
        "$ROOT/deps/cva6/core/include/ariane_pkg.sv" \
        "$ROOT/deps/cva6/core/include/build_config_pkg.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/cva6/core/include"
lappend search_path "$ROOT/deps/cva6/common/local/util"
lappend search_path "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/include"
lappend search_path "$ROOT/deps/axi/include"
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/cva6/core/cva6_accel_first_pass_decoder_stub.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/deps/cva6/core/include"
lappend search_path "$ROOT/deps/cva6/common/local/util"
lappend search_path "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/include"
lappend search_path "$ROOT/deps/axi/include"
lappend search_path "$ROOT/deps/common_cells/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
    } \
    [list \
        "$ROOT/deps/cva6/core/include/wt_cache_pkg.sv" \
        "$ROOT/deps/cva6/core/include/std_cache_pkg.sv" \
        "$ROOT/deps/cva6/core/cvxif_example/include/cvxif_instr_pkg.sv" \
        "$ROOT/deps/cva6/core/cvxif_fu.sv" \
        "$ROOT/deps/cva6/core/cvxif_issue_register_commit_if_driver.sv" \
        "$ROOT/deps/cva6/core/cvxif_compressed_if_driver.sv" \
        "$ROOT/deps/cva6/core/cvxif_example/cvxif_example_coprocessor.sv" \
        "$ROOT/deps/cva6/core/cvxif_example/instr_decoder.sv" \
        "$ROOT/deps/cva6/core/cva6_rvfi_probes.sv" \
        "$ROOT/deps/cva6/core/cva6_fifo_v3.sv" \
        "$ROOT/deps/cva6/core/cva6.sv" \
        "$ROOT/deps/cva6/core/alu.sv" \
        "$ROOT/deps/cva6/core/fpu_wrap.sv" \
        "$ROOT/deps/cva6/core/branch_unit.sv" \
        "$ROOT/deps/cva6/core/compressed_decoder.sv" \
        "$ROOT/deps/cva6/core/controller.sv" \
        "$ROOT/deps/cva6/core/csr_buffer.sv" \
        "$ROOT/deps/cva6/core/csr_regfile.sv" \
        "$ROOT/deps/cva6/core/decoder.sv" \
        "$ROOT/deps/cva6/core/ex_stage.sv" \
        "$ROOT/deps/cva6/core/acc_dispatcher.sv" \
        "$ROOT/deps/cva6/core/instr_realign.sv" \
        "$ROOT/deps/cva6/core/macro_decoder.sv" \
        "$ROOT/deps/cva6/core/id_stage.sv" \
        "$ROOT/deps/cva6/core/issue_read_operands.sv" \
        "$ROOT/deps/cva6/core/issue_stage.sv" \
        "$ROOT/deps/cva6/core/load_unit.sv" \
        "$ROOT/deps/cva6/core/load_store_unit.sv" \
        "$ROOT/deps/cva6/core/lsu_bypass.sv" \
        "$ROOT/deps/cva6/core/mult.sv" \
        "$ROOT/deps/cva6/core/multiplier.sv" \
        "$ROOT/deps/cva6/core/serdiv.sv" \
        "$ROOT/deps/cva6/core/perf_counters.sv" \
        "$ROOT/deps/cva6/core/ariane_regfile_ff.sv" \
        "$ROOT/deps/cva6/core/ariane_regfile_fpga.sv" \
        "$ROOT/deps/cva6/core/scoreboard.sv" \
        "$ROOT/deps/cva6/core/store_buffer.sv" \
        "$ROOT/deps/cva6/core/amo_buffer.sv" \
        "$ROOT/deps/cva6/core/store_unit.sv" \
        "$ROOT/deps/cva6/core/commit_stage.sv" \
        "$ROOT/deps/cva6/core/axi_shim.sv" \
        "$ROOT/deps/cva6/core/frontend/btb.sv" \
        "$ROOT/deps/cva6/core/frontend/bht.sv" \
        "$ROOT/deps/cva6/core/frontend/ras.sv" \
        "$ROOT/deps/cva6/core/frontend/instr_scan.sv" \
        "$ROOT/deps/cva6/core/frontend/instr_queue.sv" \
        "$ROOT/deps/cva6/core/frontend/frontend.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/wt_dcache_ctrl.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/wt_dcache_mem.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/wt_dcache_missunit.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/wt_dcache_wbuffer.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/wt_dcache.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/wt_cache_subsystem.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/wt_axi_adapter.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/cva6_icache.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/tag_cmp.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/cache_ctrl.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/amo_alu.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/axi_adapter.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/miss_handler.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/std_nbdcache.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/cva6_icache_axi_wrapper.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/std_cache_subsystem.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_pkg.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_resp_demux.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_to_axi_read.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_to_axi_write.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_req_read_arbiter.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_req_write_arbiter.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_demux.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_lfsr.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sync_buffer.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_fifo_reg.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_fifo_reg_initialized.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_fxarb.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_rrarb.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_mux.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_decoder.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_1hot_to_binary.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_prio_1hot_encoder.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sram.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sram_wbyteenable.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sram_wmask.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_regbank_wbyteenable_1rw.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_regbank_wmask_1rw.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_data_downsize.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_data_upsize.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_data_resize.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride_pkg.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride_arb.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride_wrapper.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_amo.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_cmo.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_core_arbiter.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_ctrl.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_ctrl_pe.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_memctrl.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_miss_handler.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_mshr.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_rtab.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_uncached.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_victim_plru.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_victim_random.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_victim_sel.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_wbuf.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_flush.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/cva6_hpdcache_if_adapter.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/cva6_hpdcache_subsystem_axi_arbiter.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/cva6_hpdcache_subsystem.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/cva6_hpdcache_wrapper.sv" \
        "$ROOT/deps/cva6/core/cache_subsystem/hpdcache_tc_sram.sv" \
        "$ROOT/deps/cva6/core/pmp/src/pmp.sv" \
        "$ROOT/deps/cva6/core/pmp/src/pmp_entry.sv" \
        "$ROOT/deps/cva6/core/pmp/src/pmp_data_if.sv" \
        "$ROOT/deps/cva6/vendor/pulp-platform/fpga-support/fpga-support-stubs.sv" \
        "$ROOT/deps/cva6/common/local/util/tc_sram_wrapper.sv" \
        "$ROOT/deps/cva6/common/local/util/tc_sram_wrapper_cache_techno.sv" \
        "$ROOT/deps/cva6/common/local/util/sram_pulp.sv" \
        "$ROOT/deps/cva6/common/local/util/sram_cache.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/include"
lappend search_path "$ROOT/deps/axi/include"
lappend search_path "$ROOT/deps/common_cells/include"
lappend search_path "$ROOT/deps/cva6/core/include"
lappend search_path "$ROOT/deps/cva6/core/cache_subsystem/hpdcache/rtl/include"
lappend search_path "$ROOT/deps/apb/include"

if {[catch {analyze -format sv \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        NR_LANES=4 \
        VLEN=4096 \
        SPYGLASS \
    } \
    [list \
        "$ROOT/include/rvv_pkg.sv" \
        "$ROOT/include/ara_pkg.sv" \
        "$ROOT/src/segment_sequencer.sv" \
        "$ROOT/src/mpu/pe.sv" \
        "$ROOT/src/ctrl_registers.sv" \
        "$ROOT/src/cva6_accel_first_pass_decoder.sv" \
        "$ROOT/src/ara_dispatcher.sv" \
        "$ROOT/src/ara_sequencer.sv" \
        "$ROOT/src/axi_inval_filter.sv" \
        "$ROOT/src/lane/lane_sequencer.sv" \
        "$ROOT/src/lane/operand_queue.sv" \
        "$ROOT/src/lane/operand_requester.sv" \
        "$ROOT/src/lane/simd_alu.sv" \
        "$ROOT/src/lane/simd_div.sv" \
        "$ROOT/src/lane/simd_mul.sv" \
        "$ROOT/src/lane/vector_regfile.sv" \
        "$ROOT/src/lane/power_gating_generic.sv" \
        "$ROOT/src/masku/masku_operands.sv" \
        "$ROOT/src/sldu/p2_stride_gen.sv" \
        "$ROOT/src/sldu/sldu_op_dp.sv" \
        "$ROOT/src/sldu/sldu.sv" \
        "$ROOT/src/vlsu/addrgen.sv" \
        "$ROOT/src/vlsu/vldu.sv" \
        "$ROOT/src/vlsu/vstu.sv" \
        "$ROOT/src/mpu/tile.sv" \
        "$ROOT/src/lane/operand_queues_stage.sv" \
        "$ROOT/src/lane/valu.sv" \
        "$ROOT/src/lane/vmfpu.sv" \
        "$ROOT/src/lane/fixed_p_rounding.sv" \
        "$ROOT/src/vlsu/vlsu.sv" \
        "$ROOT/src/masku/masku.sv" \
        "$ROOT/src/mpu/sa.sv" \
        "$ROOT/src/mpu/mpu.sv" \
        "$ROOT/src/lane/vector_fus_stage.sv" \
        "$ROOT/src/lane/lane.sv" \
        "$ROOT/src/ara.sv" \
        "$ROOT/src/ara_system.sv" \
        "$ROOT/src/ara_soc.sv" \
    ]
}]} {return 1}

set search_path $search_path_initial
