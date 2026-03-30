`ifndef BITFLY_DEBUG_SVH_
`define BITFLY_DEBUG_SVH_

// Global hardware debug switch. Keep disabled by default.
// Enable all debug prints with:
//   --define BITFLY_HW_DEBUG=1
//
// Or enable a narrower domain with one of the scoped defines below.
`ifndef BITFLY_HW_DEBUG
  `define BITFLY_HW_DEBUG 1'b0
`endif

`ifndef BITFLY_DISPATCH_DEBUG
  `define BITFLY_DISPATCH_DEBUG `BITFLY_HW_DEBUG
`endif

`ifndef BITFLY_VSTU_DEBUG
  `define BITFLY_VSTU_DEBUG `BITFLY_HW_DEBUG
`endif

`ifndef BITFLY_OPREQ_DEBUG
  `define BITFLY_OPREQ_DEBUG `BITFLY_HW_DEBUG
`endif

// Extra-opreq detail (weight block traces). Off by default.
`ifndef BITFLY_OPREQ_WGT_DEBUG
  `define BITFLY_OPREQ_WGT_DEBUG 1'b0
`endif

`ifndef BITFLY_OPQ_DEBUG
  `define BITFLY_OPQ_DEBUG `BITFLY_HW_DEBUG
`endif

`ifndef BITFLY_LANE_DEBUG
  `define BITFLY_LANE_DEBUG `BITFLY_HW_DEBUG
`endif

`ifndef BITFLY_LSEQ_DEBUG
  `define BITFLY_LSEQ_DEBUG `BITFLY_HW_DEBUG
`endif

`ifndef BITFLY_BMPU_DEBUG
  `define BITFLY_BMPU_DEBUG `BITFLY_HW_DEBUG
`endif

`endif
