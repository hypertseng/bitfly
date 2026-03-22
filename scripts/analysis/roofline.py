import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.edgecolor": "#666666",
    "axes.linewidth": 1.4,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": ":",
    "grid.color": "#C7CCD3",
    "grid.alpha": 0.34,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "savefig.dpi": 320,
})


# -----------------------------------------------------------------------------
# Architecture constants
# -----------------------------------------------------------------------------
AXI_BW_BYTES_PER_CYCLE = 16.0
RVV_INT8_PEAK_MAC = 32.0

SA_ROWS = 2
SA_COLS = 2
LBMACS_PER_PE = 8
DOT_WIDTH_PER_LBMAC = 8
PES_TOTAL = SA_ROWS * SA_COLS
NR_RESULT_QUEUES = 4
BMPU_INPUT_QUEUES = 2

SHAPE = (128, 8192, 2048)
KTILE = 128
OUT_BITS = 16

PRECISIONS = [
    {"name": "INT4", "weight_bits": 4, "planes": 4, "color": "#31688E", "marker": "D", "peak": 64.0},
    {"name": "INT2", "weight_bits": 2, "planes": 2, "color": "#35B779", "marker": "^", "peak": 128.0},
    {"name": "Binary", "weight_bits": 1, "planes": 1, "color": "#E54B4B", "marker": "s", "peak": 256.0},
]

REUSE_COLORS = {
    1: "#7C7C7C",
    2: "#4E79A7",
    4: "#59A14F",
    8: "#E15759",
}

PRECISION_LABELS = {
    "INT4": "W4A8",
    "INT2": "W2A8",
    "Binary": "W1A8",
}


def sa_cycles(k_dim, planes):
    # src/hardware/rtl/bmpu/sa.sv
    return (SA_ROWS - 1) + (k_dim // 8) * planes + SA_COLS


def rtl_compute_peak(prec):
    ops_per_bmpmm = PES_TOTAL * LBMACS_PER_PE * DOT_WIDTH_PER_LBMAC * (KTILE // 8)
    return ops_per_bmpmm / sa_cycles(KTILE, prec["planes"])


def enumerate_configs():
    vals_mt = [8, 16, 32, 64]
    vals_nt = [16, 32, 64, 128, 256]
    vals_g = [1, 2, 4, 8]
    out = []
    for mt in vals_mt:
        for nt in vals_nt:
            for gm in vals_g:
                for gn in vals_g:
                    if mt * nt * gm * gn * 16 == 16384:
                        out.append((mt, nt, gm, gn))
    return out


def ceil_div(x, y):
    return (x + y - 1) // y


def design_projection_point(cfg, prec):
    # Architecture-aware design-space projection used in the motivation section.
    # The purpose is not to claim post-implementation performance, but to show
    # how candidate tile-group configurations would move the operating point
    # under the target BitFly execution principles.
    #
    # We only keep deterministic constraints that are explicit in the current
    # software template and RTL:
    # - bmpmm_template.h executes mg -> ng -> kt -> wi -> ai in program order.
    # - bmpu.sv advances compute_ai / compute_wi sequentially, so gm*gn contexts
    #   are not executed in parallel.
    # - A is loaded inside the wi/ai loop, so it is reissued for every wi.
    # - BMPSE streams one result word per cycle over NrResultQueues words.
    # - No speculative overlap is assumed.
    mt, nt, gm, gn = cfg
    m_dim, k_dim, n_dim = SHAPE

    m_tiles = ceil_div(m_dim, mt)
    n_tiles = ceil_div(n_dim, nt)
    k_tiles = ceil_div(k_dim, KTILE)

    compute_cycles_tile = sa_cycles(KTILE, prec["planes"])
    compute_bound = rtl_compute_peak(prec)

    total_macs = 0.0
    total_bytes = 0.0
    total_cycles = 0.0

    for mg in range(0, m_tiles, gm):
        mg_len = min(gm, m_tiles - mg)
        for ng0 in range(0, n_tiles, gn):
            ng_len = min(gn, n_tiles - ng0)

            group_macs = 0.0
            group_bytes = 0.0
            group_cycles = 0.0

            bytes_w_one = nt * KTILE * (prec["weight_bits"] / 8.0)
            bytes_a_one = mt * KTILE
            bytes_c_one = mt * nt * (OUT_BITS / 8.0)
            load_w_cycles_one = bytes_w_one / AXI_BW_BYTES_PER_CYCLE
            load_a_cycles_one = bytes_a_one / AXI_BW_BYTES_PER_CYCLE
            issue_cycles_one = 1.0

            for kt in range(k_tiles):
                # One weight tile is loaded for each wi and then reused across ai.
                weight_load_cycles = ng_len * load_w_cycles_one

                # For every (wi, ai), software explicitly issues load_a() and bmpmm()
                # in sequence; bmpu.sv also advances contexts sequentially.
                body_cycles = ng_len * mg_len * (issue_cycles_one + load_a_cycles_one + compute_cycles_tile)
                k_stage_cycles = weight_load_cycles + body_cycles

                group_cycles += k_stage_cycles
                group_macs += mg_len * ng_len * mt * nt * KTILE
                group_bytes += (
                    ng_len * bytes_w_one
                    + ng_len * mg_len * bytes_a_one
                )

            # lane.sv ties bmpu_store_ready to the store-unit ready signal, and
            # vstu.sv only releases readiness after consuming a full lane payload.
            # We therefore keep both the bandwidth cost and the fixed 4-word BMPSE
            # commit granularity, plus one explicit handshake cycle per context.
            store_cycles_per_ctx = max(
                bytes_c_one / AXI_BW_BYTES_PER_CYCLE,
                float(NR_RESULT_QUEUES) + 1.0,
            )
            group_cycles += mg_len * ng_len * store_cycles_per_ctx
            group_bytes += mg_len * ng_len * bytes_c_one

            total_macs += group_macs
            total_bytes += group_bytes
            total_cycles += group_cycles

    oi = total_macs / total_bytes
    perf = min(total_macs / total_cycles, compute_bound)
    return oi, perf, compute_bound


def pareto_frontier(points):
    out = []
    for p in sorted(points, key=lambda item: (item["oi"], item["perf"])):
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["oi"] >= p["oi"] and q["perf"] >= p["perf"] and (q["oi"] > p["oi"] or q["perf"] > p["perf"]):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda item: item["oi"])


def representative_frontier(points):
    frontier = pareto_frontier(points)
    groups = {}
    for p in frontier:
        key = (round(p["oi"], 3), round(p["perf"], 3))
        groups.setdefault(key, []).append(p)

    reps = []
    for key in sorted(groups.keys()):
        members = groups[key]
        members = sorted(members, key=lambda p: (p["cfg"][2] * p["cfg"][3], p["cfg"][0] * p["cfg"][1], p["cfg"]))
        reps.append(members[0])
    return frontier, reps


def style_axis(ax):
    ax.set_facecolor("#FCFCFB")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)
    ax.tick_params(axis="x", pad=3)
    ax.tick_params(axis="y", pad=3)


def line_label(ax, x, y, text, color, dx=6, dy=0, fs=7.9):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fs,
        color=color,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
    )


def pick_representative_labels(frontier_reps):
    if len(frontier_reps) <= 3:
        return frontier_reps
    idxs = [0, len(frontier_reps) // 2, len(frontier_reps) - 1]
    return [frontier_reps[i] for i in idxs]


def cfg_label(cfg):
    mt, nt, gm, gn = cfg
    return rf"$m_t$={mt}, $n_t$={nt}, $g_m \times g_n$={gm}x{gn}"


def annotate_cfg(ax, point, text, rank):
    x = point["oi"]
    y = point["perf"]
    if x > 0.8 * ax.get_xlim()[1]:
        dx = -74
        ha = "right"
    else:
        dx = 8
        ha = "left"
    dy = [11, -13, 12, -12][rank % 4]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=11.0,
        color="#303030",
        ha=ha,
        arrowprops=dict(arrowstyle="-", lw=1.0, color="#8A8A8A", shrinkA=2, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D8D8D8", alpha=0.95),
    )


def nice_bounds(values, lower_pad=0.9, upper_pad=1.12):
    vmin = min(values)
    vmax = max(values)
    return vmin * lower_pad, vmax * upper_pad


def bounded_ylim(pts, x_min, x_max, peak):
    perf_vals = [p["perf"] for p in pts]
    mem_y0 = AXI_BW_BYTES_PER_CYCLE * x_min
    mem_y1 = AXI_BW_BYTES_PER_CYCLE * x_max
    compute_vals = [p["compute_bound"] for p in pts]
    candidates = perf_vals + compute_vals + [mem_y0, mem_y1, peak]
    y_low = min(candidates)
    y_high = max(candidates)

    # Force the compute roof to stay visibly inside the panel rather than
    # sitting on the lower edge for W4/W2.
    y_min = min(y_low * 0.78, peak * 0.72)
    y_max = max(y_high * 1.12, peak * 1.22)
    return y_min, y_max


def gather_projection_points():
    cfgs = enumerate_configs()
    out = []
    for prec in PRECISIONS:
        pts = []
        for cfg in cfgs:
            oi, perf, compute_bound = design_projection_point(cfg, prec)
            pts.append({"cfg": cfg, "oi": oi, "perf": perf, "compute_bound": compute_bound, "reuse": cfg[2] * cfg[3]})
        out.append((prec, pts))
    return out


def draw_projection_subplot(ax, prec, pts, shared_ylabel=False):
    style_axis(ax)

    frontier, frontier_reps = representative_frontier(pts)
    frontier_cfgs = {p["cfg"] for p in frontier}
    frontier_x = [p["oi"] for p in frontier]
    frontier_y = [p["perf"] for p in frontier]
    x_min, x_max = nice_bounds([p["oi"] for p in pts], lower_pad=0.92, upper_pad=1.12)
    y_min, y_max = bounded_ylim(pts, x_min, x_max, rtl_compute_peak(prec))
    oi_x_impl = np.logspace(np.log10(x_min), np.log10(x_max), 800)

    ax.plot(oi_x_impl, oi_x_impl * AXI_BW_BYTES_PER_CYCLE, color="#111111", linestyle="-.", linewidth=2.0, zorder=1)
    ax.hlines(rtl_compute_peak(prec), oi_x_impl.min(), oi_x_impl.max(), colors=prec["color"], linewidth=2.0, linestyle=":", zorder=1)
    ax.plot(frontier_x, frontier_y, color=prec["color"], linewidth=2.4, alpha=0.95, zorder=4)

    for p in pts:
        is_frontier = p["cfg"] in frontier_cfgs
        size = 84 + 24 * np.log2(p["reuse"])
        ax.scatter(
            [p["oi"]],
            [p["perf"]],
            s=size,
            marker=prec["marker"],
            facecolor=REUSE_COLORS[p["reuse"]] if is_frontier else "#F5F5F5",
            edgecolor=REUSE_COLORS[p["reuse"]],
            linewidth=1.45 if is_frontier else 0.75,
            alpha=0.97 if is_frontier else 0.26,
            zorder=5 if is_frontier else 2,
        )

    for rank, p in enumerate(pick_representative_labels(frontier_reps)):
        annotate_cfg(ax, p, cfg_label(p["cfg"]), rank)

    ax.text(
        0.02,
        0.93,
        f"{PRECISION_LABELS[prec['name']]}",
        transform=ax.transAxes,
        fontsize=14.6,
        va="top",
        ha="left",
        color=prec["color"],
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#D0D0D0", alpha=0.92),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    if shared_ylabel:
        ax.set_ylabel(r"Attained Performance (MACs / cycle)", weight="bold")
    ax.set_xlabel(r"Operational Intensity (MACs / Byte)", weight="bold")
    line_label(ax, x_min * 1.04, AXI_BW_BYTES_PER_CYCLE * x_min * 1.04, "memory roof", "#111111", dy=8, fs=12.3)
    line_label(ax, x_min * 1.04, rtl_compute_peak(prec), "compute roof", prec["color"], dy=8, fs=12.3)


def save_design_projection_figure():
    data = gather_projection_points()
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 6.6), gridspec_kw={"wspace": 0.15})
    for idx, ((prec, pts), ax) in enumerate(zip(data, axes)):
        draw_projection_subplot(ax, prec, pts, shared_ylabel=(idx == 0))
    reuse_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[1], markeredgecolor=REUSE_COLORS[1], markersize=10, label=r"$g_m g_n = 1$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[2], markeredgecolor=REUSE_COLORS[2], markersize=10, label=r"$g_m g_n = 2$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[4], markeredgecolor=REUSE_COLORS[4], markersize=10, label=r"$g_m g_n = 4$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[8], markeredgecolor=REUSE_COLORS[8], markersize=10, label=r"$g_m g_n = 8$"),
        plt.Line2D([0], [0], marker="o", color="#7A7A7A", markerfacecolor="white", alpha=0.3, markersize=10, linewidth=0, label="dominated"),
        plt.Line2D([0], [0], color="#7A7A7A", linewidth=2.2, marker="o", markerfacecolor="#7A7A7A", markersize=8, label="frontier"),
    ]
    fig.legend(handles=reuse_handles, loc="lower center", ncol=6, framealpha=0.95, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(top=0.96, bottom=0.15, left=0.065, right=0.99)
    fig.savefig("roofline_design_projection.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig("roofline_design_projection.pdf", bbox_inches="tight", pad_inches=0.03)


def main():
    save_design_projection_figure()
    print(f"Enumerated configs: {len(enumerate_configs())}")
    print("Saved figure: roofline_design_projection.[png|pdf]")


if __name__ == "__main__":
    main()
