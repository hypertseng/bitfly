import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.0,
    "legend.frameon": True,
    "legend.edgecolor": "#666666",
    "axes.linewidth": 1.1,
    "axes.grid": True,
    "grid.linewidth": 0.38,
    "grid.linestyle": ":",
    "grid.color": "#C7CCD3",
    "grid.alpha": 0.58,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "savefig.dpi": 300,
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


def roof(oi, peak):
    return np.minimum(oi * AXI_BW_BYTES_PER_CYCLE, peak)


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


def actual_upgraded_point(cfg, prec):
    # Implementation-aware model following bmpmm_operator_template.h:
    # for mg -> ng -> kt -> wi -> ai:
    #   load_w()
    #   load_a()
    #   bmpmm()
    # and stores happen after the full K sweep of one (mg, ng) group.
    #
    # This model only keeps constraints that are explicit in the software
    # template and RTL:
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
        bbox=dict(boxstyle="round,pad=0.11", fc="white", ec="none", alpha=0.8),
    )


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
    dy = [10, -11, 10, -11][rank % 4]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6.5,
        color="#303030",
        ha=ha,
        arrowprops=dict(arrowstyle="-", lw=0.55, color="#8A8A8A", shrinkA=2, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#D8D8D8", alpha=0.92),
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


def main():
    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.94, 1.08], wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(3, 1, hspace=0.15)
    ax2_list = [fig.add_subplot(right_gs[i, 0]) for i in range(3)]
    oi_x = np.logspace(np.log10(0.25), np.log10(128), 1200)
    style_axis(ax1)
    for ax in ax2_list:
        style_axis(ax)

    ax1.plot(oi_x, oi_x * 8.0, color="#C3C7CE", linestyle="-.", linewidth=1.5)
    ax1.plot(oi_x, oi_x * 12.0, color="#8B9199", linestyle="-.", linewidth=1.6)
    ax1.plot(oi_x, oi_x * AXI_BW_BYTES_PER_CYCLE, color="#111111", linestyle="-.", linewidth=1.9)
    ax1.plot(oi_x, roof(oi_x, RVV_INT8_PEAK_MAC), color="#6F6F6F", linestyle="--", linewidth=1.8)
    for prec in PRECISIONS:
        ax1.hlines(rtl_compute_peak(prec), oi_x.min(), oi_x.max(), colors=prec["color"], linewidth=1.55)

    concept_pts = {
        "rvv": (4.0, 32.0),
        "base": (7.5, 60.0),
        "reuse": (13.5, 130.0),
        "final": (24.0, 250.0),
    }
    ax1.scatter([concept_pts["rvv"][0]], [concept_pts["rvv"][1]], s=82, marker="o", facecolor="white", edgecolor="#5A5A5A", linewidth=1.3, zorder=5)
    ax1.scatter([concept_pts["base"][0]], [concept_pts["base"][1]], s=84, marker="o", facecolor="#A5A5A5", edgecolor="#A5A5A5", linewidth=1.1, zorder=5)
    ax1.scatter([concept_pts["reuse"][0]], [concept_pts["reuse"][1]], s=88, marker="D", facecolor="#8A7394", edgecolor="#8A7394", linewidth=1.1, zorder=5)
    ax1.scatter([concept_pts["final"][0]], [concept_pts["final"][1]], s=92, marker="s", facecolor="#D95B59", edgecolor="#D95B59", linewidth=1.1, zorder=5)
    ax1.annotate("", xy=concept_pts["reuse"], xytext=concept_pts["base"], arrowprops=dict(arrowstyle="->", lw=1.2, color="#8A7394"))
    ax1.annotate("", xy=concept_pts["final"], xytext=concept_pts["reuse"], arrowprops=dict(arrowstyle="->", lw=1.2, color="#D95B59"))

    for text, xy, offset, color in [
        ("RVV low-bit\nceiling", concept_pts["rvv"], (-12, -24), "#5A5A5A"),
        ("Baseline accelerator", concept_pts["base"], (10, -2), "#666666"),
        ("+ grouped reuse", concept_pts["reuse"], (12, -2), "#8A7394"),
        ("+ overlap optimizations", concept_pts["final"], (14, -2), "#D95B59"),
    ]:
        ax1.annotate(text, xy=xy, xytext=offset, textcoords="offset points", fontsize=7.9, color=color)

    ax1.text(
        0.05,
        0.95,
        "Conceptual roofline used before the implementation section",
        transform=ax1.transAxes,
        fontsize=7.7,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#D0D0D0", alpha=0.92),
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.25, 128)
    ax1.set_ylim(16, 300)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax1.set_xlabel(r"Operational Intensity (MACs / Byte)", weight="bold")
    ax1.set_ylabel(r"Attained Performance (MACs / cycle)", weight="bold")
    ax1.set_title("(a) Conceptual Roofline", weight="bold", pad=9)
    line_label(ax1, 1.0, 8.0, "memory roof (baseline)", "#9CA2AB", dy=10, fs=7.2)
    line_label(ax1, 1.55, 12.0 * 1.55, "memory roof (+ reuse)", "#6E7680", dy=-2, fs=7.2)
    line_label(ax1, 2.2, AXI_BW_BYTES_PER_CYCLE * 2.2, "memory roof (+ overlap)", "#111111", dy=8, fs=7.2)
    line_label(ax1, 4.35, RVV_INT8_PEAK_MAC, "RVV ceiling", "#666666", dy=-18, fs=7.2)
    line_label(ax1, 14.2, rtl_compute_peak(PRECISIONS[0]), "W4A8 roof", PRECISIONS[0]["color"], dy=-1, fs=7.2)
    line_label(ax1, 14.2, rtl_compute_peak(PRECISIONS[1]), "W2A8 roof", PRECISIONS[1]["color"], dy=-1, fs=7.2)
    line_label(ax1, 19.2, rtl_compute_peak(PRECISIONS[2]), "W1A8 roof", PRECISIONS[2]["color"], dy=0, fs=7.2)

    cfgs = enumerate_configs()
    all_points = []
    for prec in PRECISIONS:
        for cfg in cfgs:
            oi, perf, compute_bound = actual_upgraded_point(cfg, prec)
            all_points.append({"cfg": cfg, "oi": oi, "perf": perf, "compute_bound": compute_bound, "reuse": cfg[2] * cfg[3], "prec": prec})

    for idx, (ax, prec) in enumerate(zip(ax2_list, PRECISIONS)):
        pts = []
        for cfg in cfgs:
            oi, perf, compute_bound = actual_upgraded_point(cfg, prec)
            pts.append({"cfg": cfg, "oi": oi, "perf": perf, "compute_bound": compute_bound, "reuse": cfg[2] * cfg[3]})

        frontier, frontier_reps = representative_frontier(pts)
        frontier_cfgs = {p["cfg"] for p in frontier}
        frontier_x = [p["oi"] for p in frontier]
        frontier_y = [p["perf"] for p in frontier]
        x_min, x_max = nice_bounds([p["oi"] for p in pts], lower_pad=0.9, upper_pad=1.18)
        y_min, y_max = bounded_ylim(pts, x_min, x_max, rtl_compute_peak(prec))
        oi_x_impl = np.logspace(np.log10(x_min), np.log10(x_max), 800)

        ax.plot(oi_x_impl, oi_x_impl * AXI_BW_BYTES_PER_CYCLE, color="#111111", linestyle="-.", linewidth=1.8)
        ax.hlines(rtl_compute_peak(prec), oi_x_impl.min(), oi_x_impl.max(), colors=prec["color"], linewidth=1.55, linestyle=":")
        ax.plot(frontier_x, frontier_y, color=prec["color"], linewidth=1.3, alpha=0.9, zorder=3)

        for p in pts:
            is_frontier = p["cfg"] in frontier_cfgs
            size = 33 + 8 * np.log2(p["reuse"])
            ax.scatter(
                [p["oi"]],
                [p["perf"]],
                s=size,
                marker=prec["marker"],
                facecolor=REUSE_COLORS[p["reuse"]] if is_frontier else "white",
                edgecolor=REUSE_COLORS[p["reuse"]],
                linewidth=1.4 if is_frontier else 0.9,
                alpha=0.96 if is_frontier else 0.58,
                zorder=5 if is_frontier else 2,
            )

        for rank, p in enumerate(frontier_reps):
            annotate_cfg(ax, p, cfg_label(p["cfg"]), rank)

        ax.text(
            0.02,
            0.93,
            f"{PRECISION_LABELS[prec['name']]} implemented roofline",
            transform=ax.transAxes,
            fontsize=8.0,
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
        if idx < 2:
            ax.tick_params(labelbottom=False)
        if idx == 1:
            ax.set_ylabel(r"Attained Performance (MACs / cycle)", weight="bold")
        line_label(ax, x_min * 1.03, AXI_BW_BYTES_PER_CYCLE * x_min * 1.03, "memory roof", "#111111", dy=5, fs=7.0)
        line_label(ax, x_min * 1.03, rtl_compute_peak(prec), "compute roof", prec["color"], dy=5, fs=7.0)

    ax2_list[0].set_title("(b) Implemented Roofline Under Actual Architectural Optimizations", weight="bold", pad=8)
    ax2_list[-1].set_xlabel(r"Operational Intensity (MACs / Byte)", weight="bold")

    ax2_list[0].text(
        0.98,
        0.93,
        "\n".join([
            r"Shape: $(128, 8192, 2048)$",
            r"$m_t n_t g_m g_n \times 16 = 16384$",
            "Filled: Pareto frontier",
            "Hollow: dominated design",
        ]),
        transform=ax2_list[0].transAxes,
        fontsize=7.1,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#D0D0D0", alpha=0.92),
    )

    reuse_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[1], markeredgecolor=REUSE_COLORS[1], markersize=7, label="gm*gn = 1"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[2], markeredgecolor=REUSE_COLORS[2], markersize=7, label="gm*gn = 2"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[4], markeredgecolor=REUSE_COLORS[4], markersize=7, label="gm*gn = 4"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REUSE_COLORS[8], markeredgecolor=REUSE_COLORS[8], markersize=7, label="gm*gn = 8"),
    ]
    ax2_list[-1].legend(
        handles=reuse_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.52),
        ncol=1,
        facecolor="#F8F8F8",
        edgecolor="#DFDFDF",
        framealpha=0.94,
        title=r"Reuse Level ($g_m g_n$)",
        borderaxespad=0.0,
    )

    fig.suptitle("Roofline Analysis of Low-bit Mixed-precision Execution", y=0.982, fontsize=12.4, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.07, right=0.92)
    fig.savefig("roofline_bitfly_arch_model.png")
    fig.savefig("roofline_bitfly_arch_model.pdf")

    print(f"Enumerated configs: {len(cfgs)}")
    print("Saved architecture-aware roofline figures: roofline_bitfly_arch_model.[png|pdf]")


if __name__ == "__main__":
    main()
