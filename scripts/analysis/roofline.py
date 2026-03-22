import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.frameon": False,
    "axes.linewidth": 0.95,
    "axes.grid": True,
    "grid.linewidth": 0.7,
    "grid.linestyle": (0, (2, 4)),
    "grid.color": "#D8E1EB",
    "grid.alpha": 0.85,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.color": "#5B6878",
    "ytick.color": "#5B6878",
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

REUSE_LEVELS = [1, 2, 4, 8]
REUSE_COLORS = {
    1: "#6FA8DC",
    2: "#5CB85C",
    4: "#F0AD4E",
    8: "#E76F51",
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
    return oi, perf, compute_bound, total_cycles


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
        members = sorted(
            members,
            key=lambda p: (
                p["cycles"],
                -p["reuse"],
                p["cfg"][0] * p["cfg"][1],
                p["cfg"],
            ),
        )
        reps.append(members[0])
    return frontier, reps


def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8D2DD")
    ax.spines["bottom"].set_color("#C8D2DD")
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.tick_params(axis="x", pad=3, length=0)
    ax.tick_params(axis="y", pad=3, length=0)
    ax.grid(which="major", alpha=0.8)
    ax.grid(which="minor", alpha=0.18, linewidth=0.45)


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
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9),
    )


def cfg_label(cfg):
    mt, nt, gm, gn = cfg
    return rf"$m_t$={mt}, $n_t$={nt}, $g_m \times g_n$={gm}x{gn}"


def reuse_marker_size(reuse):
    return 42 + 24 * np.log2(reuse)


def scatter_legend_markersize(reuse):
    return np.sqrt(reuse_marker_size(reuse))


MEMORY_ROOF_LABEL_X_FACTOR = 0.58
MEMORY_ROOF_LABEL_DY = 10
COMPUTE_ROOF_LABEL_X_FACTOR = 0.72
COMPUTE_ROOF_LABEL_DY = 8


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
        fontsize=9.8,
        color="#344054",
        ha=ha,
        arrowprops=dict(arrowstyle="-", lw=0.9, color="#98A2B3", shrinkA=2, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D9E1EA", alpha=0.96),
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


def draw_conceptual_panel(ax1):
    oi_x = np.logspace(np.log10(0.25), np.log10(128), 1200)
    style_axis(ax1)
    ax1.plot(oi_x, oi_x * 6.0, color="#C9D1DB", linestyle=(0, (5, 3)), linewidth=1.25)
    ax1.plot(oi_x, oi_x * 10.0, color="#93A1B2", linestyle=(0, (5, 3)), linewidth=1.25)
    ax1.plot(oi_x, oi_x * AXI_BW_BYTES_PER_CYCLE, color="#344054", linestyle=(0, (6, 2)), linewidth=1.55)
    ax1.plot(oi_x, roof(oi_x, RVV_INT8_PEAK_MAC), color="#667085", linestyle=(0, (2, 2)), linewidth=1.4)
    for prec in PRECISIONS:
        ax1.hlines(rtl_compute_peak(prec), oi_x.min(), oi_x.max(), colors=prec["color"], linewidth=1.35, alpha=0.95)

    concept_pts = {
        "rvv": (4.0, 32.0),
        "base": (7.5, 60.0),
        "reuse": (13.5, 130.0),
        "final": (24.0, 250.0),
    }
    ax1.scatter([concept_pts["rvv"][0]], [concept_pts["rvv"][1]], s=120, marker="o", facecolor="white", edgecolor="#667085", linewidth=1.8, zorder=5)
    ax1.scatter([concept_pts["base"][0]], [concept_pts["base"][1]], s=124, marker="o", facecolor="#98A2B3", edgecolor="white", linewidth=1.2, zorder=5)
    ax1.scatter([concept_pts["reuse"][0]], [concept_pts["reuse"][1]], s=136, marker="D", facecolor="#8A7394", edgecolor="white", linewidth=1.2, zorder=5)
    ax1.scatter([concept_pts["final"][0]], [concept_pts["final"][1]], s=148, marker="s", facecolor="#D95B59", edgecolor="white", linewidth=1.2, zorder=5)
    ax1.annotate("", xy=concept_pts["reuse"], xytext=concept_pts["base"], arrowprops=dict(arrowstyle="->", lw=1.8, color="#8A7394"))
    ax1.annotate("", xy=concept_pts["final"], xytext=concept_pts["reuse"], arrowprops=dict(arrowstyle="->", lw=1.8, color="#D95B59"))

    for text, xy, offset, color in [
        ("RVV low-bit\nceiling", concept_pts["rvv"], (-34, -40), "#5A5A5A"),
        ("Baseline accelerator", concept_pts["base"], (10, -2), "#666666"),
        ("+ grouped reuse", concept_pts["reuse"], (12, -2), "#8A7394"),
        ("+ overlap optimizations", concept_pts["final"], (14, -2), "#D95B59"),
    ]:
        ax1.annotate(text, xy=xy, xytext=offset, textcoords="offset points", fontsize=11.2, color=color)

    ax1.text(
        0.05,
        0.95,
        "Concept before implementation",
        transform=ax1.transAxes,
        fontsize=10.8,
        va="top",
        ha="left",
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.16", fc="#F8FAFC", ec="#D9E1EA", alpha=0.95),
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.25, 128)
    ax1.set_ylim(16, 300)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax1.set_xlabel(r"Operational Intensity (MACs / Byte)", weight="bold")
    ax1.set_ylabel(r"Attained Performance (MACs / cycle)", weight="bold")
    ax1.set_title("Conceptual Roofline", loc="left", weight="bold", pad=10)
    line_label(ax1, 1.0, 6.0, "Baseline memory roof", "#98A2B3", dy=14, fs=10.6)
    line_label(ax1, 1.7, 10.0 * 1.7, "Reuse-aware memory roof", "#667085", dy=-5, fs=10.6)
    line_label(ax1, 2.05, AXI_BW_BYTES_PER_CYCLE * 2.05, "Final memory roof", "#344054", dy=14, fs=10.6)
    line_label(ax1, 5.15, RVV_INT8_PEAK_MAC, "RVV ceiling", "#667085", dy=-22, fs=10.6)
    line_label(ax1, 14.2, rtl_compute_peak(PRECISIONS[0]), "W4A8 roof", PRECISIONS[0]["color"], dy=0, fs=10.6)
    line_label(ax1, 14.2, rtl_compute_peak(PRECISIONS[1]), "W2A8 roof", PRECISIONS[1]["color"], dy=0, fs=10.6)
    line_label(ax1, 19.2, rtl_compute_peak(PRECISIONS[2]), "W1A8 roof", PRECISIONS[2]["color"], dy=0, fs=10.6)

def draw_motivation_projection_panel(ax):
    oi_x = np.logspace(np.log10(0.8), np.log10(80), 1200)
    style_axis(ax)
    ax.plot(oi_x, oi_x * AXI_BW_BYTES_PER_CYCLE, color="#344054", linestyle=(0, (6, 2)), linewidth=1.55)
    for prec in PRECISIONS:
        ax.hlines(rtl_compute_peak(prec), oi_x.min(), oi_x.max(), colors=prec["color"], linewidth=1.35, linestyle=(0, (1, 2)))

    stages = [
        {"name": "Compute-only\nscaling", "pts": [(8.5, 38), (10.5, 72), (13.0, 120)], "color": "#8E8E8E", "marker": "o"},
        {"name": "Reuse-aware\ntiling", "pts": [(14.0, 48), (18.0, 88), (24.0, 145)], "color": "#7A6C9D", "marker": "D"},
        {"name": "BitFly target\nregion", "pts": [(21.0, 56), (28.0, 105), (39.0, 175)], "color": "#D95B59", "marker": "s"},
    ]
    for stage in stages:
        xs = [p[0] for p in stage["pts"]]
        ys = [p[1] for p in stage["pts"]]
        ax.plot(xs, ys, color=stage["color"], linewidth=1.5, alpha=0.88)
        ax.scatter(xs, ys, s=112, marker=stage["marker"], facecolor=stage["color"], edgecolor="white", linewidth=1.2, zorder=4)
        ax.annotate(stage["name"], xy=(xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    fontsize=10.6, color=stage["color"], va="center")

    for prec, pt in zip(PRECISIONS, [(39.0, 56), (39.0, 105), (39.0, 175)]):
        ax.annotate(prec["name"], xy=pt, xytext=(0, -18), textcoords="offset points",
                    fontsize=9.8, color=prec["color"], ha="center")

    ax.text(
        0.64,
        0.90,
        "Projection used to guide BitFly design",
        transform=ax.transAxes,
        fontsize=9.6,
        va="top",
        ha="center",
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.16", fc="#F8FAFC", ec="#D9E1EA", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 80)
    ax.set_ylim(24, 260)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"Operational Intensity (MACs / Byte)", weight="bold")
    ax.set_ylabel(r"Attained Performance (MACs / cycle)", weight="bold")
    ax.set_title("Architecture-Aware Projection", loc="left", weight="bold", pad=10)
    line_label(ax, 1.0, AXI_BW_BYTES_PER_CYCLE * 1.0, "Memory roof", "#344054", dy=8, fs=10.4)
    line_label(ax, 1.0, rtl_compute_peak(PRECISIONS[0]), "W4A8 roof", PRECISIONS[0]["color"], dy=8, fs=10.4)
    line_label(ax, 1.0, rtl_compute_peak(PRECISIONS[1]), "W2A8 roof", PRECISIONS[1]["color"], dy=8, fs=10.4)
    line_label(ax, 1.0, rtl_compute_peak(PRECISIONS[2]), "W1A8 roof", PRECISIONS[2]["color"], dy=8, fs=10.4)


def gather_projection_points():
    cfgs = enumerate_configs()
    out = []
    for prec in PRECISIONS:
        pts = []
        for cfg in cfgs:
            oi, perf, compute_bound, cycles = design_projection_point(cfg, prec)
            pts.append(
                {
                    "cfg": cfg,
                    "oi": oi,
                    "perf": perf,
                    "compute_bound": compute_bound,
                    "reuse": cfg[2] * cfg[3],
                    "cycles": cycles,
                }
            )
        out.append((prec, pts))
    return out


def draw_projection_subplot(ax, prec, pts, shared_ylabel=False):
    style_axis(ax)

    frontier = pareto_frontier(pts)
    frontier_cfgs = {p["cfg"] for p in frontier}
    frontier_x = [p["oi"] for p in frontier]
    frontier_y = [p["perf"] for p in frontier]
    x_min, x_max = nice_bounds([p["oi"] for p in pts], lower_pad=0.9, upper_pad=1.18)
    knee_x = rtl_compute_peak(prec) / AXI_BW_BYTES_PER_CYCLE
    x_min = min(x_min, knee_x * 0.45)
    x_max = max(x_max, knee_x * 1.75)
    y_min, y_max = bounded_ylim(pts, x_min, x_max, rtl_compute_peak(prec))
    oi_x_impl = np.logspace(np.log10(x_min), np.log10(x_max), 800)
    memory_label_x = max(x_min * 1.08, knee_x * MEMORY_ROOF_LABEL_X_FACTOR)
    compute_label_x = max(x_min * 1.08, knee_x * COMPUTE_ROOF_LABEL_X_FACTOR)

    ax.plot(oi_x_impl, oi_x_impl * AXI_BW_BYTES_PER_CYCLE, color="#344054", linestyle=(0, (6, 2)), linewidth=1.45)
    ax.hlines(rtl_compute_peak(prec), oi_x_impl.min(), oi_x_impl.max(), colors=prec["color"], linewidth=1.35, linestyle=(0, (1, 2)))
    ax.plot(frontier_x, frontier_y, color=prec["color"], linewidth=1.25, alpha=0.95, zorder=3)

    for p in pts:
        is_frontier = p["cfg"] in frontier_cfgs
        size = reuse_marker_size(p["reuse"])
        ax.scatter(
            [p["oi"]],
            [p["perf"]],
            s=size,
            marker=prec["marker"],
            facecolor=REUSE_COLORS[p["reuse"]] if is_frontier else "white",
            edgecolor=REUSE_COLORS[p["reuse"]],
            linewidth=1.1 if is_frontier else 1.0,
            alpha=0.96 if is_frontier else 0.55,
            zorder=5 if is_frontier else 2,
        )

    ax.text(
        0.02,
        0.93,
        f"{PRECISION_LABELS[prec['name']]} design projection",
        transform=ax.transAxes,
        fontsize=12.6,
        va="top",
        ha="left",
        color=prec["color"],
        bbox=dict(boxstyle="round,pad=0.14", fc="#F8FAFC", ec="#D9E1EA", alpha=0.95),
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
    line_label(
        ax,
        memory_label_x,
        AXI_BW_BYTES_PER_CYCLE * memory_label_x,
        "Memory roof",
        "#344054",
        dy=MEMORY_ROOF_LABEL_DY,
        fs=11.2,
    )
    line_label(
        ax,
        compute_label_x,
        rtl_compute_peak(prec),
        "Compute roof",
        prec["color"],
        dy=COMPUTE_ROOF_LABEL_DY,
        fs=11.2,
    )


def save_motivation_figure():
    fig = plt.figure(figsize=(17.2, 5.8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.96], wspace=0.22)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    draw_conceptual_panel(ax_left)
    draw_motivation_projection_panel(ax_right)
    fig.suptitle("Roofline Motivation for BitFly", y=0.985, fontsize=18.5, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98)
    fig.savefig("roofline_motivation.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig("roofline_motivation.pdf", bbox_inches="tight", pad_inches=0.03)


def save_design_projection_figure():
    data = gather_projection_points()
    fig, axes = plt.subplots(1, 3, figsize=(18.4, 5.6), gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor("white")
    for idx, ((prec, pts), ax) in enumerate(zip(data, axes)):
        draw_projection_subplot(ax, prec, pts, shared_ylabel=(idx == 0))
    axes[1].set_title("Architecture-Aware Design-Space Projection", weight="bold", pad=12, fontsize=15.5)
    fig.text(
        0.985,
        0.885,
        "\n".join([
            r"Target shape: $(128, 8192, 2048)$",
            r"$m_t n_t g_m g_n \times 16 = 16384$",
            "Architecture-aware projection",
            "Filled: Pareto frontier",
            "Hollow: dominated design",
            "Color/size: higher reuse",
        ]),
        fontsize=10.0,
        va="top",
        ha="right",
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.16", fc="#F8FAFC", ec="#D9E1EA", alpha=0.95),
    )
    reuse_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=REUSE_COLORS[reuse],
            markeredgecolor=REUSE_COLORS[reuse],
            markersize=scatter_legend_markersize(reuse),
            label=f"gm*gn = {reuse}",
        )
        for reuse in REUSE_LEVELS
    ]
    fig.legend(
        handles=reuse_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        title=r"Reuse Level ($g_m g_n$)",
        columnspacing=1.8,
        handletextpad=0.8,
    )
    fig.suptitle("BitFly Design-Space Projection Under Target Architectural Constraints", y=0.99, fontsize=18.5, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.26, left=0.06, right=0.99)
    fig.savefig("roofline_design_projection.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig("roofline_design_projection.pdf", bbox_inches="tight", pad_inches=0.03)


def main():
    save_motivation_figure()
    save_design_projection_figure()
    print(f"Enumerated configs: {len(enumerate_configs())}")
    print("Saved figures: roofline_motivation.[png|pdf], roofline_design_projection.[png|pdf]")


if __name__ == "__main__":
    main()
