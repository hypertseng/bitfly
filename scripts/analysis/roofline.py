#!/usr/bin/env python3
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 17,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15.5,
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


ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT / "tmp"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.tiling_search import (
    Shape as SearchShape,
    better as better_search_score,
    legal_configs,
    max_feasible_ktile,
    prune_configs_for_shape,
    score_config,
)


AXI_BW_BYTES_PER_CYCLE = 16.0
STORE_QUEUE_DRAIN_BYTES_PER_CYCLE = 16.0
ACT_BITS = 8
OUT_BITS = 16
ARRAY_M = 8
ARRAY_N = 16
K_STEP = 8
EMIT_CFG_CYCLES = 1.0
STORE_BLOCK_CTRL_CYCLES = 1.0


PRECISIONS = [
    {
        "name": "Binary",
        "label": "W1A8",
        "weight_bits": 1,
        "color": "#D64E3B",
        "csv": TMP_DIR / "llm_shape_best_binary.csv",
    },
    {
        "name": "INT2",
        "label": "W2A8",
        "weight_bits": 2,
        "color": "#2A9D8F",
        "csv": TMP_DIR / "llm_shape_best_int2.csv",
    },
    {
        "name": "INT4",
        "label": "W4A8",
        "weight_bits": 4,
        "color": "#355C7D",
        "csv": TMP_DIR / "llm_shape_best_int4.csv",
    },
]

PAPER_REPRESENTATIVE_SHAPE = (128, 2048, 8192)

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def prec_to_planes(weight_bits: int) -> int:
    if weight_bits in (1, 2, 4):
        return weight_bits
    raise ValueError(f"Unsupported weight precision: {weight_bits}")


def compute_roof_macs_per_cycle(weight_bits: int) -> float:
    # Standard roofline horizontal roof: steady-state useful MAC/cycle after the
    # SA pipeline is filled. Finite ktile points can sit below this due to the
    # fixed ROWS-1 / COLS fill-drain overhead in sa.sv.
    planes = prec_to_planes(weight_bits)
    return ARRAY_M * ARRAY_N * K_STEP / float(planes)


def k_tile_lengths(total_k: int, ktile: int) -> list[int]:
    full = total_k // ktile
    rem = total_k % ktile
    tiles = [ktile] * full
    if rem:
        tiles.append(rem)
    return tiles


def marker_size_for_g(group_g: int) -> float:
    return 54.0 + 22.0 * math.log2(group_g)


def style_axis(ax) -> None:
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


def format_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def line_label(ax, x, y, text, color, dx=6, dy=0, fs=10.2) -> None:
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


def panel_y_ticks(y_min: float, y_max: float) -> list[float]:
    candidates = [
        80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200
    ]
    ticks = [float(t) for t in candidates if y_min <= t <= y_max]
    if len(ticks) >= 3:
        return ticks
    return [float(t) for t in candidates if (y_min * 0.95) <= t <= (y_max * 1.05)]


def row_to_point(row: dict[str, str], prec: dict[str, object]) -> dict[str, float | int | str]:
    m = int(row["M"])
    n = int(row["N"])
    k = int(row["K"])
    mtile = int(row["mtile"])
    ntile = int(row["ntile"])
    ktile = int(row["ktile"])
    k_iters = int(row["k_iters"])
    load_a_count = int(row["load_a_count"])
    load_w_count = int(row["load_w_count"])
    cycles_total = float(row["cycles_total"])
    cycles_compute = float(row["cycles_compute"])
    cycles_memory = float(row["cycles_memory"])
    cycles_control = float(row["cycles_control"])
    group_g = int(row["group_g"])

    k_tiles = k_tile_lengths(k, ktile)
    if len(k_tiles) != k_iters:
        raise ValueError(f"K tiling mismatch for shape {(m, n, k)}: {k_tiles} vs k_iters={k_iters}")

    m_tiles = ceil_div(m, mtile)
    n_tiles = ceil_div(n, ntile)
    total_pairs = m_tiles * n_tiles
    useful_macs = float(m * n * k)

    a_tile_bytes_sum = sum(mtile * k_exec * ACT_BITS / 8.0 for k_exec in k_tiles)
    w_tile_bytes_sum = sum(ntile * k_exec * int(prec["weight_bits"]) / 8.0 for k_exec in k_tiles)
    store_bytes = total_pairs * mtile * ntile * OUT_BITS / 8.0

    a_loads_per_k = load_a_count / float(k_iters)
    w_loads_per_k = load_w_count / float(k_iters)
    actual_load_bytes = a_loads_per_k * a_tile_bytes_sum + w_loads_per_k * w_tile_bytes_sum
    actual_total_bytes = actual_load_bytes + store_bytes

    pairwise_load_bytes = total_pairs * (a_tile_bytes_sum + w_tile_bytes_sum)
    pairwise_total_bytes = pairwise_load_bytes + store_bytes

    store_instr_count = int(row["store_instr_count"])
    pairwise_emit_cfg_count = total_pairs * k_iters
    pairwise_load_cycles = pairwise_load_bytes / AXI_BW_BYTES_PER_CYCLE
    pairwise_store_cycles = store_bytes / STORE_QUEUE_DRAIN_BYTES_PER_CYCLE
    pairwise_control_cycles = (
        pairwise_emit_cfg_count * EMIT_CFG_CYCLES
        + store_instr_count * STORE_BLOCK_CTRL_CYCLES
    )
    pairwise_total_cycles = cycles_compute + pairwise_load_cycles + pairwise_store_cycles + pairwise_control_cycles

    no_overlap_cycles = cycles_compute + cycles_memory + cycles_control

    return {
        "shape": f"({m}, {n}, {k})",
        "shape_key": (m, n, k),
        "g": group_g,
        "gm": int(row["gm"]),
        "gn": int(row["gn"]),
        "mtile": mtile,
        "ntile": ntile,
        "ktile": ktile,
        "resident_a_slots": int(row["resident_a_slots"]),
        "resident_w_slots": int(row["resident_w_slots"]),
        "resident_row_snake": int(row["resident_row_snake"]),
        "reuse_a": int(row["reuse_a"]),
        "cycles_compute": cycles_compute,
        "cycles_memory": cycles_memory,
        "cycles_control": cycles_control,
        "hidden_memory_share": float(row["hidden_memory_share"]),
        "baseline_oi": useful_macs / pairwise_total_bytes,
        "baseline_perf": useful_macs / pairwise_total_cycles,
        "reuse_oi": useful_macs / actual_total_bytes,
        "reuse_perf": useful_macs / no_overlap_cycles,
        "best_oi": useful_macs / actual_total_bytes,
        "best_perf": useful_macs / cycles_total,
        "compute_ceiling": useful_macs / cycles_compute,
        "oi_gain": pairwise_total_bytes / actual_total_bytes,
        "overlap_gain": no_overlap_cycles / cycles_total,
        "total_gain": pairwise_total_cycles / cycles_total,
        "weight_bits": int(prec["weight_bits"]),
    }


def load_points(prec: dict[str, object]) -> list[dict[str, float | int | str]]:
    csv_path = Path(prec["csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    return [row_to_point(row, prec) for row in rows]


def score_detail_to_point(
    shape: tuple[int, int, int],
    cfg,
    detail,
    prec: dict[str, object],
) -> dict[str, float | int | str]:
    m, n, k = shape
    k_tiles = k_tile_lengths(k, int(detail.ktile))
    total_pairs = ceil_div(m, cfg.mtile) * ceil_div(n, cfg.ntile)
    useful_macs = float(m * n * k)

    a_tile_bytes_sum = sum(cfg.mtile * k_exec * ACT_BITS / 8.0 for k_exec in k_tiles)
    w_tile_bytes_sum = sum(cfg.ntile * k_exec * int(prec["weight_bits"]) / 8.0 for k_exec in k_tiles)
    store_bytes = total_pairs * cfg.mtile * cfg.ntile * OUT_BITS / 8.0

    a_loads_per_k = detail.load_a_count / float(detail.k_iters)
    w_loads_per_k = detail.load_w_count / float(detail.k_iters)
    actual_load_bytes = a_loads_per_k * a_tile_bytes_sum + w_loads_per_k * w_tile_bytes_sum
    actual_total_bytes = actual_load_bytes + store_bytes

    pairwise_load_bytes = total_pairs * (a_tile_bytes_sum + w_tile_bytes_sum)
    pairwise_total_bytes = pairwise_load_bytes + store_bytes
    store_instr_count = int(detail.store_instr_count)
    pairwise_emit_cfg_count = total_pairs * int(detail.k_iters)
    pairwise_load_cycles = pairwise_load_bytes / AXI_BW_BYTES_PER_CYCLE
    pairwise_store_cycles = store_bytes / STORE_QUEUE_DRAIN_BYTES_PER_CYCLE
    pairwise_control_cycles = (
        pairwise_emit_cfg_count * EMIT_CFG_CYCLES
        + store_instr_count * STORE_BLOCK_CTRL_CYCLES
    )
    pairwise_total_cycles = float(detail.cycles_compute) + pairwise_load_cycles + pairwise_store_cycles + pairwise_control_cycles
    no_overlap_cycles = float(detail.cycles_compute) + float(detail.cycles_memory) + float(detail.cycles_control)

    return {
        "shape": f"({m}, {n}, {k})",
        "shape_key": (m, n, k),
        "g": int(detail.group_g),
        "gm": int(cfg.gm),
        "gn": int(cfg.gn),
        "mtile": int(cfg.mtile),
        "ntile": int(cfg.ntile),
        "ktile": int(detail.ktile),
        "resident_a_slots": int(detail.resident_a_slots),
        "resident_w_slots": int(detail.resident_w_slots),
        "resident_row_snake": int(detail.resident_row_snake),
        "reuse_a": int(detail.reuse_a),
        "cycles_compute": float(detail.cycles_compute),
        "cycles_memory": float(detail.cycles_memory),
        "cycles_control": float(detail.cycles_control),
        "hidden_memory_share": float(detail.hidden_memory_share),
        "baseline_oi": useful_macs / pairwise_total_bytes,
        "baseline_perf": useful_macs / pairwise_total_cycles,
        "reuse_oi": useful_macs / actual_total_bytes,
        "reuse_perf": useful_macs / no_overlap_cycles,
        "best_oi": useful_macs / actual_total_bytes,
        "best_perf": useful_macs / float(detail.cycles_total),
        "compute_ceiling": useful_macs / float(detail.cycles_compute),
        "oi_gain": pairwise_total_bytes / actual_total_bytes,
        "overlap_gain": no_overlap_cycles / float(detail.cycles_total),
        "total_gain": pairwise_total_cycles / float(detail.cycles_total),
        "weight_bits": int(prec["weight_bits"]),
    }


def pareto_frontier(points: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    ordered = sorted(points, key=lambda p: (float(p["best_oi"]), float(p["best_perf"])))
    frontier: list[dict[str, float | int | str]] = []
    best_perf = -1.0
    for p in ordered:
        perf = float(p["best_perf"])
        if perf > best_perf:
            frontier.append(p)
            best_perf = perf
    return frontier


def simplify_frontier(points: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    if not points:
        return []

    collapsed: list[dict[str, float | int | str]] = []
    bucket = [points[0]]
    for point in points[1:]:
        prev_x = float(bucket[-1]["best_oi"])
        curr_x = float(point["best_oi"])
        if abs(math.log(curr_x) - math.log(prev_x)) < 0.018:
            bucket.append(point)
            continue
        collapsed.append(max(bucket, key=lambda p: float(p["best_perf"])))
        bucket = [point]
    collapsed.append(max(bucket, key=lambda p: float(p["best_perf"])))

    simplified = [collapsed[0]]
    for point in collapsed[1:-1]:
        prev = simplified[-1]
        dx = abs(math.log(float(point["best_oi"])) - math.log(float(prev["best_oi"])))
        dy = abs(math.log(float(point["best_perf"])) - math.log(float(prev["best_perf"])))
        if dx >= 0.085 or dy >= 0.070:
            simplified.append(point)
    if collapsed[-1] is not simplified[-1]:
        simplified.append(collapsed[-1])
    return simplified


def aggregate_cloud_points(
    points: list[dict[str, float | int | str]],
    *,
    x_key: str = "best_oi",
    y_key: str = "best_perf",
) -> list[dict[str, float]]:
    buckets: dict[tuple[float, float], dict[str, float]] = {}
    for p in points:
        x = float(p[x_key])
        y = float(p[y_key])
        key = (round(x, 12), round(y, 12))
        if key not in buckets:
            buckets[key] = {"x": x, "y": y, "count": 0.0}
        buckets[key]["count"] += 1.0
    return sorted(buckets.values(), key=lambda item: (item["x"], item["y"]))


def build_search_cloud_for_shape(
    shape_key: tuple[int, int, int],
    prec: dict[str, object],
) -> dict[str, object]:
    shape = SearchShape(*shape_key)
    configs = prune_configs_for_shape(shape, legal_configs(16384))
    cloud_points: list[dict[str, float | int | str]] = []
    best_cfg = None
    best_detail = None

    for cfg in configs:
        ktile_max = max_feasible_ktile(shape.k, cfg.mtile, cfg.ntile, int(prec["weight_bits"]))
        if ktile_max is None:
            continue
        for ktile in range(K_STEP, ktile_max + K_STEP, K_STEP):
            detail = score_config(
                shape,
                cfg,
                k_block=ktile,
                prec_bits=int(prec["weight_bits"]),
                out_bits=OUT_BITS,
                bw_bytes_per_cycle=AXI_BW_BYTES_PER_CYCLE,
                group_load_model="resident_pair",
            )
            cloud_points.append(score_detail_to_point(shape_key, cfg, detail, prec))
            if best_detail is None or better_search_score(detail, best_detail):
                best_cfg = cfg
                best_detail = detail

    if best_cfg is None or best_detail is None:
        raise RuntimeError(f"No searchable configs for shape {shape_key} and precision {prec['name']}")

    winner = score_detail_to_point(shape_key, best_cfg, best_detail, prec)
    return {
        "shape_key": shape_key,
        "num_configs": len(cloud_points),
        "points": cloud_points,
        "frontier": pareto_frontier(cloud_points),
        "winner": winner,
    }


def compute_axis_bounds(
    data: list[tuple[dict[str, object], list[dict[str, float | int | str]]]]
) -> tuple[float, float, float, float]:
    all_points = [p for _, points in data for p in points]
    x_values = []
    y_values = []
    for p in all_points:
        x_values.extend([float(p["baseline_oi"]), float(p["reuse_oi"]), float(p["best_oi"])])
        y_values.extend([float(p["baseline_perf"]), float(p["reuse_perf"]), float(p["best_perf"])])
    x_min = min(x_values) * 0.80
    x_max = max(x_values) * 1.22
    y_min = min(y_values) * 0.80
    y_max = max(
        max(y_values),
        max(compute_roof_macs_per_cycle(int(prec["weight_bits"])) for prec, _ in data),
    ) * 1.14
    return x_min, x_max, y_min, y_max


def plot_roofs(
    ax,
    data: list[tuple[dict[str, object], list[dict[str, float | int | str]]]],
    *,
    x_min: float,
    x_max: float,
) -> None:
    oi_x = np.logspace(np.log10(x_min), np.log10(x_max), 600)
    ax.plot(oi_x, oi_x * AXI_BW_BYTES_PER_CYCLE, color="#344054", linestyle=(0, (6, 2)), linewidth=1.45, zorder=0)
    roof_label_x = {
        "Binary": max(x_min * 1.05, x_max * 0.18),
        "INT2": max(x_min * 1.05, x_max * 0.31),
        "INT4": max(x_min * 1.05, x_max * 0.44),
    }
    for prec, _ in data:
        roof = compute_roof_macs_per_cycle(int(prec["weight_bits"]))
        ax.hlines(roof, x_min, x_max, colors=prec["color"], linewidth=1.25, linestyle=(0, (2, 2)), alpha=0.90, zorder=0)
        line_label(
            ax,
            roof_label_x[str(prec["name"])],
            roof,
            f"{prec['label']} compute roof",
            str(prec["color"]),
            dy=8,
            fs=12.0,
        )
    line_label(ax, max(x_min * 1.06, x_max * 0.18), AXI_BW_BYTES_PER_CYCLE * max(x_min * 1.06, x_max * 0.18), "Memory roof", "#344054", dy=10, fs=12.2)


def pick_stage_centroid(
    points: list[dict[str, float | int | str]],
    x_key: str,
    y_key: str,
) -> dict[str, float | int | str]:
    target_x = float(np.median([float(p[x_key]) for p in points]))
    target_y = float(np.median([float(p[y_key]) for p in points]))
    return min(
        points,
        key=lambda p: (
            abs(math.log(float(p[x_key])) - math.log(target_x)),
            abs(math.log(float(p[y_key])) - math.log(target_y)),
        ),
    )


def dominant_config_summary(points: list[dict[str, float | int | str]]) -> str:
    keys = [
        (
            int(p["g"]),
            int(p["gm"]),
            int(p["gn"]),
            int(p["mtile"]),
            int(p["ntile"]),
            int(p["ktile"]),
            int(p["resident_a_slots"]),
            int(p["resident_w_slots"]),
        )
        for p in points
    ]
    (g, gm, gn, mtile, ntile, ktile, a_slots, w_slots), count = Counter(keys).most_common(1)[0]
    return f"{count}/{len(points)} shapes: g={g}, gmxgn={gm}x{gn}, tile={mtile}x{ntile}x{ktile}, slots={a_slots}A/{w_slots}W"


def select_shared_representative_shape(
    data: list[tuple[dict[str, object], list[dict[str, float | int | str]]]]
) -> tuple[int, int, int] | None:
    shape_sets = [
        {tuple(int(v) for v in p["shape_key"]) for p in points}
        for _, points in data
    ]
    common_shapes = set.intersection(*shape_sets) if shape_sets else set()
    if not common_shapes:
        return None

    by_prec = []
    for _, points in data:
        point_by_shape = {tuple(int(v) for v in p["shape_key"]): p for p in points}
        med_log_gain = float(np.median([math.log(float(p["total_gain"])) for p in points]))
        med_log_perf = float(np.median([math.log(float(p["best_perf"])) for p in points]))
        med_log_oi = float(np.median([math.log(float(p["best_oi"])) for p in points]))
        by_prec.append((point_by_shape, med_log_gain, med_log_perf, med_log_oi))

    def score(shape_key: tuple[int, int, int]) -> tuple[float, float]:
        total = 0.0
        for point_by_shape, med_log_gain, med_log_perf, med_log_oi in by_prec:
            p = point_by_shape[shape_key]
            total += abs(math.log(float(p["total_gain"])) - med_log_gain)
            total += abs(math.log(float(p["best_perf"])) - med_log_perf)
            total += abs(math.log(float(p["best_oi"])) - med_log_oi)
        return total, float(np.prod(shape_key))

    return min(common_shapes, key=score)


def draw_stage_panel(
    ax,
    data: list[tuple[dict[str, object], list[dict[str, float | int | str]]]],
    *,
    x_key: str,
    y_key: str,
    title: str,
    note: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    show_ylabel: bool,
) -> None:
    style_axis(ax)
    plot_roofs(ax, data, x_min=x_min, x_max=x_max)

    for prec, points in data:
        xs = [float(p[x_key]) for p in points]
        ys = [float(p[y_key]) for p in points]
        sizes = [marker_size_for_g(int(p["g"])) * 0.58 for p in points]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            marker="o",
            facecolor=prec["color"],
            edgecolor="white",
            linewidth=0.75,
            alpha=0.44,
            zorder=2,
        )
        center = pick_stage_centroid(points, x_key, y_key)
        ax.scatter(
            [float(center[x_key])],
            [float(center[y_key])],
            s=140,
            marker="o",
            facecolor=prec["color"],
            edgecolor="#0F172A",
            linewidth=1.0,
            zorder=4,
        )

    format_axis(ax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Operational Intensity (useful MACs / DRAM byte)", weight="bold")
    if show_ylabel:
        ax.set_ylabel("Attained Performance (useful MACs / cycle)", weight="bold")
    ax.set_title(title, loc="left", weight="bold", pad=10)
    ax.text(
        0.02,
        0.97,
        note,
        transform=ax.transAxes,
        fontsize=9.4,
        va="top",
        ha="left",
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.22", fc="#F8FAFC", ec="#D9E1EA", alpha=0.95),
        zorder=5,
    )


def draw_trajectory_panel(
    ax,
    data: list[tuple[dict[str, object], list[dict[str, float | int | str]]]],
    *,
    shape_key: tuple[int, int, int] | None,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    style_axis(ax)
    plot_roofs(ax, data, x_min=x_min, x_max=x_max)

    fallback_offsets = {
        "Binary": (12, 12),
        "INT2": (10, -18),
        "INT4": (12, -10),
    }

    for prec, points in data:
        if shape_key is None:
            rep = pick_stage_centroid(points, "best_oi", "best_perf")
        else:
            rep = next(p for p in points if tuple(int(v) for v in p["shape_key"]) == shape_key)

        baseline = (float(rep["baseline_oi"]), float(rep["baseline_perf"]))
        reuse = (float(rep["reuse_oi"]), float(rep["reuse_perf"]))
        best = (float(rep["best_oi"]), float(rep["best_perf"]))

        ax.annotate(
            "",
            xy=reuse,
            xytext=baseline,
            arrowprops=dict(arrowstyle="-|>", color=prec["color"], lw=2.0, alpha=0.80, shrinkA=3, shrinkB=3),
            zorder=3,
        )
        ax.annotate(
            "",
            xy=best,
            xytext=reuse,
            arrowprops=dict(arrowstyle="-|>", color=prec["color"], lw=2.0, alpha=0.96, shrinkA=3, shrinkB=3),
            zorder=3,
        )
        ax.scatter([baseline[0]], [baseline[1]], s=74, marker="o", facecolor="white", edgecolor=prec["color"], linewidth=1.5, zorder=4)
        ax.scatter([reuse[0]], [reuse[1]], s=86, marker="X", color=prec["color"], linewidth=1.0, zorder=5)
        ax.scatter([best[0]], [best[1]], s=126, marker="o", facecolor=prec["color"], edgecolor="white", linewidth=1.2, zorder=6)

        reuse_mid_x = math.sqrt(baseline[0] * reuse[0])
        reuse_mid_y = math.sqrt(baseline[1] * reuse[1])
        overlap_mid_x = math.sqrt(reuse[0] * best[0])
        overlap_mid_y = math.sqrt(reuse[1] * best[1])
        ax.text(reuse_mid_x, reuse_mid_y * 1.03, f"{float(rep['oi_gain']):.2f}x OI", fontsize=9.4, color=prec["color"], ha="center", va="bottom")
        ax.text(overlap_mid_x * 1.01, overlap_mid_y * 1.03, f"{float(rep['overlap_gain']):.2f}x perf", fontsize=9.4, color=prec["color"], ha="left", va="bottom")

        ax.annotate(
            f"{prec['label']} {rep['shape']}\n"
            f"g={int(rep['g'])}, gmxgn={int(rep['gm'])}x{int(rep['gn'])}, "
            f"tile={int(rep['mtile'])}x{int(rep['ntile'])}x{int(rep['ktile'])}",
            xy=best,
            xytext=fallback_offsets[str(prec["name"])],
            textcoords="offset points",
            fontsize=9.6,
            color=prec["color"],
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=prec["color"], alpha=0.94),
            arrowprops=dict(arrowstyle="-", color=prec["color"], lw=1.0, alpha=0.90),
            zorder=7,
        )

    summaries = [
        f"{prec['label']}: {dominant_config_summary(points)}"
        for prec, points in data
    ]
    ax.text(
        0.02,
        0.05,
        "\n".join(summaries),
        transform=ax.transAxes,
        fontsize=9.2,
        va="bottom",
        ha="left",
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.24", fc="#F8FAFC", ec="#D9E1EA", alpha=0.95),
        zorder=8,
    )

    format_axis(ax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Operational Intensity (useful MACs / DRAM byte)", weight="bold")
    ax.set_ylabel("Attained Performance (useful MACs / cycle)", weight="bold")
    ax.set_title("D. Representative Searched Trajectory", loc="left", weight="bold", pad=10)

    stage_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#667085", markersize=7.3, label="Pairwise baseline"),
        plt.Line2D([0], [0], marker="X", color="#667085", linewidth=0, markersize=7.3, label="Template reuse, no overlap"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#667085", markeredgecolor="white", markersize=7.6, label="RTL-aware searched best"),
    ]
    precision_handles = [plt.Line2D([0], [0], color=prec["color"], lw=2.4, label=prec["label"]) for prec, _ in data]
    stage_legend = ax.legend(handles=stage_handles, loc="lower right", title="Stage", fontsize=9.3, title_fontsize=9.6)
    ax.add_artist(stage_legend)
    ax.legend(handles=precision_handles, loc="upper right", title="Precision", fontsize=9.3, title_fontsize=9.6)


def build_precision_summary(
    prec: dict[str, object],
    points: list[dict[str, float | int | str]],
) -> str:
    dominant = dominant_config_summary(points)
    median_speedup = float(np.median([float(p["total_gain"]) for p in points]))
    rep = pick_stage_centroid(points, "best_oi", "best_perf")
    return (
        f"{prec['label']}: rep {rep['shape']}, median {median_speedup:.2f}x, "
        f"{dominant}"
    )


def plot_precision_roofs(
    ax,
    prec: dict[str, object],
    *,
    x_min: float,
    x_max: float,
    show_memory_label: bool,
) -> None:
    oi_x = np.logspace(np.log10(x_min), np.log10(x_max), 600)
    raw_y = oi_x * AXI_BW_BYTES_PER_CYCLE
    roof = compute_roof_macs_per_cycle(int(prec["weight_bits"]))

    ax.plot(oi_x, raw_y, color="#344054", linestyle=(0, (6, 2)), linewidth=1.35, zorder=0)
    ax.hlines(roof, x_min, x_max, colors=prec["color"], linewidth=1.25, linestyle=(0, (2, 2)), alpha=0.90, zorder=0)

    comp_label_x = x_max / 1.17
    if show_memory_label:
        mem_label_x = max(x_min * 1.06, x_max * 0.105)
        line_label(ax, mem_label_x, AXI_BW_BYTES_PER_CYCLE * mem_label_x, "DRAM roof", "#344054", dy=-16, fs=11.8)
    ax.text(
        comp_label_x,
        roof * 1.01,
        "Compute roof",
        fontsize=11.8,
        color=str(prec["color"]),
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
        clip_on=True,
        zorder=5,
    )


def draw_precision_search_panel(
    ax,
    prec: dict[str, object],
    best_points: list[dict[str, float | int | str]],
    cloud: dict[str, object],
    *,
    shape_key: tuple[int, int, int],
    show_ylabel: bool,
    show_memory_label: bool,
    panel_tag: str,
    shared_x_bounds: tuple[float, float] | None,
) -> None:
    style_axis(ax)
    winner = cloud["winner"]
    frontier = cloud["frontier"]
    points = cloud["points"]
    aggregated_points = aggregate_cloud_points(points)

    x_values = [float(p["best_oi"]) for p in points] + [
        float(winner["baseline_oi"]),
        float(winner["reuse_oi"]),
        float(winner["best_oi"]),
    ]
    y_values = [float(p["best_perf"]) for p in points] + [
        float(winner["baseline_perf"]),
        float(winner["reuse_perf"]),
        float(winner["best_perf"]),
    ]
    x_min = min(x_values) * 0.78
    x_max = max(x_values) * 1.18
    y_min = min(y_values) * 0.78
    y_max = max(max(y_values), compute_roof_macs_per_cycle(int(prec["weight_bits"]))) * 1.14
    if shared_x_bounds is not None:
        x_min, x_max = shared_x_bounds
    roof_knee = compute_roof_macs_per_cycle(int(prec["weight_bits"])) / AXI_BW_BYTES_PER_CYCLE
    knee_pad = 1.45 if int(prec["weight_bits"]) == 1 else 1.10
    x_max = max(x_max, roof_knee * knee_pad)

    plot_precision_roofs(ax, prec, x_min=x_min, x_max=x_max, show_memory_label=show_memory_label)

    ax.scatter(
        [float(p["x"]) for p in aggregated_points],
        [float(p["y"]) for p in aggregated_points],
        s=[10.0 + 12.0 * math.log2(float(p["count"]) + 1.0) for p in aggregated_points],
        marker="o",
        facecolor="#CBD3DD",
        edgecolor="white",
        linewidth=0.24,
        alpha=0.22,
        zorder=2,
    )
    ax.scatter(
        [float(p["best_oi"]) for p in frontier],
        [float(p["best_perf"]) for p in frontier],
        s=12,
        marker="o",
        facecolor=prec["color"],
        edgecolor="white",
        linewidth=0.40,
        alpha=0.98,
        zorder=3,
    )

    baseline = (float(winner["baseline_oi"]), float(winner["baseline_perf"]))
    reuse = (float(winner["reuse_oi"]), float(winner["reuse_perf"]))
    best = (float(winner["best_oi"]), float(winner["best_perf"]))

    ax.annotate("", xy=reuse, xytext=baseline, arrowprops=dict(arrowstyle="-|>", color=prec["color"], lw=2.15, alpha=0.82, shrinkA=3, shrinkB=3), zorder=4)
    ax.annotate("", xy=best, xytext=reuse, arrowprops=dict(arrowstyle="-|>", color=prec["color"], lw=2.15, alpha=0.96, shrinkA=3, shrinkB=3), zorder=4)
    ax.scatter([baseline[0]], [baseline[1]], s=76, marker="o", facecolor="white", edgecolor=prec["color"], linewidth=1.45, zorder=5)
    ax.scatter([reuse[0]], [reuse[1]], s=88, marker="X", color=prec["color"], linewidth=1.1, zorder=6)
    ax.scatter([best[0]], [best[1]], s=130, marker="o", facecolor=prec["color"], edgecolor="white", linewidth=1.0, zorder=7)

    reuse_mid_x = math.sqrt(baseline[0] * reuse[0])
    reuse_mid_y = math.sqrt(baseline[1] * reuse[1])
    overlap_mid_x = math.sqrt(reuse[0] * best[0])
    overlap_mid_y = math.sqrt(reuse[1] * best[1])
    oi_label_offsets = {
        "Binary": (22, -26),
        "INT2": (-8, 12),
        "INT4": (-10, 12),
    }
    perf_label_offsets = {
        "Binary": (12, 12),
        "INT2": (12, 10),
        "INT4": (12, 10),
    }
    ax.annotate(
        f"{float(winner['oi_gain']):.2f}x OI",
        xy=(reuse_mid_x, reuse_mid_y),
        xytext=oi_label_offsets[str(prec["name"])],
        textcoords="offset points",
        fontsize=12.4,
        color=prec["color"],
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.82),
        zorder=8,
    )
    ax.annotate(
        f"{float(winner['overlap_gain']):.2f}x perf",
        xy=(overlap_mid_x, overlap_mid_y),
        xytext=perf_label_offsets[str(prec["name"])],
        textcoords="offset points",
        fontsize=12.4,
        color=prec["color"],
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.82),
        zorder=8,
    )

    format_axis(ax)
    x_ticks = [tick for tick in (8, 12, 16, 24, 32, 48, 64, 80, 96) if x_min <= tick <= x_max]
    if x_ticks:
        ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    y_ticks = panel_y_ticks(y_min, y_max)
    if y_ticks:
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{panel_tag} {prec['label']}", loc="left", weight="bold", pad=14)


def save_search_roofline_figure() -> None:
    data = [(prec, load_points(prec)) for prec in PRECISIONS]
    shared_shapes = set.intersection(*[
        {tuple(int(v) for v in p["shape_key"]) for p in points}
        for _, points in data
    ])
    representative_shape = PAPER_REPRESENTATIVE_SHAPE
    if representative_shape not in shared_shapes:
        raise RuntimeError(f"Paper shape {representative_shape} is not shared across precisions")
    search_clouds = {
        str(prec["name"]): build_search_cloud_for_shape(representative_shape, prec)
        for prec, _ in data
    }
    shared_x_values: list[float] = []
    for cloud in search_clouds.values():
        winner = cloud["winner"]
        shared_x_values.extend(float(p["best_oi"]) for p in cloud["points"])
        shared_x_values.extend([
            float(winner["baseline_oi"]),
            float(winner["reuse_oi"]),
            float(winner["best_oi"]),
        ])
    shared_x_bounds = (
        min(shared_x_values) * 0.78,
        max(shared_x_values) * 1.18,
    )

    fig = plt.figure(figsize=(17.6, 6.9))
    fig.patch.set_facecolor("white")
    axes = fig.subplots(1, 3)

    panel_tags = ["(a)", "(b)", "(c)"]
    for idx, ((prec, best_points), ax) in enumerate(zip(data, axes)):
        draw_precision_search_panel(
            ax,
            prec,
            best_points,
            search_clouds[str(prec["name"])],
            shape_key=representative_shape,
            show_ylabel=(idx == 0),
            show_memory_label=(idx == 0),
            panel_tag=panel_tags[idx],
            shared_x_bounds=shared_x_bounds,
        )

    fig.text(
        0.5,
        0.965,
        "Roofline Search Space at Shape (128, 2048, 8192)",
        ha="center",
        va="top",
        fontsize=20.2,
        fontweight="bold",
        color="#344054",
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#CBD3DD", markeredgecolor="white", markeredgewidth=0.5, markersize=7.2, alpha=0.8, label="Legal configs"),
        plt.Line2D([0], [0], marker="o", color="#667085", markerfacecolor="#667085", markeredgecolor="white", markeredgewidth=0.5, lw=0, alpha=0.95, markersize=4.8, label="Pareto frontier"),
        plt.Line2D([0], [0], color="#344054", lw=1.35, linestyle=(0, (6, 2)), label="DRAM roof"),
        plt.Line2D([0], [0], color="#667085", lw=1.25, linestyle=(0, (2, 2)), label="Compute roof"),
        plt.Line2D([0], [0], color="#667085", lw=0, marker="o", markerfacecolor="white", markeredgecolor="#667085", markersize=6.8, label="Winner baseline"),
        plt.Line2D([0], [0], color="#667085", lw=0, marker="X", markerfacecolor="#667085", markeredgecolor="#667085", markersize=7.0, label="Winner + reuse"),
        plt.Line2D([0], [0], color="#667085", lw=0, marker="o", markerfacecolor="#667085", markeredgecolor="white", markeredgewidth=0.8, markersize=7.2, label="Winner best"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        fontsize=12.8,
        columnspacing=1.0,
        handlelength=1.9,
        handletextpad=0.6,
    )

    fig.supxlabel("Operational Intensity (useful MACs / DRAM byte)", y=0.10, fontsize=20.2, fontweight="bold")
    fig.supylabel("Attained Performance (useful MACs / cycle)", x=0.02, fontsize=20.2, fontweight="bold")

    fig.subplots_adjust(top=0.845, bottom=0.22, left=0.09, right=0.99, wspace=0.24)

    outputs = [
        ROOT / "roofline_search_results.png",
        ROOT / "roofline_search_results.pdf",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    save_search_roofline_figure()
    print("Saved figures:")
    print("  roofline_search_results.[png|pdf]")


if __name__ == "__main__":
    main()
