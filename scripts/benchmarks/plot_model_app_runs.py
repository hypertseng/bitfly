import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


MODEL_ORDER = [
    "gemma3_270m",
    "smollm2_360m",
    "qwen25_05b",
    "tinyllama_11b",
    "opt_13b",
    "qwen25_15b",
    "stablelm2_16b",
    "smollm2_17b",
    "gemma2_2b",
    "replit_code_v1_3b",
]

MODEL_LABELS = {
    "gemma3_270m": "Gemma 3\n270M",
    "smollm2_360m": "SmolLM2\n360M",
    "qwen25_05b": "Qwen2.5\n0.5B",
    "tinyllama_11b": "TinyLlama\n1.1B",
    "opt_13b": "OPT\n1.3B",
    "qwen25_15b": "Qwen2.5\n1.5B",
    "stablelm2_16b": "StableLM 2\n1.6B",
    "smollm2_17b": "SmolLM2\n1.7B",
    "gemma2_2b": "Gemma 2\n2B",
    "replit_code_v1_3b": "Replit Code\n3B",
}

MODEL_LABELS_LINE = {
    "gemma3_270m": "Gemma 3 270M",
    "smollm2_360m": "SmolLM2 360M",
    "qwen25_05b": "Qwen2.5 0.5B",
    "tinyllama_11b": "TinyLlama 1.1B",
    "opt_13b": "OPT 1.3B",
    "qwen25_15b": "Qwen2.5 1.5B",
    "stablelm2_16b": "StableLM 2 1.6B",
    "smollm2_17b": "SmolLM2 1.7B",
    "gemma2_2b": "Gemma 2 2B",
    "replit_code_v1_3b": "Replit Code 3B",
}

MODEL_LABELS_COMPACT = {
    "gemma3_270m": "Gemma3-270M",
    "smollm2_360m": "SmolLM2-360M",
    "qwen25_05b": "Qwen2.5-0.5B",
    "tinyllama_11b": "TinyLlama-1.1B",
    "opt_13b": "OPT-1.3B",
    "qwen25_15b": "Qwen2.5-1.5B",
    "stablelm2_16b": "StableLM2-1.6B",
    "smollm2_17b": "SmolLM2-1.7B",
    "gemma2_2b": "Gemma2-2B",
    "replit_code_v1_3b": "ReplitCode-3B",
}

MODEL_LABELS_TIGHT = {
    "gemma3_270m": "G3-270M",
    "smollm2_360m": "S2-360M",
    "qwen25_05b": "Q0.5B",
    "tinyllama_11b": "T1.1B",
    "opt_13b": "O1.3B",
    "qwen25_15b": "Q1.5B",
    "stablelm2_16b": "S1.6B",
    "smollm2_17b": "S1.7B",
    "gemma2_2b": "G2-2B",
    "replit_code_v1_3b": "R3B",
}

PRECISION_ORDER = ["binary", "INT2", "INT4"]
PRECISION_TITLES = {"binary": "Binary", "INT2": "INT2", "INT4": "INT4"}

RUN_COLORS = {
    "rvv": {"face": "#D7DEE8", "edge": "#64748B"},
    "bmpmm": {"face": "#F97316", "edge": "#9A3412"},
}

SPEEDUP_COLORS = {"pos": "#EA580C", "neg": "#B91C1C"}
SINGLE_COLORS = {"binary": "#FED7AA", "INT2": "#FB923C", "INT4": "#C2410C"}

TOTAL_RE = re.compile(r"\[(bmpmm_[^\]]+|rvv_[^\]]+)\] model_total model=(.+?) (bmpmm_cycles|rvv_cycles)=(\d+)")
MODEL_RE = re.compile(r"\[[^\]]+\] model=(.+?) scale=")


@dataclass
class AppRecord:
    app: str
    arch: str
    precision: str
    model_key: str
    model_id: str
    cycles: int
    source_run: str
    source_log: str


def parse_args():
    parser = argparse.ArgumentParser(description="Merge model-app benchmark logs and plot a paper-ready wide figure.")
    parser.add_argument("--run-dir", dest="run_dirs", action="append", required=True, help="Run directory. Later runs override earlier runs for the same app.")
    parser.add_argument("--output-prefix", required=True, help="Output path prefix, without extension.")
    parser.add_argument(
        "--layout",
        choices=["wide", "single-column", "single-column-camera", "compact-3col", "compact-3col-hbar", "conference-3col"],
        default="wide",
        help="Figure layout. Use single-column for narrow paper figures.",
    )
    parser.add_argument("--title", default="BitFly Acceleration over RVV on Transformer Linear Layers", help="Figure title.")
    parser.add_argument(
        "--subtitle",
        default="Each point aggregates the seven linear layers from transformer block 0 for one model and precision.",
        help="Figure subtitle.",
    )
    parser.add_argument(
        "--footer",
        default=None,
        help="Optional footer. Pass an empty string to suppress it entirely; omit to auto-derive from input runs.",
    )
    return parser.parse_args()


def load_run(run_dir: Path) -> Dict[str, AppRecord]:
    records = {}
    for log_path in sorted(run_dir.glob("batch_*/*.log")):
        text = log_path.read_text(errors="ignore")
        total_match = TOTAL_RE.search(text)
        if not total_match:
            continue
        model_match = MODEL_RE.search(text)
        app = log_path.stem
        parts = app.split("_", 2)
        records[app] = AppRecord(
            app=app,
            arch=parts[0],
            precision=parts[1],
            model_key=parts[2],
            model_id=model_match.group(1) if model_match else parts[2],
            cycles=int(total_match.group(4)),
            source_run=run_dir.name,
            source_log=str(log_path),
        )
    return records


def merge_runs(run_dirs: List[Path]) -> Dict[str, AppRecord]:
    merged = {}
    for run_dir in run_dirs:
        merged.update(load_run(run_dir))
    return merged


def validate_records(records: Dict[str, AppRecord]):
    expected = len(MODEL_ORDER) * len(PRECISION_ORDER) * 2
    if len(records) != expected:
        raise ValueError(f"Expected {expected} app results after merge, found {len(records)}")

    for precision in PRECISION_ORDER:
        for model in MODEL_ORDER:
            for arch in ("rvv", "bmpmm"):
                app = f"{arch}_{precision}_{model}"
                if app not in records:
                    raise ValueError(f"Missing expected app result: {app}")


def write_csv(records: Dict[str, AppRecord], output_prefix: Path):
    row_path = output_prefix.with_name(output_prefix.name + "_rows.csv")
    pair_path = output_prefix.with_name(output_prefix.name + "_pairs.csv")
    speedup_path = output_prefix.with_name(output_prefix.name + "_speedup_summary.csv")

    row_fields = [
        "app",
        "architecture",
        "precision",
        "model_key",
        "model_label",
        "model_id",
        "cycles",
        "source_run",
        "source_log",
    ]
    with row_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_fields)
        writer.writeheader()
        for app in sorted(records):
            record = records[app]
            writer.writerow(
                {
                    "app": record.app,
                    "architecture": "BitFly" if record.arch == "bmpmm" else "RVV",
                    "precision": record.precision,
                    "model_key": record.model_key,
                    "model_label": MODEL_LABELS[record.model_key].replace("\n", " "),
                    "model_id": record.model_id,
                    "cycles": record.cycles,
                    "source_run": record.source_run,
                    "source_log": record.source_log,
                }
            )

    pair_fields = [
        "precision",
        "model_key",
        "model_label",
        "model_id",
        "rvv_cycles",
        "bitfly_cycles",
        "speedup_rvv_over_bitfly",
        "rvv_source_run",
        "bitfly_source_run",
    ]
    with pair_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields)
        writer.writeheader()
        for precision in PRECISION_ORDER:
            for model in MODEL_ORDER:
                rvv = records[f"rvv_{precision}_{model}"]
                bmpmm = records[f"bmpmm_{precision}_{model}"]
                writer.writerow(
                    {
                        "precision": precision,
                        "model_key": model,
                        "model_label": MODEL_LABELS[model].replace("\n", " "),
                        "model_id": rvv.model_id,
                        "rvv_cycles": rvv.cycles,
                        "bitfly_cycles": bmpmm.cycles,
                        "speedup_rvv_over_bitfly": f"{rvv.cycles / bmpmm.cycles:.6f}",
                        "rvv_source_run": rvv.source_run,
                        "bitfly_source_run": bmpmm.source_run,
                    }
                )

    speedup_fields = [
        "precision",
        "mean_speedup_rvv_over_bitfly",
        "geomean_speedup_rvv_over_bitfly",
        "max_speedup_rvv_over_bitfly",
        "best_model",
    ]
    with speedup_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=speedup_fields)
        writer.writeheader()
        for precision in PRECISION_ORDER:
            speedups = []
            best_model = ""
            best_speedup = -1.0
            for model in MODEL_ORDER:
                rvv = records[f"rvv_{precision}_{model}"]
                bmpmm = records[f"bmpmm_{precision}_{model}"]
                speedup = rvv.cycles / bmpmm.cycles
                speedups.append(speedup)
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_model = model
            writer.writerow(
                {
                    "precision": precision,
                    "mean_speedup_rvv_over_bitfly": f"{sum(speedups) / len(speedups):.6f}",
                    "geomean_speedup_rvv_over_bitfly": f"{math.exp(sum(math.log(v) for v in speedups) / len(speedups)):.6f}",
                    "max_speedup_rvv_over_bitfly": f"{max(speedups):.6f}",
                    "best_model": best_model,
                }
            )


def format_cycles(value, _):
    if value <= 0:
        return "0"
    units = [(1e9, "B"), (1e6, "M"), (1e3, "K")]
    for scale, suffix in units:
        if value >= scale:
            scaled = value / scale
            if scaled >= 100:
                return f"{scaled:.0f}{suffix}"
            if scaled >= 10:
                return f"{scaled:.1f}{suffix}"
            return f"{scaled:.2f}{suffix}"
    return f"{value:.0f}"


def format_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}x"
    return f"{value:.2f}x"


def apply_axes_style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


def add_stripes(ax, n_models: int):
    for idx in range(n_models):
        if idx % 2 == 0:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#F8FAFC", zorder=0)


def plot(records: Dict[str, AppRecord], output_prefix: Path, title: str, subtitle: str, footer: str):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 11.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.5,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(19.5, 5.95))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.95, 1.35], hspace=0.04, wspace=0.18)
    x = np.arange(len(MODEL_ORDER))

    for col, precision in enumerate(PRECISION_ORDER):
        ax_cycles = fig.add_subplot(gs[0, col])
        ax_speedup = fig.add_subplot(gs[1, col], sharex=ax_cycles)
        apply_axes_style(ax_cycles)
        apply_axes_style(ax_speedup)
        add_stripes(ax_cycles, len(MODEL_ORDER))
        add_stripes(ax_speedup, len(MODEL_ORDER))

        rvv_cycles = np.array([records[f"rvv_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        bmpmm_cycles = np.array([records[f"bmpmm_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        speedups = rvv_cycles / bmpmm_cycles

        for idx, (rvv, bmpmm) in enumerate(zip(rvv_cycles, bmpmm_cycles)):
            ax_cycles.plot([idx, idx], [min(rvv, bmpmm), max(rvv, bmpmm)], color="#CBD5E1", linewidth=2.2, zorder=1)

        ax_cycles.scatter(
            x - 0.075,
            rvv_cycles,
            s=62,
            marker="s",
            color=RUN_COLORS["rvv"]["face"],
            edgecolor=RUN_COLORS["rvv"]["edge"],
            linewidth=1.0,
            zorder=3,
            label="RVV" if col == 0 else None,
        )
        ax_cycles.scatter(
            x + 0.075,
            bmpmm_cycles,
            s=68,
            marker="o",
            color=RUN_COLORS["bmpmm"]["face"],
            edgecolor=RUN_COLORS["bmpmm"]["edge"],
            linewidth=1.0,
            zorder=4,
            label="BitFly" if col == 0 else None,
        )

        ymin = min(rvv_cycles.min(), bmpmm_cycles.min()) / 1.85
        ymax = max(rvv_cycles.max(), bmpmm_cycles.max()) * 1.85
        ax_cycles.set_yscale("log")
        ax_cycles.set_ylim(ymin, ymax)
        ax_cycles.yaxis.set_major_formatter(FuncFormatter(format_cycles))
        ax_cycles.grid(axis="y", color="#E2E8F0", linestyle=(0, (2, 3)), linewidth=0.8)
        ax_cycles.set_axisbelow(True)
        ax_cycles.tick_params(axis="x", length=0, labelbottom=False)
        ax_cycles.set_title(PRECISION_TITLES[precision], pad=10, fontweight="bold", color="#0F172A")
        if col == 0:
            ax_cycles.set_ylabel("Aggregated Runtime (cycles, log scale)", labelpad=8)

        mean_speedup = speedups.mean()
        best_idx = int(np.argmax(speedups))
        ax_cycles.text(
            0.02,
            0.97,
            f"avg {mean_speedup:.1f}x, peak {speedups[best_idx]:.1f}x\nbest on {MODEL_LABELS[MODEL_ORDER[best_idx]].replace(chr(10), ' ')}",
            transform=ax_cycles.transAxes,
            ha="left",
            va="top",
            fontsize=10.2,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFFFF", "edgecolor": "#E2E8F0", "linewidth": 0.8},
        )

        bar_colors = [SPEEDUP_COLORS["pos"] if v >= 1.0 else SPEEDUP_COLORS["neg"] for v in speedups]
        ax_speedup.bar(x, speedups, width=0.62, color=bar_colors, edgecolor="#7C2D12", linewidth=0.6, zorder=3)
        ax_speedup.grid(axis="y", color="#E2E8F0", linestyle=(0, (2, 3)), linewidth=0.8)
        ax_speedup.set_axisbelow(True)
        ax_speedup.set_xticks(x)
        ax_speedup.set_xticklabels(
            [MODEL_LABELS_COMPACT[model] for model in MODEL_ORDER],
            rotation=48,
            ha="right",
            rotation_mode="anchor",
        )
        ax_speedup.tick_params(axis="x", length=0, pad=12)
        if col == 0:
            ax_speedup.set_ylabel("RVV / BitFly", labelpad=8)

        if precision == "binary":
            ax_speedup.axhline(1.0, color="#475569", linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
            ax_speedup.set_ylim(0, max(2.1, speedups.max() * 1.28))
        else:
            ax_speedup.set_ylim(0, speedups.max() * 1.24)

        offset = ax_speedup.get_ylim()[1] * 0.02
        for idx, speedup in enumerate(speedups):
            ax_speedup.text(
                idx,
                speedup + offset,
                format_speedup(speedup),
                ha="center",
                va="bottom",
                fontsize=7.6,
                color="#0F172A",
                rotation=0,
            )

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=RUN_COLORS["rvv"]["face"], markeredgecolor=RUN_COLORS["rvv"]["edge"], markersize=8, label="RVV baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RUN_COLORS["bmpmm"]["face"], markeredgecolor=RUN_COLORS["bmpmm"]["edge"], markersize=8, label="BitFly"),
        Line2D([0], [0], color="#CBD5E1", linewidth=2.2, label="paired model result"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.978))
    fig.suptitle(title, y=0.996, fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.948,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.2,
        color="#475569",
    )
    if footer:
        fig.text(
            0.5,
            0.012,
            footer,
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#64748B",
        )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)


def plot_single_column(records: Dict[str, AppRecord], output_prefix: Path, title: str, subtitle: str, footer: str):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.2,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.3,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(3.45, 7.35))
    fig.patch.set_facecolor("white")
    y = np.arange(len(MODEL_ORDER))
    labels = [MODEL_LABELS_COMPACT[m] for m in MODEL_ORDER]

    for ax, precision in zip(axes, PRECISION_ORDER):
        apply_axes_style(ax)
        ax.grid(axis="x", color="#E2E8F0", linestyle=(0, (2, 3)), linewidth=0.75)
        ax.set_axisbelow(True)

        rvv_cycles = np.array([records[f"rvv_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        bmpmm_cycles = np.array([records[f"bmpmm_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        speedups = rvv_cycles / bmpmm_cycles
        geomean = math.exp(np.log(speedups).mean())
        mean_speedup = speedups.mean()

        ax.barh(
            y,
            speedups,
            height=0.68,
            color=SINGLE_COLORS[precision],
            edgecolor="#7C2D12",
            linewidth=0.7,
            zorder=3,
        )
        ax.axvline(1.0, color="#94A3B8", linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
        ax.axvline(geomean, color="#0F172A", linewidth=1.0, linestyle=(0, (5, 2)), zorder=2)

        xmax = max(speedups.max() * 1.22, 2.0)
        ax.set_xlim(0, xmax)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_title(PRECISION_TITLES[precision], loc="left", pad=3, fontweight="bold", color="#0F172A")

        for idx, speedup in enumerate(speedups):
            ax.text(
                min(speedup + xmax * 0.018, xmax * 0.985),
                idx,
                f"{speedup:.2f}x",
                ha="left",
                va="center",
                fontsize=6.9,
                color="#111827",
            )

        ax.text(
            0.985,
            0.05,
            f"GM {geomean:.2f}x | AVG {mean_speedup:.2f}x",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.9,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFF7ED", "edgecolor": "#FED7AA", "linewidth": 0.7},
        )

    axes[-1].set_xlabel("Speedup over RVV (RVV cycles / BitFly cycles)")
    fig.suptitle(title, y=0.992, fontsize=11.2, fontweight="bold")
    fig.text(0.5, 0.965, subtitle, ha="center", va="center", fontsize=7.7, color="#475569")
    if footer:
        fig.text(0.5, 0.006, footer, ha="center", va="bottom", fontsize=6.4, color="#64748B")

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.955])
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)


def plot_single_column_camera(records: Dict[str, AppRecord], output_prefix: Path, title: str, subtitle: str, footer: str):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 7.0))
    fig.patch.set_facecolor("white")
    y = np.arange(len(MODEL_ORDER))
    labels = [MODEL_LABELS_COMPACT[m] for m in MODEL_ORDER]
    face_colors = {"binary": "#E5E7EB", "INT2": "#9CA3AF", "INT4": "#374151"}

    for ax, precision in zip(axes, PRECISION_ORDER):
        apply_axes_style(ax)
        ax.grid(axis="x", color="#E5E7EB", linestyle="-", linewidth=0.6)
        ax.set_axisbelow(True)

        rvv_cycles = np.array([records[f"rvv_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        bmpmm_cycles = np.array([records[f"bmpmm_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        speedups = rvv_cycles / bmpmm_cycles
        geomean = math.exp(np.log(speedups).mean())

        ax.barh(
            y,
            speedups,
            height=0.60,
            color=face_colors[precision],
            edgecolor="#111827",
            linewidth=0.6,
            zorder=3,
        )
        ax.axvline(1.0, color="#6B7280", linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        ax.axvline(geomean, color="#111827", linewidth=1.0, linestyle=(0, (5, 2)), zorder=2)

        xmax = max(speedups.max() * 1.16, 2.0)
        ax.set_xlim(0, xmax)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{PRECISION_TITLES[precision]}", loc="left", pad=2, fontweight="bold", color="#111827")

        for idx, speedup in enumerate(speedups):
            ax.text(
                min(speedup + xmax * 0.015, xmax * 0.985),
                idx,
                f"{speedup:.2f}x",
                ha="left",
                va="center",
                fontsize=6.6,
                color="#111827",
            )

        ax.text(
            0.985,
            0.04,
            f"geomean {geomean:.2f}x",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.7,
            color="#111827",
        )

    axes[-1].set_xlabel("Speedup over RVV")
    fig.suptitle(title, y=0.99, fontsize=10.7, fontweight="bold")
    fig.text(0.5, 0.962, subtitle, ha="center", va="center", fontsize=7.2, color="#374151")
    if footer:
        fig.text(0.5, 0.006, footer, ha="center", va="bottom", fontsize=6.1, color="#6B7280")

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)


def plot_compact_3col(records: Dict[str, AppRecord], output_prefix: Path, title: str, subtitle: str, footer: str):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )

    panel_colors = {"binary": "#F97316", "INT2": "#0EA5E9", "INT4": "#14B8A6"}
    edge_colors = {"binary": "#9A3412", "INT2": "#0C4A6E", "INT4": "#134E4A"}

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharey=False)
    fig.patch.set_facecolor("white")
    x = np.arange(len(MODEL_ORDER))
    speedup_map = {}
    for precision in PRECISION_ORDER:
        rvv_cycles = np.array([records[f"rvv_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        bmpmm_cycles = np.array([records[f"bmpmm_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        speedups = rvv_cycles / bmpmm_cycles
        speedup_map[precision] = speedups

    for idx, precision in enumerate(PRECISION_ORDER):
        ax = axes[idx]
        apply_axes_style(ax)
        ax.grid(axis="y", color="#E2E8F0", linestyle=(0, (2, 3)), linewidth=0.65)
        ax.set_axisbelow(True)

        speedups = speedup_map[precision]
        geomean = math.exp(np.log(speedups).mean())
        mean_speedup = speedups.mean()
        ymax = max(float(speedups.max()) * 1.10, geomean * 1.18)

        bars = ax.bar(
            x,
            speedups,
            width=0.58,
            color=panel_colors[precision],
            edgecolor=edge_colors[precision],
            linewidth=0.7,
            zorder=3,
        )
        ax.axhline(1.0, color="#94A3B8", linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        ax.axhline(geomean, color=edge_colors[precision], linewidth=1.0, linestyle=(0, (5, 2)), zorder=2)

        ax.set_ylim(0, ymax)
        ax.set_title(PRECISION_TITLES[precision], pad=5, fontweight="bold", color="#0F172A")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS_TIGHT[m] for m in MODEL_ORDER], rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", pad=2)

        if idx == 0:
            ax.set_ylabel("Speedup over RVV")
        else:
            ax.spines["left"].set_visible(False)

        # Label only the best bar to avoid clutter.
        best_idx = int(np.argmax(speedups))
        for j in [best_idx]:
            bar = bars[j]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.022,
                f"{speedups[j]:.2f}x",
                ha="center",
                va="bottom",
                fontsize=6.7,
                color=edge_colors[precision],
                fontweight="bold",
            )

        ax.text(
            0.97,
            0.96,
            f"GM {geomean:.2f}x",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "#FFFFFF", "edgecolor": "#E2E8F0", "linewidth": 0.7},
        )

    if title:
        fig.suptitle(title, y=0.975, fontsize=10.0, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=6.8, color="#475569")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.83 if subtitle else 0.90])
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)


def plot_compact_3col_hbar(records: Dict[str, AppRecord], output_prefix: Path, title: str, subtitle: str, footer: str):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.1,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.2,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )

    panel_colors = {"binary": "#F26B5B", "INT2": "#4C8BF5", "INT4": "#1FA187"}
    stem_colors = {"binary": "#FBD1CB", "INT2": "#D6E4FF", "INT4": "#CDEFE6"}
    edge_colors = {"binary": "#A3382A", "INT2": "#2855A6", "INT4": "#0E6B57"}

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.25), sharey=True)
    fig.patch.set_facecolor("white")
    y = np.arange(len(MODEL_ORDER))

    for idx, precision in enumerate(PRECISION_ORDER):
        ax = axes[idx]
        apply_axes_style(ax)
        ax.set_facecolor("#FCFDFE")
        ax.grid(axis="x", color="#E5E7EB", linestyle=(0, (2, 3)), linewidth=0.7)
        ax.set_axisbelow(True)

        rvv_cycles = np.array([records[f"rvv_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        bmpmm_cycles = np.array([records[f"bmpmm_{precision}_{model}"].cycles for model in MODEL_ORDER], dtype=float)
        speedups = rvv_cycles / bmpmm_cycles
        geomean = math.exp(np.log(speedups).mean())
        xmax = max(float(speedups.max()) * 1.12, geomean * 1.12)

        for row, val in enumerate(speedups):
            lo = min(1.0, val)
            hi = max(1.0, val)
            ax.hlines(row, lo, hi, color=stem_colors[precision], linewidth=3.0, zorder=1, capstyle="round")

        ax.scatter(
            speedups,
            y,
            s=48,
            color=panel_colors[precision],
            edgecolor=edge_colors[precision],
            linewidth=0.9,
            zorder=3,
        )
        ax.axvline(1.0, color="#94A3B8", linewidth=1.0, linestyle=(0, (2, 2)), zorder=0)
        ax.axvline(geomean, color=edge_colors[precision], linewidth=1.0, linestyle=(0, (5, 2)), zorder=2)

        ax.set_xlim(0.8, xmax)
        ax.set_title(PRECISION_TITLES[precision], pad=5, fontweight="bold", color="#0F172A")
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
            ax.set_ylabel("Model")
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
            ax.spines["left"].set_visible(False)
        ax.invert_yaxis()
        ax.tick_params(axis="y", pad=3)

        best_idx = int(np.argmax(speedups))
        ax.text(
            min(speedups[best_idx] + 0.03 * xmax, xmax * 0.98),
            best_idx,
            f"{speedups[best_idx]:.2f}x",
            ha="left",
            va="center",
            fontsize=6.8,
            color=edge_colors[precision],
            fontweight="bold",
        )
        ax.text(
            0.97,
            0.95,
            f"GM {geomean:.2f}x",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "#FFFFFF", "edgecolor": "#E2E8F0", "linewidth": 0.7},
        )

    axes[1].set_xlabel("Speedup over RVV (RVV cycles / BitFly cycles)")
    if title:
        fig.suptitle(title, y=0.982, fontsize=10.2, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="center", fontsize=6.9, color="#475569")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84 if subtitle else 0.92])
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)


def plot_conference_3col(records: Dict[str, AppRecord], output_prefix: Path, title: str, subtitle: str, footer: str):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )

    panel_cfg = {
        "binary": {"face": "#F28E2B", "edge": "#B95E0A", "accent": "#B95E0A"},
        "INT2": {"face": "#4E79A7", "edge": "#2F5B88", "accent": "#2F5B88"},
        "INT4": {"face": "#59AFA0", "edge": "#2B7A6B", "accent": "#2B7A6B"},
    }
    plot_models = MODEL_ORDER
    labels = [MODEL_LABELS_LINE[m] for m in plot_models]
    n_rows = len(plot_models)
    y = np.arange(n_rows)

    fig = plt.figure(figsize=(6.9, 2.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.15, 1.1, 1.1, 1.1], wspace=0.08)
    ax_labels = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
    fig.patch.set_facecolor("white")

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(-0.5, n_rows - 0.5)
    ax_labels.invert_yaxis()
    ax_labels.axis("off")
    for row, label in enumerate(labels):
        ax_labels.text(0.98, row, label, ha="right", va="center", fontsize=6.6, color="#111827")

    for idx, precision in enumerate(PRECISION_ORDER):
        ax = axes[idx]
        values = np.array(
            [
                records[f"rvv_{precision}_{model}"].cycles / records[f"bmpmm_{precision}_{model}"].cycles
                for model in plot_models
            ],
            dtype=float,
        )
        geomean = math.exp(np.log(values).mean())
        cfg = panel_cfg[precision]
        xmin = 0.0
        xmax = max(float(values.max()) * 1.08, geomean * 1.12)

        ax.barh(
            y,
            values,
            height=0.72,
            color=cfg["face"],
            edgecolor=cfg["edge"],
            linewidth=0.6,
            zorder=3,
        )
        ax.axvline(geomean, color=cfg["accent"], linewidth=1.0, linestyle=(0, (4, 2)), zorder=4)
        ax.grid(axis="x", color="#E5E7EB", linestyle=(0, (2, 3)), linewidth=0.65, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_title(f"{PRECISION_TITLES[precision]}\nGM {geomean:.2f}x", pad=4, fontweight="bold", color="#0F172A")

        if precision == "binary":
            ax.set_xticks([0, 5, 10])
        else:
            ax.set_xticks([0, 1, 2, 3])

        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")

        for row, val in enumerate(values):
            x_text = max(val - xmax * 0.05, xmax * 0.03)
            ha = "right" if val > xmax * 0.18 else "left"
            txt_color = "white" if val > xmax * 0.22 else "#0F172A"
            if ha == "left":
                x_text = val + xmax * 0.015
            ax.text(
                x_text,
                row,
                f"{val:.2f}x",
                ha=ha,
                va="center",
                fontsize=6.6,
                color=txt_color,
                fontweight="bold",
                zorder=5,
            )

    top = 0.88 if title or subtitle else 0.94
    bottom = 0.08 if footer else 0.06
    if title:
        fig.suptitle(title, y=0.985, fontsize=9.2, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.947, subtitle, ha="center", va="center", fontsize=6.6, color="#4B5563")
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=6.0, color="#6B7280")

    fig.subplots_adjust(left=0.03, right=0.995, bottom=bottom, top=top, wspace=0.06)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)


def main():
    args = parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    records = merge_runs(run_dirs)
    validate_records(records)
    write_csv(records, output_prefix)
    footer = args.footer
    if footer is None:
        footer = "Source run: " + " + ".join(run_dir.name for run_dir in run_dirs)
    if args.layout == "single-column":
        plot_single_column(records, output_prefix, args.title, args.subtitle, footer)
    elif args.layout == "single-column-camera":
        plot_single_column_camera(records, output_prefix, args.title, args.subtitle, footer)
    elif args.layout == "compact-3col":
        plot_compact_3col(records, output_prefix, args.title, args.subtitle, footer)
    elif args.layout == "compact-3col-hbar":
        plot_compact_3col_hbar(records, output_prefix, args.title, args.subtitle, footer)
    elif args.layout == "conference-3col":
        plot_conference_3col(records, output_prefix, args.title, args.subtitle, footer)
    else:
        plot(records, output_prefix, args.title, args.subtitle, footer)


if __name__ == "__main__":
    main()
