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

PRECISION_ORDER = ["binary", "INT2", "INT4"]
PRECISION_TITLES = {"binary": "Binary", "INT2": "INT2", "INT4": "INT4"}

RUN_COLORS = {
    "rvv": {"face": "#D7DEE8", "edge": "#64748B"},
    "bmpmm": {"face": "#F97316", "edge": "#9A3412"},
}

SPEEDUP_COLORS = {"pos": "#EA580C", "neg": "#B91C1C"}

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


def plot(records: Dict[str, AppRecord], output_prefix: Path):
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
    fig.suptitle("BitFly vs. RVV on Linear-Layer Workloads Extracted from 10 LLMs", y=0.996, fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.948,
        "Each point aggregates the seven linear layers from transformer block 0 at the target precision; later reruns override failed earlier logs.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#475569",
    )
    fig.text(
        0.5,
        0.012,
        "Source runs: 20260401_fast_default_p60_60apps_allparallel + 20260401_fast_default_p7_7bmpmm_binary_rerun",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#64748B",
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def main():
    args = parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    records = merge_runs(run_dirs)
    validate_records(records)
    write_csv(records, output_prefix)
    plot(records, output_prefix)


if __name__ == "__main__":
    main()
