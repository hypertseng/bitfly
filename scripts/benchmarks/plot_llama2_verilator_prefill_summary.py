#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


MODEL_ORDER = ["15M", "42M", "110M", "1B", "3B"]
PREC_ORDER = ["W1A8", "W2A8", "W4A8"]
SEQ_ORDER = [32, 64, 128, 256]
SEQ_COLORS = {
    32: "#4C78A8",
    64: "#72B7B2",
    128: "#F2A541",
    256: "#E45756",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot compact summary figure for llama2 Verilator prefill results."
    )
    parser.add_argument(
        "--input",
        default="tmp/llama2_verilator_prefill_all_models_20260401/summary.csv",
        help="Merged summary CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="tmp/llama2_verilator_prefill_all_models_20260401/figures",
        help="Directory for output figures.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speedup = row["speedup"].strip()
            if speedup.endswith("x"):
                row["speedup_value"] = float(speedup[:-1])
            else:
                bmpmm = float(row["bmpmm_cycles"])
                rvv = float(row["rvv_cycles"])
                row["speedup_value"] = rvv / bmpmm
            row["seq_len"] = int(row["seq_len"])
            rows.append(row)
    return rows


def build_lookup(rows):
    data = {}
    for row in rows:
        data[(row["precision"], row["model"], row["seq_len"])] = row["speedup_value"]
    return data


def speedup_formatter(value, _pos):
    return f"{value:.0f}x"


def main():
    args = parse_args()
    csv_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    data = build_lookup(rows)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.0), sharey=True)
    model_x = np.arange(len(MODEL_ORDER), dtype=float)
    bar_width = 0.16
    offsets = {
        32: -1.5 * bar_width,
        64: -0.5 * bar_width,
        128: 0.5 * bar_width,
        256: 1.5 * bar_width,
    }

    all_speedups = [row["speedup_value"] for row in rows]
    y_max = max(all_speedups)
    y_top = min(32.0, max(6.0, y_max * 1.10))
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SEQ_COLORS[seq_len],
            linewidth=2.4,
            marker="o",
            markersize=5.2,
            markerfacecolor=SEQ_COLORS[seq_len],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"L={seq_len}",
        )
        for seq_len in SEQ_ORDER
    ]

    for ax, prec in zip(axes, PREC_ORDER):
        for seq_len in SEQ_ORDER:
            xs = model_x + offsets[seq_len]
            ys = [data[(prec, model, seq_len)] for model in MODEL_ORDER]

            ax.bar(
                xs,
                ys,
                width=bar_width * 0.84,
                color=SEQ_COLORS[seq_len],
                edgecolor="none",
                linewidth=0.0,
                alpha=0.20,
                zorder=2,
            )
            ax.plot(
                xs,
                ys,
                color=SEQ_COLORS[seq_len],
                linewidth=2.35,
                marker="o",
                markersize=5.1,
                markerfacecolor=SEQ_COLORS[seq_len],
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )

        ax.set_title(prec, pad=8)
        ax.set_xticks(model_x, MODEL_ORDER)
        ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
        ax.set_ylim(0, y_top)
        ax.grid(True, axis="y", color="#D7D7DB", linewidth=0.80, alpha=0.95)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(speedup_formatter))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.spines["left"].set_color("#2F2F2F")
        ax.spines["bottom"].set_color("#2F2F2F")
        ax.tick_params(width=1.0, length=5, color="#2F2F2F")
        ax.set_xlabel("Model Size")

    axes[0].set_ylabel("Speedup over RVV INT8")

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.24, top=0.83, wspace=0.10)

    png_path = outdir / "llama2_verilator_prefill_speedup_summary.png"
    pdf_path = outdir / "llama2_verilator_prefill_speedup_summary.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
