import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ======================
# Raw cycles
# ======================
rvv_cycles = {
    "matmul": [7664595, 16618081, 25183149],
    "softmax": [345169, 1317386, 1974464],
    "rmsnorm": [316533, 564278, 868666],
    "else": [856152, 1504538, 2275116],
}

mixed_cycles = {
    "matmul": [1006651, 873799, 1000445],
    "softmax": [334734, 1318148, 1976651],
    "rmsnorm": [316372, 564693, 866399],
    "else": [875838, 1520263, 2304048],
}

# ======================
# Timing
# ======================
freq_rvv = 90.02 * 1e6
freq_mixed = 89.69 * 1e6

print(f"Calculated frequency for RVV: {freq_rvv / 1e6:.2f} MHz")
print(f"Calculated frequency for Mixed: {freq_mixed / 1e6:.2f} MHz")


def cycles_to_ms(cycles_list, freq_hz):
    return [c / freq_hz * 1000 for c in cycles_list]


def format_latency(value, _):
    return f"{value / 1000:.2f}s" if value >= 1000 else f"{value:.0f}ms"


models = ["15M", "42M", "110M"]
components = ["matmul", "softmax", "rmsnorm", "else"]
component_labels = {
    "matmul": "MatMul",
    "softmax": "Softmax",
    "rmsnorm": "RMSNorm",
    "else": "Other",
}
mode_specs = [("rvv", "ARA", rvv_cycles), ("mixed", "BitFly", mixed_cycles)]

latency_data = {
    mode: {comp: cycles_to_ms(cycle_map[comp], freq_rvv if mode == "rvv" else freq_mixed) for comp in components}
    for mode, _, cycle_map in mode_specs
}

totals = {
    mode: [sum(latency_data[mode][comp][i] for comp in components) for i in range(len(models))]
    for mode, _, _ in mode_specs
}

shares = {
    mode: [
        {comp: latency_data[mode][comp][i] / totals[mode][i] * 100 for comp in components}
        for i in range(len(models))
    ]
    for mode, _, _ in mode_specs
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 12,
        "axes.titlesize": 14.5,
        "axes.labelsize": 14.2,
        "xtick.labelsize": 12.2,
        "ytick.labelsize": 12.2,
        "legend.fontsize": 11.5,
        "axes.linewidth": 0.85,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

max_total = max(max(v) for v in totals.values())
mode_colors = {
    "rvv": {"face": "#DCE3EC", "edge": "#95A3B8", "text": "#334155"},
    "mixed": {"face": "#1F4E79", "edge": "#163A5A", "text": "white"},
}

display_order = [2, 1, 0]
group_centers = np.arange(len(display_order)) * 1.78
pair_offset = 0.24
bar_height = 0.34
row_specs = []
row_positions = {}

for center, model_idx in zip(group_centers, display_order):
    for mode, label, _ in mode_specs:
        y = center - pair_offset if mode == "rvv" else center + pair_offset
        row_specs.append((model_idx, mode, label))
        row_positions[(model_idx, mode)] = y

fig = plt.figure(figsize=(12.8, 4.55))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.025)
ax_bar = fig.add_subplot(gs[0, 0])
ax_share = fig.add_subplot(gs[0, 1], sharey=ax_bar)

for ax in (ax_bar, ax_share):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D2DAE4")
    ax.spines["bottom"].set_color("#D2DAE4")

for idx, center in enumerate(group_centers):
    if idx % 2 == 0:
        for ax in (ax_bar, ax_share):
            ax.axhspan(center - 0.58, center + 0.58, color="#F8FAFC", zorder=0)

for divider in (group_centers[:-1] + group_centers[1:]) / 2:
    ax_bar.axhline(divider, color="#E6EBF2", linewidth=0.9, zorder=1)
    ax_share.axhline(divider, color="#E6EBF2", linewidth=0.9, zorder=1)

ax_bar.grid(axis="x", color="#E6EDF5", linestyle=(0, (2, 4)), linewidth=0.8)
ax_bar.set_axisbelow(True)

for model_idx, mode, label in row_specs:
    y = row_positions[(model_idx, mode)]
    total = totals[mode][model_idx]
    style = mode_colors[mode]

    ax_bar.barh(
        y,
        total,
        height=bar_height,
        color=style["face"],
        edgecolor=style["edge"],
        linewidth=1.0,
        zorder=3,
    )
    ax_bar.text(
        total + max_total * 0.02,
        y,
        f"{total:.1f} ms",
        ha="left",
        va="center",
        fontsize=10.9,
        fontweight="bold",
        color="#1F2937",
        zorder=5,
    )

speedup_x = max_total * 1.19
for center, model_idx in zip(group_centers, display_order):
    speedup = totals["rvv"][model_idx] / totals["mixed"][model_idx]
    ax_bar.text(
        speedup_x,
        center,
        f"{speedup:.1f}x speedup",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#C84C3A",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="#FFF7F4",
            edgecolor="#F1C5BD",
            linewidth=0.9,
        ),
        zorder=6,
    )

yticks = [row_positions[(model_idx, mode)] for model_idx, mode, _ in row_specs]
ylabels = [label for _, _, label in row_specs]
ax_bar.set_yticks(yticks)
ax_bar.set_yticklabels(ylabels, fontweight="bold", color="#475467")
ax_bar.tick_params(axis="y", length=0, pad=8)
ax_bar.tick_params(axis="x", length=0, colors="#5B6878")
ax_bar.set_xlim(0, max_total * 1.34)
ax_bar.xaxis.set_major_formatter(FuncFormatter(format_latency))
ax_bar.set_xlabel("End-to-End Latency", labelpad=8)
ax_bar.set_title("Total Latency", loc="left", pad=12, fontweight="bold", color="#111827")

for center, model_idx in zip(group_centers, display_order):
    ax_bar.text(
        -0.18,
        center,
        models[model_idx],
        transform=ax_bar.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color="#111827",
    )

ax_bar.text(
    -0.18,
    1.025,
    "Model",
    transform=ax_bar.transAxes,
    ha="right",
    va="bottom",
    fontsize=10.8,
    fontweight="bold",
    color="#667085",
)
ax_bar.text(
    -0.015,
    1.025,
    "System",
    transform=ax_bar.transAxes,
    ha="right",
    va="bottom",
    fontsize=10.8,
    fontweight="bold",
    color="#667085",
)

share_cmap = LinearSegmentedColormap.from_list(
    "share_map",
    ["#F5F9FD", "#D6E6F3", "#95BAD8", "#4A79A8", "#1F4E79"],
)
share_norm = Normalize(vmin=0, vmax=85)

for x_idx, comp in enumerate(components):
    for model_idx, mode, _ in row_specs:
        y = row_positions[(model_idx, mode)]
        share = shares[mode][model_idx][comp]
        cell = plt.Rectangle(
            (x_idx - 0.5, y - bar_height / 2),
            1.0,
            bar_height,
            facecolor=share_cmap(share_norm(share)),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax_share.add_patch(cell)
        ax_share.text(
            x_idx,
            y,
            f"{share:.0f}%",
            ha="center",
            va="center",
            fontsize=10.3,
            fontweight="bold",
            color="white" if share >= 45 else "#233143",
            zorder=4,
        )

for boundary in np.arange(len(components) + 1) - 0.5:
    ax_share.axvline(boundary, color="#E6EBF2", linewidth=0.9, zorder=1)

ax_share.set_xlim(-0.5, len(components) - 0.5)
ax_share.set_xticks(np.arange(len(components)))
ax_share.set_xticklabels([component_labels[comp] for comp in components], fontweight="bold")
ax_share.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=6)
ax_share.tick_params(axis="y", left=False, labelleft=False)
ax_share.spines["left"].set_visible(False)
ax_share.set_title("Operator Share (%)", loc="left", pad=12, fontweight="bold", color="#111827")

lower_limit = group_centers[-1] + 0.62
upper_limit = -0.62
ax_bar.set_ylim(lower_limit, upper_limit)
ax_share.set_ylim(lower_limit, upper_limit)

plt.subplots_adjust(left=0.16, right=0.985, top=0.84, bottom=0.16)

plt.savefig("latency_breakdown.pdf", format="pdf")
plt.savefig("latency_breakdown.png", format="png", dpi=600)
