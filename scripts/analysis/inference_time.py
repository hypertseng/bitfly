import matplotlib.pyplot as plt
import numpy as np
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


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def label_color(fill_hex):
    r, g, b = hex_to_rgb(fill_hex)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#1f1f1f" if luminance > 0.62 else "white"


models = ["15M", "42M", "110M"]
components = ["matmul", "softmax", "rmsnorm", "else"]
mode_specs = [
    ("rvv", "ARA", rvv_cycles, -1),
    ("mixed", "BitFly", mixed_cycles, 1),
]

latency_data = {
    mode: {comp: cycles_to_ms(cycle_map[comp], freq_rvv if mode == "rvv" else freq_mixed) for comp in components}
    for mode, _, cycle_map, _ in mode_specs
}

totals = {
    mode: [sum(latency_data[mode][comp][i] for comp in components) for i in range(len(models))]
    for mode, _, _, _ in mode_specs
}

ratios = {
    mode: [
        {comp: latency_data[mode][comp][i] / totals[mode][i] for comp in components}
        for i in range(len(models))
    ]
    for mode, _, _, _ in mode_specs
}

colors = {
    "matmul": "#2F5C8A",
    "softmax": "#D97B2D",
    "rmsnorm": "#3F8F6B",
    "else": "#8A6FB3",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 17,
        "axes.titlesize": 21,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.linewidth": 0.9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "hatch.linewidth": 1.0,
    }
)

x = np.arange(len(models)) * 1.35
width = 0.34

fig, ax = plt.subplots(figsize=(20, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FBFBFD")

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#AAB4C3")

ax.grid(axis="y", color="#D7DEE8", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.9)
ax.set_axisbelow(True)

max_total = max(max(v) for v in totals.values())

for mode, mode_label, _, direction in mode_specs:
    x_pos = x + direction * width / 2
    bottom = np.zeros(len(models))

    for comp in components:
        values = np.array(latency_data[mode][comp])
        bars = ax.bar(
            x_pos,
            values,
            width,
            bottom=bottom,
            color=colors[comp],
            alpha=0.97 if mode == "rvv" else 0.88,
            edgecolor="white",
            linewidth=1.1,
            hatch="////" if mode == "mixed" else None,
            zorder=3,
        )

        for i, bar in enumerate(bars):
            ratio = ratios[mode][i][comp]
            height = values[i]
            center_y = bottom[i] + height / 2

            if ratio >= 0.08 and height >= max_total * 0.035:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    center_y,
                    f"{ratio * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=13.5,
                    fontweight="bold",
                    color=label_color(colors[comp]),
                    zorder=5,
                )
            elif ratio >= 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[i] + height + max_total * 0.012,
                    f"{ratio * 100:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11.5,
                    color="#4E5968",
                    zorder=5,
                )

        bottom += values

    for i, total in enumerate(totals[mode]):
        ax.text(
            x_pos[i],
            total + max_total * 0.018,
            f"{total:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=14.5,
            fontweight="bold",
            color="#202633",
        )
        ax.text(
            x_pos[i],
            -max_total * 0.055,
            mode_label,
            ha="center",
            va="top",
            fontsize=13,
            color="#4E5968",
            fontweight="bold",
        )

for i, model in enumerate(models):
    speedup = totals["rvv"][i] / totals["mixed"][i]
    top_y = max(totals["rvv"][i], totals["mixed"][i]) + max_total * 0.09
    ax.plot(
        [x[i] - width / 2, x[i] + width / 2],
        [top_y, top_y],
        color="#C33D3D",
        linewidth=1.6,
        solid_capstyle="round",
        zorder=4,
    )
    ax.text(
        x[i],
        top_y + max_total * 0.012,
        f"{speedup:.1f}x faster",
        ha="center",
        va="bottom",
        fontsize=13.5,
        fontweight="bold",
        color="#C33D3D",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFF5F5",
            edgecolor="#E7B3B3",
            linewidth=0.8,
        ),
        zorder=5,
    )

ax.set_xlabel("Model Size", labelpad=12)
ax.set_ylabel("Inference Latency", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(format_latency))

ax.set_ylim(-max_total * 0.09, max_total * 1.17)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", colors="#4E5968")

component_legend = [
    Patch(facecolor=colors[comp], edgecolor="none", label=comp.upper()) for comp in components
]
mode_legend = [
    Patch(facecolor="#C9D1DB", edgecolor="#7D8896", label="ARA", linewidth=0.9),
    Patch(facecolor="#C9D1DB", edgecolor="#7D8896", hatch="////", label="BitFly", linewidth=0.9),
]

legend_top = ax.legend(
    handles=component_legend,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.10),
    ncol=4,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.0,
)
ax.add_artist(legend_top)

ax.legend(
    handles=mode_legend,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.10),
    ncol=2,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.2,
)

plt.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.18)

plt.savefig("latency_breakdown.pdf", format="pdf")
plt.savefig("latency_breakdown.png", format="png", dpi=600)
