import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ======================
# ✅ 原始数据：cycles（你提供的单位）
# ======================
rvv_cycles = {
    "matmul":  [7664595, 16618081, 25183149],
    "softmax": [345169, 1317386,  1974464],
    "rmsnorm": [316533, 564278,  868666],
    "else":    [856152, 1504538,  2275116],
}

mixed_cycles = {
    "matmul":  [1006651, 873799, 1000445],
    "softmax": [334734,  1318148,  1976651],
    "rmsnorm": [316372,  564693,  866399],
    "else":    [875838, 1520263, 2304048],
}

# ======================
# ✅ 关键路径延迟（来自时序分析）
# ======================
# critical_path_rvv_ns = 36.01    # ns
# critical_path_mixed_ns = 36.05  # ns

# 转换为频率（Hz）
# freq_rvv = 1 / (critical_path_rvv_ns * 1e-9)     # Hz
# freq_mixed = 1 / (critical_path_mixed_ns * 1e-9) # Hz
freq_rvv = 90.02 * 1e6
freq_mixed = 89.69 * 1e6

print(f"Calculated frequency for RVV: {freq_rvv / 1e6:.2f} MHz")
print(f"Calculated frequency for Mixed: {freq_mixed / 1e6:.2f} MHz")

# ======================
# ✅ 转换 cycles -> 时间 (ms)
# ======================
def cycles_to_ms(cycles_list, freq_hz):
    return [c / freq_hz * 1000 for c in cycles_list]  # ms

models = ["15M", "42M", "110M"]
components = ["matmul", "softmax", "rmsnorm", "else"]

# 转换为 ms
rvv_data = {c: cycles_to_ms(rvv_cycles[c], freq_rvv) for c in components}
mixed_data = {c: cycles_to_ms(mixed_cycles[c], freq_mixed) for c in components}

# 计算总时间（ms）—— 用于柱顶标注
total_rvv_ms = [sum(rvv_data[c][i] for c in components) for i in range(len(models))]
total_mixed_ms = [sum(mixed_data[c][i] for c in components) for i in range(len(models))]

# ======================
# ✅ 自动计算各组件占比（用于堆叠柱状图内标注）
# ======================
ratios = {"rvv": [], "mixed": []}
for i in range(len(models)):
    total_rvv = total_rvv_ms[i]
    total_mixed = total_mixed_ms[i]
    ratios["rvv"].append({
        c: rvv_data[c][i] / total_rvv for c in components
    })
    ratios["mixed"].append({
        c: mixed_data[c][i] / total_mixed for c in components
    })

# ======================
# 配色
# ======================
colors = {
    "matmul": "#4E79A7",
    "softmax": "#F28E2B",
    "rmsnorm": "#59A14F",
    "else": "#B07AA1",
}

# ======================
# 字体设置
# ======================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "patch.linewidth": 0.8,
    }
)

# ======================
# 布局参数
# ======================
gap = 1.0
width = 0.35
x = np.array([0, gap, 2 * gap])

# ======================
# 创建图像
# ======================
fig, ax = plt.subplots(figsize=(12, 9))
occupied_positions = {}  # 用于标注防重叠

# ======================
# 绘图 + 标注（占比 %）
# ======================
for mode_idx, (mode, data) in enumerate([("rvv", rvv_data), ("mixed", mixed_data)]):
    bottom = np.zeros(len(models))
    ratio_list = ratios[mode]
    for comp in components:
        values = data[comp]
        bars = ax.bar(
            x + (width / 2 if mode == "mixed" else -width / 2),
            values,
            width,
            bottom=bottom,
            color=colors[comp],
            alpha=0.95 if mode == "rvv" else 0.85,
            edgecolor="white",
            linewidth=1.0,
            hatch="///" if mode == "mixed" else "",
            label=f"{comp.upper()}" if mode == "rvv" else "",
        )
        # ========== 柱内标注：占比 % ==========
        for i, (val, r) in enumerate(zip(values, ratio_list)):
            total_height = total_rvv_ms[i] if mode == "rvv" else total_mixed_ms[i]
            center_x = x[i] + (width / 2 if mode == "mixed" else -width / 2)
            y_pos = bottom[i] + val / 2
            label = f"{r[comp]*100:.0f}%"

            height_ok = val > 10
            width_ok = width > 0.3

            if height_ok and width_ok:
                ax.text(
                    center_x,
                    y_pos,
                    label,
                    ha="center",
                    va="center",
                    fontsize=17,
                    color="white",
                    fontweight="bold",
                    bbox=None,
                )
            else:
                key = (center_x, mode)
                if key not in occupied_positions:
                    occupied_positions[key] = []
                base_offset = 0.1 * total_height
                candidate_y = y_pos + base_offset
                attempt = 0
                while any(abs(candidate_y - y0) < 14 for y0 in occupied_positions[key]):
                    attempt += 1
                    candidate_y = y_pos + base_offset + attempt * 0.4 * total_height
                    if attempt > 100:
                        candidate_y += 0.1 * total_height
                occupied_positions[key].append(candidate_y)

                ax.annotate(
                    label,
                    xy=(center_x, y_pos),
                    xytext=(center_x + 0.28, candidate_y),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=16,
                    color="black",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, shrinkA=3, shrinkB=3,),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9),
                )
        bottom += np.array(values)

# ======================
# 柱顶标注总耗时（ms）
# ======================
for i in range(len(models)):
    ax.text(
        x[i] - width / 2,
        total_rvv_ms[i] + 5,
        f"{total_rvv_ms[i]:.1f}ms",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="black",
    )
    ax.text(
        x[i] + width / 2,
        total_mixed_ms[i] + 5,
        f"{total_mixed_ms[i]:.1f}ms",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="black",
    )
# ======================
# ✅ 标注：Mixed 相比 RVV 的加速倍数
# ======================
for i in range(len(models)):
    speedup = total_rvv_ms[i] / total_mixed_ms[i]  # 加速倍数
    center_x = x[i] + width / 2  # Mixed 柱子中心
    top = total_mixed_ms[i]  # Mixed 柱子顶部
    ax.text(
        center_x,
        top + 26,  # 稍高于总耗时标签
        f"×{speedup:.1f}",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="red",  # 红色突出
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", linewidth=1.0, alpha=0.9),
    )
# ======================
# 坐标轴设置
# ======================
ax.set_xlabel("Model Size", labelpad=8)
ax.set_ylabel("Inference Latency (ms)", labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(models)

def format_ms(x, pos):
    return f"{x/1000:.1f}s" if x >= 1000 else f"{x:.0f}ms"
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ms))

ax.grid(axis="y", linestyle="--", alpha=0.6, linewidth=0.6)
ax.set_axisbelow(True)

# ======================
# 图例
# ======================
comp_legend_elements = [Patch(facecolor=colors[c], label=c.upper(), edgecolor="none") for c in components]
mode_legend_elements = [
    Patch(facecolor="#d9d9d9", edgecolor="black", label="ARA", linewidth=1.0),
    Patch(facecolor="#d9d9d9", edgecolor="black", hatch="///", label="BitFly", linewidth=1.0),
]
legend_elements = comp_legend_elements + mode_legend_elements

ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=6,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.4,
    borderpad=0.2,
    fontsize=17,
)

# ======================
# 布局调整
# ======================
plt.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.15)

# ======================
# 保存图像
# ======================
plt.savefig("latency_breakdown.pdf", format="pdf")
plt.savefig("latency_breakdown.png", format="png", dpi=600)
# plt.show()