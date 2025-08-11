import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- 全局字体设置 ---
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
    }
)
# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 平台参数
peak_perf_high = 4096  # ops/cycle
peak_perf_low = 64  # ops/cycle
bw_vrf_high = 64 * 4  # bytes/cycle
bw_vrf_low = 16 * 4  # bytes/cycle
# bw_dram = 16  # bytes/cycle  # 已移除，不再需要

# 时钟频率和单位转换
clock_freq_hz = 101 * 1e6  # 101 MHz
ops_per_cycle_to_gop_per_sec = clock_freq_hz / 1e9  # 转换因子

# X轴: 运算强度
oi = np.logspace(-1, 4, 1000)  # OPs/Byte

# ==================== 实验数据图 ====================
fig1, ax1 = plt.subplots(figsize=(12, 7))

# 绘制所有基础限制线 (转换为 GOPs/s)，移除了 y_dram
y_compute_low = np.full_like(oi, peak_perf_low * ops_per_cycle_to_gop_per_sec)
y_compute_high = np.full_like(oi, peak_perf_high * ops_per_cycle_to_gop_per_sec)
y_vrf_low = oi * bw_vrf_low * ops_per_cycle_to_gop_per_sec
y_vrf_high = oi * bw_vrf_high * ops_per_cycle_to_gop_per_sec

# 优化线条样式
line_styles = {
    "VRF Low": {"color": "#FF7F0E", "ls": "--", "lw": 3},
    "VRF High": {"color": "#2CA02C", "ls": "-.", "lw": 3},
    "Compute Low": {"color": "#1F77B4", "ls": ":", "lw": 3},
    "Compute High": {"color": "#9467BD", "ls": ":", "lw": 3},
}

# 绘制线条，移除了 ax1.loglog(oi, y_dram, ...)
ax1.loglog(oi, y_vrf_low, label="VRF Bound (6.46 GB/s)", **line_styles["VRF Low"])
ax1.loglog(oi, y_vrf_high, label="VRF Bound (25.86 GB/s)", **line_styles["VRF High"])
ax1.loglog(
    oi, y_compute_low, label="Compute Bound (6.46 GOPs/s)", **line_styles["Compute Low"]
)
ax1.loglog(
    oi,
    y_compute_high,
    label="Compute Bound (413.70 GOPs/s)",
    **line_styles["Compute High"],
)

# 实验数据点
sizes = [32, 64, 128, 256, 512, 1024, 2048]
mixed_perf = [61.88, 139.88, 200.62, 440.19, 1286.31, 1907.06, 2049.46]
vector_perf = [9.45, 19.08, 22.65, 24.23, 25.54, 51.11, 51.13]
mixed_perf_gop = [perf * ops_per_cycle_to_gop_per_sec for perf in mixed_perf]
vector_perf_gop = [perf * ops_per_cycle_to_gop_per_sec for perf in vector_perf]

# 计算OI值
oi_mixed = []
oi_vector = []
for size in sizes:
    M = N = K = size
    ops_mixed = M * N * K
    bytes_mixed = M * K + (K * N) // 8 + M * N * 2
    oi_mixed.append(ops_mixed / bytes_mixed)
    ops_vector = 2 * M * N * K
    bytes_vector = M * K + K * N + M * N * 2
    oi_vector.append(ops_vector / bytes_vector)

# 绘制数据点
ax1.scatter(
    oi_mixed,
    mixed_perf_gop,
    color="#FF6B6B",
    s=300,
    marker="o",
    edgecolors="white",
    linewidth=1.5,
    zorder=20,
    label="Binary Mixed Precision GEMM",
)
ax1.scatter(
    oi_vector,
    vector_perf_gop,
    color="#4ECDC4",
    s=300,
    marker="s",
    edgecolors="white",
    linewidth=1.5,
    zorder=20,
    label="RVV INT8 GEMM Baseline",
)

# 绘制趋势线
ax1.plot(oi_mixed, mixed_perf_gop, "--", color="#FF6B6B", linewidth=2, alpha=0.8)
ax1.plot(oi_vector, vector_perf_gop, "--", color="#4ECDC4", linewidth=2, alpha=0.8)

# 关键尺寸标注
key_sizes = [64, 256, 1024]
for i, (oi_m, perf_m_gop, size) in enumerate(zip(oi_mixed, mixed_perf_gop, sizes)):
    if size in key_sizes:
        ax1.annotate(
            f"{size}³",
            xy=(oi_m, perf_m_gop),
            xytext=(oi_m * 0.8, perf_m_gop * 2.0),
            arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=1.2),
            fontsize=16,
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.8,
                edgecolor="#FF6B6B",
            ),
        )
for i, (oi_v, perf_v_gop, size) in enumerate(zip(oi_vector, vector_perf_gop, sizes)):
    if size in key_sizes:
        ax1.annotate(
            f"{size}³",
            xy=(oi_v, perf_v_gop),
            xytext=(oi_v * 0.9, perf_v_gop * 0.4),
            arrowprops=dict(arrowstyle="->", color="#4ECDC4", lw=1.2),
            fontsize=16,
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.8,
                edgecolor="#4ECDC4",
            ),
        )

# 坐标轴设置
ax1.set_xlim(0.5, 2000)
ax1.set_ylim(0.5, 1100)
ax1.set_xlabel("Operational Intensity (OPs/Byte)", labelpad=10)
ax1.set_ylabel("Performance (GOPs/s)", labelpad=10)
# ax1.set_title(
#     "Roofline Analysis - Experimental Results\n(Binary Mixed Precision GEMM vs RVV INT8 GEMM)",
#     pad=20,
# )

# 优化刻度显示
ax1.set_xticks([0.5, 10, 100, 1000])
ax1.set_xticklabels(["0.5", "10", "100", "1K"])
ax1.set_yticks([0.5, 10, 100, 200, 1000])
ax1.set_yticklabels(["0.5", "10", "100", "200", "1000"])

# 网格
ax1.grid(True, which="both", linestyle=":", alpha=0.6)

# 图例优化
legend1 = ax1.legend(
    loc="best", frameon=True, fancybox=True, shadow=True, ncol=1, borderpad=0.5
)
legend1.get_frame().set_facecolor("white")
legend1.get_frame().set_alpha(0.8)

plt.tight_layout()
plt.savefig("roofline_experimental.png", dpi=300, bbox_inches="tight")
print("Experimental Roofline plot saved as roofline_experimental.png")
# ==================== 理论性能图 ====================
fig2, ax2 = plt.subplots(figsize=(12, 7))

# 绘制基础限制线，移除了 y_dram
ax2.loglog(oi, y_vrf_low, label="VRF Bound (6.46 GB/s)", **line_styles["VRF Low"])
ax2.loglog(oi, y_vrf_high, label="VRF Bound (25.86 GB/s)", **line_styles["VRF High"])
ax2.loglog(
    oi, y_compute_low, label="Compute Bound (6.46 GOPs/s)", **line_styles["Compute Low"]
)
ax2.loglog(
    oi,
    y_compute_high,
    label="Compute Bound (413.70 GOPs/s)",
    **line_styles["Compute High"],
)

# 计算三种配置的Roofline线
y_config1 = np.minimum(y_vrf_low, y_compute_low)
y_config2 = np.minimum(y_vrf_low, y_compute_high)
y_config3 = np.minimum(y_vrf_high, y_compute_high)

# 绘制配置线 (关键：移除 label)
config_styles = {
    "Config 1": {"color": "#1F77B4", "ls": "-", "lw": 4, "alpha": 0.9},
    "Config 2": {"color": "#D62728", "ls": "-", "lw": 4, "alpha": 0.9},
    "Config 3": {"color": "#2CA02C", "ls": "-", "lw": 4, "alpha": 0.9},
}
# 移除 label 参数，因此它们不会出现在图例中
ax2.loglog(oi, y_config1, **config_styles["Config 1"])
ax2.loglog(oi, y_config2, **config_styles["Config 2"])
ax2.loglog(oi, y_config3, **config_styles["Config 3"])

# 计算理论性能点
sizes_theory = [32, 64, 128, 256, 512, 1024, 2048]
oi_config1, perf_config1_gop = [], []
oi_config2, perf_config2_gop = [], []
oi_config3, perf_config3_gop = [], []
for size in sizes_theory:
    M = N = K = size
    # Config 1: int8×int8
    ops = 2 * M * N * K
    bytes_data = M * K + K * N + M * N * 2
    oi_val = ops / bytes_data
    perf_val = min(oi_val * bw_vrf_low, peak_perf_low)
    oi_config1.append(oi_val)
    perf_config1_gop.append(perf_val * ops_per_cycle_to_gop_per_sec)
    # Config 2: int8×int1
    ops = M * N * K
    bytes_data = M * K + (K * N) // 8 + M * N * 2
    oi_val = ops / bytes_data
    perf_val = min(oi_val * bw_vrf_low, peak_perf_high)
    oi_config2.append(oi_val)
    perf_config2_gop.append(perf_val * ops_per_cycle_to_gop_per_sec)
    # Config 3: int8×int1
    ops = M * N * K
    bytes_data = M * K + (K * N) // 8 + M * N * 2
    oi_val = ops / bytes_data
    perf_val = min(oi_val * bw_vrf_high, peak_perf_high)
    oi_config3.append(oi_val)
    perf_config3_gop.append(perf_val * ops_per_cycle_to_gop_per_sec)

# 绘制理论点 (关键：在 scatter 上保留 label)
marker_styles = {
    "Config 1": {"color": "#1F77B4", "marker": "o", "s": 300},
    "Config 2": {"color": "#D62728", "marker": "s", "s": 300},
    "Config 3": {"color": "#2CA02C", "marker": "^", "s": 300},
}

# 关键步骤：使用 plot 函数绘制带标记的线
# 这样图例会显示为一条线加一个点
ax2.plot(
    oi_config1,
    perf_config1_gop,
    label="Config 1: INT8 GEMM (Low VRF BW)",
    color="#1F77B4",
    linewidth=2,
    alpha=0.8,
    marker="o",
    markersize=16,
    markerfacecolor="#1F77B4",
    markeredgecolor="white",
    markeredgewidth=1.5,
    zorder=5,
)
ax2.plot(
    oi_config2,
    perf_config2_gop,
    label="Config 2: BMP GEMM (Low VRF BW)",
    color="#D62728",
    linewidth=2,
    alpha=0.8,
    marker="s",
    markersize=16,
    markerfacecolor="#D62728",
    markeredgecolor="white",
    markeredgewidth=1.5,
    zorder=5,
)
ax2.plot(
    oi_config3,
    perf_config3_gop,
    label="Config 3: BMP GEMM (High VRF BW)",
    color="#2CA02C",
    linewidth=2,
    alpha=0.8,
    marker="^",
    markersize=18,
    markerfacecolor="#2CA02C",
    markeredgecolor="white",
    markeredgewidth=1.5,
    zorder=5,
)

# ------------------- 新增：添加性能演进箭头 -------------------
# 选择一个具有代表性的数据点来展示演变过程，例如 size=256
target_size = 64
# 找到 target_size 在 sizes_theory 列表中的索引
try:
    target_idx = sizes_theory.index(target_size)
except ValueError:
    # 如果 target_size 不在列表中，则选择一个默认索引
    target_idx = 4  # 选择列表中的第5个点 (256)

# 获取三个配置在该尺寸下的性能和OI值 (GOPs/s)
oi_1, perf_1 = oi_config1[target_idx], perf_config1_gop[target_idx]
oi_2, perf_2 = oi_config2[target_idx], perf_config2_gop[target_idx]
oi_3, perf_3 = oi_config3[target_idx], perf_config3_gop[target_idx]

# 计算平均加速比
speedup_1_to_2 = 0
speedup_2_to_3 = 0
for i in range(0, 7):
    _, p1 = oi_config1[i], perf_config1_gop[i]
    _, p2 = oi_config2[i], perf_config2_gop[i]
    _, p3 = oi_config3[i], perf_config3_gop[i]
    speedup_1_to_2 += p2 / p1
    speedup_2_to_3 += p3 / p2
speedup_1_to_2 /= (len(sizes_theory) - target_idx - 1)
speedup_2_to_3 /= (len(sizes_theory) - target_idx - 1)

# 定义箭头样式
arrowprops = dict(
    arrowstyle="->",
    lw=3,
    color="gray",
    alpha=0.9,
    mutation_scale=20,  # 调整箭头头的大小
)

# 添加两个箭头并标注加速比
# 从 Config 1 (蓝线) 到 Config 2 (红线)
ax2.annotate("", xy=(oi_2, perf_2), xytext=(oi_1, perf_1), arrowprops=arrowprops)
ax2.text(
    (oi_1 + oi_2) / 2,
    (perf_1 + perf_2) / 2 - 40,
    f"×{speedup_1_to_2:.2f}",
    fontsize=20,
    ha="center",
    va="center",
    color="red",
    weight="bold",
)

# 从 Config 2 (红线) 到 Config 3 (绿线)
ax2.annotate("", xy=(oi_3, perf_3), xytext=(oi_2, perf_2), arrowprops=arrowprops)
ax2.text(
    (oi_2 + oi_3) / 2,
    (perf_2 + perf_3) / 2 - 20,
    f"×{speedup_2_to_3:.2f}",
    fontsize=20,
    ha="center",
    va="center",
    color="red",
    weight="bold",
)

# 可选：在箭头附近添加小标签，明确指示配置
ax2.text(
    oi_1,
    perf_1 * 1.5,
    "Config 1",
    fontsize=16,
    ha="center",
    va="top",
    color="#1F77B4",
    weight="bold",
)
ax2.text(
    oi_2,
    perf_2 * 0.4,
    "Config 2",
    fontsize=16,
    ha="center",
    va="bottom",
    color="#D62728",
    weight="bold",
)
ax2.text(
    oi_3,
    perf_3 * 2.0,
    "Config 3",
    fontsize=16,
    ha="center",
    va="top",
    color="#2CA02C",
    weight="bold",
)

# 坐标轴设置
ax2.set_xlim(0.5, 2000)
ax2.set_ylim(1, 1100)  # 根据数据微调
ax2.set_xlabel("Operational Intensity (OPs/Byte)", labelpad=10)
ax2.set_ylabel("Performance (GOPs/s)", labelpad=10)
# ax2.set_title(
#     "Roofline Analysis - Theoretical Results\n(Binary Mixed Precision GEMM vs RVV INT8 GEMM)",
#     pad=20,
# )

# 优化刻度显示
ax2.set_xticks([0.5, 10, 100, 1000])
ax2.set_xticklabels(["0.5", "10", "100", "1K"])
ax2.set_yticks([1, 10, 100, 200, 1000])
ax2.set_yticklabels(["1", "10", "100", "200", "1000"])

# 网格
ax2.grid(True, which="both", linestyle=":", alpha=0.6)

# 图例优化
legend2 = ax2.legend(
    loc="best", frameon=True, fancybox=True, shadow=True, ncol=1, borderpad=0.5
)
legend2.get_frame().set_facecolor("white")
legend2.get_frame().set_alpha(0.8)

plt.tight_layout()
plt.savefig("roofline_theoretical.png", dpi=300, bbox_inches="tight")
print("Theoretical Roofline plot saved as roofline_theoretical.png")
