import matplotlib.pyplot as plt
import numpy as np

# 数据
K_values = [8, 16, 32, 64, 128, 256, 480]
perf_mixed = [12.37, 26.51, 48.33, 82.85, 127.88, 229.75, 247.87]  # OP/cycle
perf_vector = [5.48, 8.38, 9.41, 10.16, 10.61, 10.72, 10.94]
perf_scalar = [0.18, 0.20, 0.21, 0.22, 0.22, 0.20, 0.10]
speedup_mixed_vs_vector = [m / v for m, v in zip(perf_mixed, perf_vector)]

# 设置位置与宽度
x = np.arange(len(K_values))
width = 0.25

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 20,
        "axes.titlesize": 19,
        "axes.labelsize": 19,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,
    }
)

# 创建画布
fig, ax1 = plt.subplots(figsize=(8, 10))

# 配色方案
colors = {
    "scalar": "#a6a6a6",
    "vector": "#1f78b4",
    "mixed": "#33a02c",
    "speedup": "#e31a1c",
}

# 绘制柱状图（但不再标注 OP/cycle 数值）
bars1 = ax1.bar(
    x - width,
    perf_scalar,
    width,
    label="Scalar",
    color=colors["scalar"],
    edgecolor="black",
    linewidth=0.5,
)
bars2 = ax1.bar(
    x,
    perf_vector,
    width,
    label="RVV INT8",
    color=colors["vector"],
    edgecolor="black",
    linewidth=0.5,
)
bars3 = ax1.bar(
    x + width,
    perf_mixed,
    width,
    label="Binary Mixed Precision",
    color=colors["mixed"],
    edgecolor="black",
    linewidth=0.5,
)

# 主 Y 轴设置
ax1.set_yscale("log")
ax1.set_xlabel("K Dimension (M=16, N=32)", labelpad=10)
ax1.set_ylabel("Performance (OP/cycle), log scale", labelpad=10)
ax1.set_xticks(x)
ax1.set_xticklabels([str(k) for k in K_values])
ax1.grid(axis="y", linestyle="--", linewidth=0.6, which="both", alpha=0.7)
ax1.set_axisbelow(True)

# 图例（暂不显示，稍后合并）
ax1.legend(loc="upper left", frameon=False)

# 第二个 Y 轴：加速比曲线
ax2 = ax1.twinx()
line = ax2.plot(
    x,
    speedup_mixed_vs_vector,
    marker="D",
    markersize=8,
    color=colors["speedup"],
    linewidth=2.5,
    label="Speedup vs. RVV INT8",
)

# 只标注加速比数值（这才是重点）
for i, s in enumerate(speedup_mixed_vs_vector):
    ax2.annotate(
        f"{s:.1f}×",
        xy=(x[i], s),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=14,
        color=colors["speedup"],
    )

# 填充曲线下方区域以增强可读性
ax2.fill_between(x, speedup_mixed_vs_vector, color=colors["speedup"], alpha=0.1)

# 副 Y 轴设置
ax2.set_ylabel("Speedup (×)", color=colors["speedup"], labelpad=10)
ax2.tick_params(axis="y", labelcolor=colors["speedup"])
ax2.set_ylim(bottom=1)  # 加速比从1开始

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

# 去除顶部和右侧边框
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

# 紧凑布局
fig.tight_layout()

# 保存高清图
plt.savefig("kernel_profiling_seqlen.png", dpi=300, bbox_inches="tight")




# import matplotlib.pyplot as plt
# import numpy as np

# # 数据：M=K=N，从16到256
# sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
# mixed_perf = [10.27, 61.88, 139.88, 200.62, 440.19, 1286.31, 1907.06, 2049.46]  # Binary Mixed Precision (OP/cycle)
# vector_perf = [3.75, 9.45, 19.08, 22.65, 24.23, 25.54, 51.11, 51.13]  # RVV INT8 Baseline (OP/cycle)

# # 计算加速比
# speedup = [m / v for m, v in zip(mixed_perf, vector_perf)]

# # 设置位置
# x = np.arange(len(sizes))
# width = 0.35

# # 高级字体设置
# plt.rcParams.update(
#     {
#         "font.family": "sans-serif",
#         "font.size": 20,
#         "axes.titlesize": 19,
#         "axes.labelsize": 19,
#         "xtick.labelsize": 18,
#         "ytick.labelsize": 18,
#         "legend.fontsize": 17,
#     }
# )

# # 创建图像
# fig, ax1 = plt.subplots(figsize=(8, 10))

# # 配色方案（HPCA 推荐：清晰、高对比度）
# colors = {
#     "vector": "#1f78b4",  # 蓝色：baseline
#     "mixed": "#33a02c",  # 绿色：our design
#     "speedup": "#e31a1c",  # 红色：speedup line
# }

# # 绘制柱状图（双柱）
# bars1 = ax1.bar(
#     x - width / 2,
#     vector_perf,
#     width,
#     label="RVV INT8",
#     color=colors["vector"],
#     edgecolor="black",
#     linewidth=0.5,
# )
# bars2 = ax1.bar(
#     x + width / 2,
#     mixed_perf,
#     width,
#     label="Binary Mixed Precision",
#     color=colors["mixed"],
#     edgecolor="black",
#     linewidth=0.5,
# )

# # 主 Y 轴设置
# ax1.set_yscale("log")
# ax1.set_xlabel("Matrix Size (M=K=N)", labelpad=10)
# ax1.set_ylabel("Performance (OP/cycle), log scale", labelpad=10)
# ax1.set_xticks(x)
# ax1.set_xticklabels([f"{s}" for s in sizes])
# ax1.grid(axis="y", linestyle="--", linewidth=0.6, which="both", alpha=0.7)
# ax1.set_axisbelow(True)

# # 图例
# ax1.legend(loc="upper left", frameon=False)

# # 第二个 Y 轴：加速比
# ax2 = ax1.twinx()
# line = ax2.plot(
#     x,
#     speedup,
#     marker="D",
#     markersize=8,
#     color=colors["speedup"],
#     linewidth=2.5,
#     label="Speedup vs. RVV INT8",
# )

# # 填充加速比区域（可选，增强视觉效果）
# ax2.fill_between(x, speedup, 1, color=colors["speedup"], alpha=0.15)

# # 副 Y 轴设置
# ax2.axhline(y=1, color="gray", linestyle=":", linewidth=1.5)  # 加速比=1参考线
# ax2.set_ylabel("Speedup (×)", color=colors["speedup"], labelpad=10)
# ax2.tick_params(axis="y", labelcolor=colors["speedup"])
# ax2.set_ylim(bottom=1)  # 最小加速比为1

# # 添加加速比数值标签
# for i, s in enumerate(speedup):
#     ax2.annotate(
#         f"{s:.1f}×",
#         (x[i], s),
#         textcoords="offset points",
#         xytext=(0, 10),
#         ha="center",
#         fontsize=14,
#         color=colors["speedup"],
#     )

# # 合并图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

# # 去除顶部和右侧边框
# for spine in ["top", "right"]:
#     ax1.spines[spine].set_visible(False)
#     ax2.spines[spine].set_visible(False)

# # 紧凑布局
# fig.tight_layout()

# # 保存高清图
# plt.savefig("kernel_profiling_square.png", dpi=300, bbox_inches="tight")
