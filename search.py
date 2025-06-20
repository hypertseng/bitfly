from itertools import product
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numba import njit  # For performance optimization
from matplotlib.colors import Normalize

# Constants
VLEN = 4096
NR_LANES = 8
LMUL = 8
MAX_PPE = 64
MAX_PRODUCT = 64

# Configuration ranges
k_values = np.array([16, 64, 256, 480], dtype=int)
R_values = np.arange(1, 8, dtype=int)
C_values = np.arange(1, 8, dtype=int)
Ppe_values = np.arange(1, 9, dtype=int)
Ptile_values = np.arange(1, 9, dtype=int)

# Precompute valid R,C pairs
valid_RC = [(R, C) for R in R_values for C in C_values if R + C <= 8]


@njit
def compute_tops(k, R, C, Pmac, Ppe):
    """Optimized TOPS calculation using Numba"""
    t = math.ceil(k / Pmac)

    # Check constraints
    if  8 * R * Pmac + C * Ppe * Pmac > 64 * 8 or 2 * R * Ppe > 64 * 8:
        return -1
    
    if (8 * t * R * Pmac) + (t * C * Ppe * Pmac) + (8 * R * C * Ppe) > 32 * VLEN:
        return -1

    numerator = 2 * R * C * Ppe * (k - 1)
    denominator = R + t + C
    return numerator / denominator


def find_best_config(k):
    """Find the best configuration for a given k value"""
    best_tops = -1
    best_config = None

    # First filter by valid R,C pairs
    for R, C in valid_RC:
        for Pmac in Ppe_values:
            # Early exit for Pmac constraints
            if 8 * Pmac > MAX_PPE:
                continue

            for Ppe in Ptile_values:
                tops = compute_tops(k, R, C, Pmac, Ppe)
                if tops > best_tops:
                    best_tops = tops
                    t = math.ceil(k / Pmac)
                    best_config = (k, R, C, Pmac, Ppe, t, tops)

    return best_config


def print_results():
    """Print the best configurations in a formatted table"""
    header = (
        f"{'k':>4} {'R':>3} {'C':>3} {'Pmac':>5} {'Ppe':>6} {'t':>5} {'TOPS':>10} |"
        f" {'SRAM':^6} {'BW_Ppe':^8} {'BW_Tile':^10} {'BW_Total':^10}"
    )
    print("best config:")
    print(header)
    print("-" * len(header))

    for k in k_values:
        config = find_best_config(k)
        if config:
            k, R, C, Pmac, Ppe, t, tops = config
            print(
                f"{k:>4} {R:>3} {C:>3} {Pmac:>5} {Ppe:>6} {t:>5} {tops:>10.2f} |"
                f" {'✅':^6} {'✅' if 8*Pmac<=64 else '❌':^8} "
                f"{'✅' if Ppe*Pmac<=64 else '❌':^10} {'✅' if 8*Ppe<=64 else '❌':^10}"
            )
        else:
            print(f"{k:>4} 无合法配置")


def generate_heatmaps():
    """Generate optimized heatmaps for each k value"""
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    # 为统一色阶，预先找出最大 TOPS 值
    global_max_tops = 0
    all_heatmaps = []

    for k in k_values:
        heatmap_data = np.full((len(R_values), len(C_values)), np.nan)
        for R, C in valid_RC:
            best_tops = -1
            for Pmac in Ppe_values:
                for Ppe in Ptile_values:
                    tops = compute_tops(k, R, C, Pmac, Ppe)
                    if tops > best_tops:
                        best_tops = tops
            if best_tops > 0:
                heatmap_data[R - 1, C - 1] = best_tops
                global_max_tops = max(global_max_tops, best_tops)
        all_heatmaps.append(heatmap_data)

    # 统一色阶范围
    norm = Normalize(vmin=0, vmax=global_max_tops)
    cmap = plt.colormaps["YlOrRd"]

    for idx, k in enumerate(k_values):
        ax = axes[idx]
        sns.heatmap(
            all_heatmaps[idx],
            xticklabels=C_values,
            yticklabels=R_values,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "TOPS"},
            mask=np.isnan(all_heatmaps[idx]),
            cmap=cmap,
            linewidths=0.5,
            linecolor="gray",
            square=True,
            ax=ax,
            norm=norm,
        )
        ax.set_title(f"Heatmap of TOPS (k={k})", fontsize=16)
        ax.set_xlabel("C (columns of array)", fontsize=14)
        ax.set_ylabel("R (rows of array)", fontsize=14)

    plt.tight_layout()
    plt.savefig("TOPS_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_3d_tops_surface(R=4, C=4, k=64):
    """
    3D曲面图展示Ppe和Ptile对TOPS的影响
    新增功能：
    1. 3D曲面+热力图叠加
    2. 自动标记最优配置点
    3. 动态视角旋转
    4. 约束边界可视化
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    # 生成网格数据
    Ppe_grid, Ptile_grid = np.meshgrid(Ppe_values, Ptile_values)
    tops_grid = np.zeros_like(Ppe_grid, dtype=float)

    # 计算每个点的TOPS
    for i in range(Ppe_grid.shape[0]):
        for j in range(Ppe_grid.shape[1]):
            tops_grid[i, j] = compute_tops(k, R, C, Ppe_grid[i, j], Ptile_grid[i, j])

    # 创建3D图形
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制3D曲面
    surf = ax.plot_surface(
        Ppe_grid,
        Ptile_grid,
        tops_grid,
        cmap=cm.coolwarm,
        alpha=0.8,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
    )

    # 添加等高线投影
    ax.contourf(
        Ppe_grid,
        Ptile_grid,
        tops_grid,
        zdir="z",
        offset=np.nanmin(tops_grid) - 10,
        cmap=cm.coolwarm,
        alpha=0.5,
    )

    # 标记最优配置点
    max_idx = np.unravel_index(np.nanargmax(tops_grid), tops_grid.shape)
    ax.scatter(
        Ppe_grid[max_idx],
        Ptile_grid[max_idx],
        tops_grid[max_idx],
        color="red",
        s=200,
        label=f"Max TOPS: {tops_grid[max_idx]:.1f}",
    )

    # 添加约束边界线
    for ppe in Ppe_values:
        valid_ptile = [
            pt for pt in Ptile_values if not np.isnan(compute_tops(k, R, C, ppe, pt))
        ]
        if valid_ptile:
            ax.plot(
                [ppe] * len(valid_ptile),
                valid_ptile,
                [np.nanmin(tops_grid) - 5] * len(valid_ptile),
                color="green",
                linewidth=3,
                alpha=0.7,
            )

    # 图形装饰
    ax.set_xlabel("Pmac Value", labelpad=15)
    ax.set_ylabel("Ppe Value", labelpad=15)
    ax.set_zlabel("TOPS Performance", labelpad=15)
    ax.set_title(
        f"3D TOPS Performance Surface\n(R={R}, C={C}, k={k})", y=1.05, fontsize=14
    )
    ax.legend()

    # 设置视角
    ax.view_init(elev=30, azim=45)

    # 保存动态旋转GIF
    try:
        from matplotlib.animation import FuncAnimation

        def update_view(frame):
            ax.view_init(elev=30, azim=frame)
            return (fig,)

        anim = FuncAnimation(fig, update_view, frames=np.arange(0, 360, 2), interval=50)
        anim.save(f"tops_3d_rotation_R{R}_C{C}_k{k}.gif", writer="pillow", dpi=100)
    except ImportError:
        pass

    # 保存静态图
    plt.savefig(f"tops_3d_surface_R{R}_C{C}_k{k}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print_results()
    generate_heatmaps()
    plot_3d_tops_surface(R=4, C=4, k=64)
