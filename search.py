import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib import gridspec
from numba import njit
import seaborn as sns
import pandas as pd

# ======================
# 配置参数
# ======================
configs = [(2,2048), (4,4096), (8,8192), (16,16384)]
R_values = np.arange(1, 8)
C_values = np.arange(1, 8)
valid_RC = [(R, C) for R in R_values for C in C_values if R + C <= 8]

@njit(fastmath=True)
def compute_OPC(k, NL, VLEN, R, C):
    """计算指定参数下的OPC矩阵(Plb×Ppe)"""
    Z = np.full((8, 8), np.nan)
    for i in range(8):  # Ppe
        Ppe = i + 1
        t = (k + Ppe - 1) // Ppe
        for j in range(8):  # Plb
            Plb = j + 1
            # 约束条件
            if 8*R*Plb + C*Plb*Ppe > 512 or 2*R*Ppe > 512:
                continue
            if (8*NL*t*R*Plb + NL*t*C*Plb*Ppe + NL*16*R*C*Ppe) > 32*VLEN:
                continue
            OPC = NL * R * C * Ppe * (2*t*Plb - 1) / (R + t + C)
            Z[j, i] = OPC
    return Z

def search_best_params(configs, k_base=256):
    """固定基准k搜索每个配置的最优参数"""
    rows = []
    for NL, VLEN in configs:
        best_val = -1
        best = None
        for (R, C) in valid_RC:
            Z = compute_OPC(k_base, NL, VLEN, R, C)
            if np.all(np.isnan(Z)):
                continue
            val = np.nanmax(Z)
            if val > best_val:
                j, i = np.unravel_index(np.nanargmax(Z), Z.shape)
                best = (NL, VLEN, k_base, R, C, i+1, j+1, val)
                best_val = val
        if best:
            rows.append(best)
    # ✅ 仅加 r 前缀，文字内容完全不变
    return pd.DataFrame(rows, columns=[
        r"$N_{\text{LANE}}$", 
        r"$\mathrm{VLEN}$", 
        "k", 
        "R", 
        "C", 
        r"$P_{PE}$", 
        r"$P_{LB-MAC}$", 
        "OPC"
    ])

def scan_k_for_fixed_params(df_best, k_values):
    """固定每个配置最优参数，扫描k值的OPC曲线"""
    rows = []
    for _, row in df_best.iterrows():
        # ✅ 仅加 r 前缀，文字不变
        NL, VLEN, R, C, Ppe, Plb = (
            row[r"$N_{\text{LANE}}$"], 
            row[r"$\mathrm{VLEN}$"], 
            row["R"], 
            row["C"], 
            row[r"$P_{PE}$"], 
            row[r"$P_{LB-MAC}$"]
        )
        for k in k_values:
            t = (k + Ppe - 1) // Ppe
            if 8*R*Plb + C*Plb*Ppe > 512 or 2*R*Ppe > 512:
                continue
            if (8*NL*t*R*Plb + NL*t*C*Plb*Ppe + NL*16*R*C*Ppe) > 32*VLEN:
                continue
            OPC = NL * R * C * Ppe * (2*t*Plb - 1) / (R + t + C)
            rows.append((NL, VLEN, k, OPC))
    # ✅ 仅加 r 前缀
    return pd.DataFrame(rows, columns=[
        r"$N_{\text{LANE}}$", 
        r"$\mathrm{VLEN}$", 
        "k", 
        "OPC"
    ])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib import gridspec
from numba import njit
import seaborn as sns
import pandas as pd

# 配置参数
configs = [(2,2048), (4,4096), (8,8192), (16,16384)]
R_values, C_values = np.arange(1, 8), np.arange(1, 8)
valid_RC = [(R, C) for R in R_values for C in C_values if R + C <= 8]

@njit(fastmath=True)
def compute_OPC(k, NL, VLEN, R, C):
    Z = np.full((8, 8), np.nan)
    for i in range(8):  # Ppe
        Ppe = i + 1
        t = (k + Ppe - 1) // Ppe
        for j in range(8):  # Plb
            Plb = j + 1
            if (8*R*Plb + C*Plb*Ppe > 512 or 2*R*Ppe > 512 or
                (8*NL*t*R*Plb + NL*t*C*Plb*Ppe + NL*16*R*C*Ppe) > 32*VLEN):
                continue
            OPC = NL * R * C * Ppe * (2*t*Plb - 1) / (R + t + C)
            Z[j, i] = OPC
    return Z

def search_best_params(configs, k_base=256):
    rows = []
    for NL, VLEN in configs:
        best_val = -1
        for (R, C) in valid_RC:
            Z = compute_OPC(k_base, NL, VLEN, R, C)
            if np.all(np.isnan(Z)): continue
            val = np.nanmax(Z)
            if val > best_val:
                j, i = np.unravel_index(np.nanargmax(Z), Z.shape)
                best = (NL, VLEN, k_base, R, C, i+1, j+1, val)
                best_val = val
        if best: rows.append(best)
    return pd.DataFrame(rows, columns=[
        r"$N_{\text{LANE}}$", r"$\mathrm{VLEN}$", "k", "R", "C", 
        r"$P_{PE}$", r"$P_{LB-MAC}$", "OPC"
    ])

def scan_k_for_fixed_params(df_best, k_values):
    rows = []
    for _, row in df_best.iterrows():
        NL, VLEN, R, C, Ppe, Plb = (
            row[r"$N_{\text{LANE}}$"], row[r"$\mathrm{VLEN}$"],
            row["R"], row["C"], row[r"$P_{PE}$"], row[r"$P_{LB-MAC}$"]
        )
        for k in k_values:
            t = (k + Ppe - 1) // Ppe
            if (8*R*Plb + C*Plb*Ppe > 512 or 2*R*Ppe > 512 or
                (8*NL*t*R*Plb + NL*t*C*Plb*Ppe + NL*16*R*C*Ppe) > 32*VLEN):
                continue
            OPC = NL * R * C * Ppe * (2*t*Plb - 1) / (R + t + C)
            rows.append((NL, VLEN, k, OPC))
    return pd.DataFrame(rows, columns=[
        r"$N_{\text{LANE}}$", r"$\mathrm{VLEN}$", "k", "OPC"
    ])

def visualize_hpca_double_row(df_k, df_best):
    """Vertical layout with improved aesthetics and section titles"""
    # ======== 放大字体 & 设置风格 ========
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 13,
            "text.usetex": False,
        }
    )
    fig = plt.figure(figsize=(10, 12))  # 总体高度稍作调整
    # Grid layout: 6 rows (titles, heatmaps, titles, contours, title, curve), 5 columns
    gs = gridspec.GridSpec(
        6, 5,  # 现在是6行
        figure=fig,
        height_ratios=[0.08, 1, 0.08, 1, 0.08, 1],  # 标题行高度设为0.08，等高线图高度压缩为0.8
        width_ratios=[1, 1, 1, 1, 0.05],
        wspace=0.3,  # 水平间距
        hspace=0.5,  # 垂直间距，足够大以避免重叠
    )
    # 全局归一化
    vmin = df_best["OPC"].min() * 0.7
    vmax = df_best["OPC"].max() * 1.1
    norm = PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax)
    cmap = "viridis"
    
    # ========== 第一行：热力图标题 ==========
    ax_title_hm = fig.add_subplot(gs[0, :4])
    ax_title_hm.axis('off')
    ax_title_hm.text(0.5, 0.5, 
                    "Design Space Exploration: R-C Dimension View",
                    ha='center', va='center', 
                    fontsize=13, weight='bold')
    
    # ========== 第二行：R-C 热力图 ==========
    for col, (NL, VLEN) in enumerate(configs):
        g = df_best[
            (df_best[r"$N_{\text{LANE}}$"] == NL) & 
            (df_best[r"$\mathrm{VLEN}$"] == VLEN)
        ]
        if g.empty:
            continue
        R_opt, C_opt, Ppe_opt, Plb_opt, k_opt, val_opt = g[
            ["R", "C", r"$P_{PE}$", r"$P_{LB-MAC}$", "k", "OPC"]
        ].iloc[0]
        ax_hm = fig.add_subplot(gs[1, col])
        heat = np.full((len(R_values), len(C_values)), np.nan)
        for r_idx, r in enumerate(R_values):
            for c_idx, c in enumerate(C_values):
                if (r, c) not in valid_RC:
                    continue
                Z = compute_OPC(k_opt, NL, VLEN, r, c)
                if not np.all(np.isnan(Z)):
                    heat[r_idx, c_idx] = np.nanmax(Z)
        mask = np.isnan(heat)
        sns.heatmap(
            heat,
            mask=mask,
            xticklabels=C_values,
            yticklabels=(R_values if col == 0 else False),
            cmap=cmap,
            cbar=False,
            square=True,
            ax=ax_hm,
            linewidths=0.5,
            linecolor="gray",
            norm=norm,
            alpha=0.9,
            annot=True,
            fmt=".0f",  # 关键修改：格式化为整数
            annot_kws={"fontsize": 6},  # 关键修改：减小字体大小
            vmin=vmin,
            vmax=vmax,
        )
        r_idx_opt = R_values.tolist().index(R_opt)
        c_idx_opt = C_values.tolist().index(C_opt)
        ax_hm.add_patch(
            plt.Rectangle(
                (c_idx_opt, r_idx_opt),
                1,
                1,
                fill=False,
                edgecolor="red",
                lw=2,
                zorder=5,
            )
        )
        ax_hm.text(
            c_idx_opt + 0.5,
            r_idx_opt + 0.5,
            f"{val_opt:.0f}",
            color="white",
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='white', boxstyle='round,pad=0.2'),
            zorder=6,
        )
        ax_hm.set_title(f"$N_{{LANE}}$={NL}, {r'$\mathrm{{VLEN}}$'}={VLEN//1024}K", fontsize=13, pad=2)  # 关键修改：添加 pad=2
        if col == 0:
            ax_hm.set_ylabel("$R$", fontsize=14)
        ax_hm.set_xlabel("$C$", fontsize=14)
    
    # ========== 第三行：等高线标题 ==========
    ax_title_contour = fig.add_subplot(gs[2, :4])
    ax_title_contour.axis('off')
    ax_title_contour.text(0.5, 0.5, 
                         "Design Space Exploration: $P_{PE}$-$P_{LB-MAC}$ Dimension View",
                         ha='center', va='center', 
                         fontsize=15, weight='bold')
    
    # ========== 第四行：Ppe-Plb 等高线 ==========
    for col, (NL, VLEN) in enumerate(configs):
        g = df_best[
            (df_best[r"$N_{\text{LANE}}$"] == NL) & 
            (df_best[r"$\mathrm{VLEN}$"] == VLEN)
        ]
        if g.empty:
            continue
        R_opt, C_opt, Ppe_opt, Plb_opt, k_opt, val_opt = g[
            ["R", "C", r"$P_{PE}$", r"$P_{LB-MAC}$", "k", "OPC"]
        ].iloc[0]
        ax_contour = fig.add_subplot(gs[3, col])
        Z_final = compute_OPC(k_opt, NL, VLEN, R_opt, C_opt)
        X, Y = np.meshgrid(np.arange(1, 9), np.arange(1, 9))
        cf = ax_contour.contourf(
            X, Y, Z_final, levels=12, cmap=cmap, norm=norm, alpha=0.85
        )
        cs = ax_contour.contour(
            X, Y, Z_final, levels=6, colors="white", linewidths=0.5, alpha=0.8
        )
        ax_contour.clabel(cs, fmt="%d", fontsize=10, inline=True, colors="w")
        ax_contour.scatter(
            Ppe_opt, Plb_opt, 
            c="red", 
            edgecolors="white", 
            s=70, 
            linewidth=1.2, 
            zorder=5
        )
        ax_contour.text(
            Ppe_opt - 1.0,
            Plb_opt + 0.3,
            f"{val_opt:.0f}",
            color="white",
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='white', boxstyle='round,pad=0.2'),
        )
        ax_contour.set_xlim(0.5, 8.5)
        ax_contour.set_ylim(0.5, 8.5)
        ax_contour.set_xlabel("$P_{PE}$", fontsize=14)
        if col == 0:
            ax_contour.set_ylabel("$P_{LB-MAC}$", fontsize=14)
        ax_contour.set_xticks(range(1, 9))
        ax_contour.set_yticks(range(1, 9))
        ax_contour.text(
            0.05, 
            0.95, 
            f"$R$={R_opt}, $C$={C_opt}", 
            transform=ax_contour.transAxes,
            fontsize=11,
            ha='left', 
            va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        # ax_contour.set_title(f"$N_{{LANE}}$={NL}, {r'$\mathrm{{VLEN}}$'}={VLEN//1024}K", fontsize=10, pad=2)  # 关键修改：添加 pad=2
    
    # ========== 第五行：OPC vs k 曲线标题 ==========
    ax_title_curve = fig.add_subplot(gs[4, :4])
    ax_title_curve.axis('off')
    ax_title_curve.text(0.5, 0.5, 
                        "Effect of $k$ on Achieved OPC Across Configurations",
                        ha='center', va='center', 
                        fontsize=15, weight='bold')
    
    # ========== 第六行：OPC vs k 曲线 ==========
    ax_curve = fig.add_subplot(gs[5, :5])  # 现在是第六行
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    for idx, ((NL, VLEN), g) in enumerate(df_k.groupby([r"$N_{\text{LANE}}$", r"$\mathrm{VLEN}$"])):
        color = colors[idx]
        ax_curve.plot(
            g["k"],
            g["OPC"],
            marker="o",
            markersize=4,
            color=color,
            label=f"$N_{{LANE}}$={NL}, {r'$\mathrm{{VLEN}}$'}={VLEN//1024}K",
            linewidth=1.5,
        )
        idx_max = g["OPC"].idxmax()
        ax_curve.scatter(
            g.loc[idx_max, "k"],
            g.loc[idx_max, "OPC"],
            c="red",
            edgecolors="black",
            s=70,
            zorder=5,
            linewidth=1.2,
        )
        ax_curve.text(
            g.loc[idx_max, "k"],
            g.loc[idx_max, "OPC"] * 1.05,  # Position above point
            f"{g.loc[idx_max, 'OPC']:.0f}",
            fontsize=13,
            color="red",
            ha="center",
            va="bottom",
            weight="bold",
        )
    ax_curve.set_xlabel("k", fontsize=14)
    ax_curve.set_ylabel("OPC", fontsize=14)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="upper left", frameon=True, ncol=2, fontsize=13)
    ax_curve.set_xlim(0, 520)
    
    # 右侧 colorbar
    cbar_ax = fig.add_subplot(gs[1:4, -1])  # Span heatmap and contour rows
    cbar = fig.colorbar(cf, cax=cbar_ax, label="OPC")
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("OPC", fontsize=14)
    
    # 保存图像
    plt.savefig("dse.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.savefig("dse.pdf", bbox_inches="tight")
    plt.close()
    print("✅ Saved: dse.{png,pdf}")

if __name__ == "__main__":
    df_best = search_best_params(configs, k_base=256)
    k_values = np.arange(8, 512 + 8, 8)
    df_k = scan_k_for_fixed_params(df_best, k_values)
    visualize_hpca_double_row(df_k, df_best)