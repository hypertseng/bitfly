from itertools import product
import math

vlen = 8192
nr_lanes = 8
lmul = 8

k_values = [32, 64, 128, 256]
R_values = range(1, 8)
C_values = range(1, 8)
Ppe_values = range(1, 9)
Ptile_values = range(1, 9)


def check(constraint):
    return "✅" if constraint else "❌"


# 表头格式设置
header = (
    f"{'k':>4} {'R':>3} {'C':>3} {'Ppe':>5} {'Ptile':>6} {'t':>5} {'TOPS':>10} |"
    f" {'SRAM':^6} {'BW_Ppe':^8} {'BW_Tile':^10} {'BW_Total':^10} |"
    f" {'Actv_Valid':^11} {'Wght_Valid':^12} {'Out_Valid':^11}"
)
print("best config:")
print(header)
print("-" * len(header))

for k in k_values:
    best_config = None
    best_tops = -1

    for R, C, Ppe, Ptile in product(R_values, C_values, Ppe_values, Ptile_values):
        if R + C > 8:
            continue
        if 8 * Ppe > 64 or Ptile * Ppe > 64 or 8 * Ptile > 64:
            continue

        t = math.ceil(k / Ppe)

        actv_ok = (8 * t * R * Ppe) <= (vlen / nr_lanes * lmul)
        wght_ok = (t * C * (Ptile * Ppe // 8)) <= (vlen / nr_lanes * lmul)
        out_ok = (8 * R * C * Ptile) <= (vlen / nr_lanes * lmul)

        if not (actv_ok and wght_ok and out_ok):
            continue

        numerator = 2 * R * C * Ptile * (k - 1)
        denominator = R + t + C
        tops = numerator / denominator

        if tops > best_tops:
            best_tops = tops
            best_config = (k, R, C, Ppe, Ptile, t, tops, actv_ok, wght_ok, out_ok)

    if best_config:
        k, R, C, Ppe, Ptile, t, tops, actv_ok, wght_ok, out_ok = best_config
        print(
            f"{k:>4} {R:>3} {C:>3} {Ppe:>5} {Ptile:>6} {t:>5} {tops:>10.2f} |"
            f" {check(R + C <= 8):^6} {check(8 * Ppe <= 64):^8} {check(Ptile * Ppe <= 64):^10} {check(8 * Ptile <= 64):^10} |"
            f" {check(actv_ok):^11} {check(wght_ok):^12} {check(out_ok):^11}"
        )
    else:
        print(f"{k:>4} 无合法配置")
