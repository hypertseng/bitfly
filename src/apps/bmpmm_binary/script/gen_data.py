import argparse
import math
from pathlib import Path

import numpy as np


SEED = 42
np.random.seed(SEED)

TOP_SHAPES = [
    ("top1", 128, 128, 896),
    ("top2", 128, 256, 640),
    ("top3", 128, 256, 1536),
    ("top4", 128, 256, 2048),
    ("top5", 128, 320, 960),
]


def _pack_activation_row_to_words(row_int8: np.ndarray):
    k_dim = row_int8.shape[0]
    d = math.ceil(k_dim / 8)
    padded = np.pad(row_int8.astype(np.int8), (0, d * 8 - k_dim), constant_values=0)
    words = []
    for i in range(d):
        chunk = padded[i * 8 : (i + 1) * 8]
        word = int(np.frombuffer(chunk.tobytes(), dtype=np.uint64)[0])
        words.append(word)
    return words


def pack_activations_lp(array: np.ndarray):
    m_dim, k_dim = array.shape
    d = math.ceil(k_dim / 8)
    words = []
    for m in range(m_dim):
        words.extend(_pack_activation_row_to_words(array[m]))
    return words, d


def _pack_8x8_bit_block(block_bits: np.ndarray):
    word = 0
    for k_row in range(8):
        byte_val = 0
        for n_col in range(8):
            bit = int(block_bits[k_row, n_col]) & 0x1
            byte_val |= bit << n_col
        word |= (byte_val & 0xFF) << (k_row * 8)
    return word


def pack_weights_binary(weight_mat: np.ndarray):
    k_dim, n_dim = weight_mat.shape
    d = math.ceil(k_dim / 8)
    n_groups = math.ceil(n_dim / 8)

    packed = []
    for k_blk in range(d):
        k0 = k_blk * 8
        for n_grp in range(n_groups):
            n0 = n_grp * 8
            block = np.zeros((8, 8), dtype=np.uint8)
            for kr in range(8):
                for nc in range(8):
                    kg = k0 + kr
                    ng = n0 + nc
                    if kg < k_dim and ng < n_dim:
                        block[kr, nc] = int(weight_mat[kg, ng]) & 0x1
            packed.append(_pack_8x8_bit_block(block))
    return packed, d


def make_top_shape_activation(m_dim: int, k_dim: int):
    activation = np.zeros((m_dim, k_dim), dtype=np.int16)
    for m_idx in range(m_dim):
        for k_idx in range(k_dim):
            activation[m_idx, k_idx] = ((m_idx * 13 + k_idx * 7 + 5) % 255) - 127
    return activation.astype(np.int8)


def make_top_shape_weight(k_dim: int, n_dim: int):
    weight = np.zeros((k_dim, n_dim), dtype=np.int8)
    for k_idx in range(k_dim):
        for n_idx in range(n_dim):
            weight[k_idx, n_idx] = (k_idx * 11 + n_idx * 3 + 1) & 0x1
    return weight


def emit_quad_symbol(lines, name, words, align=8):
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 4):
        chunk = words[i : i + 4]
        lines.append("    .quad " + ", ".join(f"0x{w:016x}" for w in chunk))


def emit_int16_col_major(lines, name, array, align="NR_LANES*4"):
    flat = np.asarray(array, dtype=np.int16).T.flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i : i + 8]))


def emit_int16_row_major(lines, name, array, align="NR_LANES*4"):
    flat = np.asarray(array, dtype=np.int16).flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i : i + 8]))


def emit_int8_row_major(lines, name, array, align="NR_LANES*4"):
    flat = np.asarray(array, dtype=np.int8).flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i : i + 8]))


def compute_m_v_binary(d: int):
    planes = 1
    num = 2 * planes * d
    den = d * (planes - 1) + 3 * planes
    return max(2, (num + den - 1) // den)


def make_representative_activation(s: int):
    """Build one-shape activations with mixed patterns for better coverage."""
    a = np.random.randint(-128, 128, size=(s, s), dtype=np.int16)

    # Add deterministic edge-heavy rows to exercise saturation/packing corners.
    for i in range(min(4, s)):
        a[i, :] = 127 if (i % 2 == 0) else -128

    # Add a checkerboard-like mid-region to mix signs and small magnitudes.
    if s >= 8:
        r0 = s // 4
        r1 = min(s, r0 + s // 4)
        c0 = s // 4
        c1 = min(s, c0 + s // 4)
        rr, cc = np.indices((r1 - r0, c1 - c0))
        checker = np.where((rr + cc) % 2 == 0, 17, -19).astype(np.int16)
        a[r0:r1, c0:c1] = checker

    return a


def make_representative_binary_weight(s: int):
    """Build one-shape weights with mixed density/structure in the same matrix."""
    w = np.random.randint(0, 2, size=(s, s), dtype=np.int16)

    # Force a sparse stripe block.
    if s >= 8:
        w[: s // 8, :] = 0

    # Force a dense stripe block.
    if s >= 8:
        w[s // 8 : s // 4, :] = 1

    # Embed checkerboard block.
    if s >= 16:
        r0 = s // 2
        r1 = min(s, r0 + s // 8)
        c0 = s // 2
        c1 = min(s, c0 + s // 8)
        rr, cc = np.indices((r1 - r0, c1 - c0))
        w[r0:r1, c0:c1] = ((rr + cc) & 1).astype(np.int16)

    return w


def build_square_dataset(lines, s: int):
    a = make_representative_activation(s)
    w_src = make_representative_binary_weight(s)

    d = math.ceil(s / 8)
    m_v = compute_m_v_binary(d)
    lines.append(f"/* square={s}: binary P=1, D={d}, adaptive M_v={m_v} */")

    a_lp_words, _ = pack_activations_lp(a.astype(np.int8))
    emit_quad_symbol(lines, f"activation_lp_square_{s}", a_lp_words)

    w_lp_words, _ = pack_weights_binary(w_src.astype(np.int8))
    emit_quad_symbol(lines, f"weight_lp_square_{s}", w_lp_words)
    lines.append(f"/* weight_lp_square_{s} layout: single-plane, depth={d} */")

    emit_int16_col_major(lines, f"result_lp_square_{s}", np.zeros((s, s), dtype=np.int16))

    emit_int8_row_major(lines, f"activation_hp_square_{s}", a)
    b_hp = np.where(w_src == 0, -1, 1).astype(np.int16)
    emit_int8_row_major(lines, f"weight_hp_square_{s}", b_hp)
    emit_int16_row_major(lines, f"result_hp_square_{s}", np.zeros((s, s), dtype=np.int16))

    c = (a.astype(np.int32) @ b_hp.astype(np.int32)).astype(np.int16)
    emit_int16_row_major(lines, f"result_torch_square_{s}", c)


def build_top_shape_dataset(lines, name: str, m_dim: int, n_dim: int, k_dim: int):
    activation = make_top_shape_activation(m_dim, k_dim)
    weight = make_top_shape_weight(k_dim, n_dim)

    lines.append(f"/* {name}: shape=({m_dim},{n_dim},{k_dim}) */")
    activation_words, _ = pack_activations_lp(activation)
    emit_quad_symbol(lines, f"activation_lp_{name}", activation_words)

    weight_words, _ = pack_weights_binary(weight)
    emit_quad_symbol(lines, f"weight_lp_{name}", weight_words)

    emit_int16_col_major(lines, f"result_lp_{name}", np.zeros((m_dim, n_dim), dtype=np.int16))


def main():
    parser = argparse.ArgumentParser(description="Generate bare-metal bmpmm binary datasets")
    parser.add_argument("--out", type=str, default="-", help="Output .S file path, '-' for stdout")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64], help="Square matrix sizes")
    args = parser.parse_args()

    lines = [".section .data", f"/* auto-generated by gen_data.py, seed={SEED} */"]
    for s in args.sizes:
        build_square_dataset(lines, s)
    for name, m_dim, n_dim, k_dim in TOP_SHAPES:
        build_top_shape_dataset(lines, name, m_dim, n_dim, k_dim)

    text = "\n".join(lines) + "\n"
    if args.out == "-":
        print(text, end="")
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
