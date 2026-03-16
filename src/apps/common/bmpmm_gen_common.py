import math
import numpy as np

SEED = 42
np.random.seed(SEED)

BENCH_CASES = [
    ("case1", 128, 128, 896),
    ("case2", 128, 256, 640),
    ("case3", 128, 256, 1536),
    ("case4", 128, 256, 2048),
    ("case5", 128, 320, 960),
]


def emit_quad_symbol(lines, name, words, align="32 * NR_LANES"):
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 4):
        chunk = words[i:i + 4]
        lines.append("    .quad " + ", ".join(f"0x{w:016x}" for w in chunk))


def emit_int16_col_major(lines, name, array, align="32 * NR_LANES"):
    flat = np.asarray(array, dtype=np.int16).T.flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i:i + 8]))


def emit_int16_row_major(lines, name, array, align="32 * NR_LANES"):
    flat = np.asarray(array, dtype=np.int16).flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i:i + 8]))


def emit_int8_row_major(lines, name, array, align="32 * NR_LANES"):
    flat = np.asarray(array, dtype=np.int8).flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i:i + 8]))


def _pack_activation_row_to_words(row_int8):
    k_dim = row_int8.shape[0]
    d = math.ceil(k_dim / 8)
    padded = np.pad(row_int8.astype(np.int8), (0, d * 8 - k_dim), constant_values=0)
    words = []
    for i in range(d):
        chunk = padded[i * 8:(i + 1) * 8]
        words.append(int(np.frombuffer(chunk.tobytes(), dtype=np.uint64)[0]))
    return words


def pack_activations_lp(array):
    m_dim, _ = array.shape
    words = []
    for m in range(m_dim):
        words.extend(_pack_activation_row_to_words(array[m]))
    return words


def _pack_8x8_bit_block(block_bits):
    word = 0
    for k_row in range(8):
        byte_val = 0
        for n_col in range(8):
            bit = int(block_bits[k_row, n_col]) & 0x1
            byte_val |= bit << n_col
        word |= (byte_val & 0xFF) << (k_row * 8)
    return word


def pack_weights_bitplanes(weight_mat, bits):
    k_dim, n_dim = weight_mat.shape
    d = math.ceil(k_dim / 8)
    n_groups = math.ceil(n_dim / 8)
    mask = (1 << bits) - 1
    packed = []
    for kd_ext in range(d * bits):
        k_blk = kd_ext // bits
        plane = kd_ext % bits
        k0 = k_blk * 8
        for n_grp in range(n_groups):
            n0 = n_grp * 8
            block = np.zeros((8, 8), dtype=np.uint8)
            for kr in range(8):
                for nc in range(8):
                    kg = k0 + kr
                    ng = n0 + nc
                    if kg < k_dim and ng < n_dim:
                        raw = int(weight_mat[kg, ng]) & mask
                        block[kr, nc] = (raw >> plane) & 0x1
            packed.append(_pack_8x8_bit_block(block))
    return packed


def make_bench_case_activation(m_dim, k_dim):
    activation = np.zeros((m_dim, k_dim), dtype=np.int16)
    for m_idx in range(m_dim):
        for k_idx in range(k_dim):
            activation[m_idx, k_idx] = ((m_idx * 13 + k_idx * 7 + 5) % 255) - 127
    return activation.astype(np.int8)


def make_bench_case_weight(k_dim, n_dim, prec):
    if prec == 2:
        lut = np.array([-2, -1, 0, 1], dtype=np.int8)
        weight = np.zeros((k_dim, n_dim), dtype=np.int8)
        for k_idx in range(k_dim):
            for n_idx in range(n_dim):
                weight[k_idx, n_idx] = lut[(k_idx * 11 + n_idx * 3 + 1) & 0x3]
        return weight
    if prec == 3:
        weight = np.zeros((k_dim, n_dim), dtype=np.int8)
        for k_idx in range(k_dim):
            for n_idx in range(n_dim):
                weight[k_idx, n_idx] = np.int8(((k_idx * 11 + n_idx * 5 + 3) & 0xF) - 8)
        return weight
    raise ValueError(f"unsupported prec {prec}")


def make_square_weight(s, prec):
    if prec == 2:
        lut = np.array([-2, -1, 0, 1], dtype=np.int8)
        idx = np.random.randint(0, 4, size=(s, s))
        return lut[idx]
    if prec == 3:
        return np.random.randint(-8, 8, size=(s, s), dtype=np.int8)
    raise ValueError(f"unsupported prec {prec}")


def build_square_dataset(lines, s, prec):
    A = np.random.randint(-128, 128, size=(s, s), dtype=np.int16).astype(np.int8)
    W = make_square_weight(s, prec)
    weight_bits = 2 if prec == 2 else 4

    emit_quad_symbol(lines, f"activation_lp_square_{s}", pack_activations_lp(A))
    emit_quad_symbol(lines, f"weight_lp_square_{s}", pack_weights_bitplanes(W, weight_bits))
    emit_int16_col_major(lines, f"result_lp_square_{s}", np.zeros((s, s), dtype=np.int16))

    emit_int8_row_major(lines, f"activation_hp_square_{s}", A)
    emit_int8_row_major(lines, f"weight_hp_square_{s}", W)
    emit_int16_row_major(lines, f"result_hp_square_{s}", np.zeros((s, s), dtype=np.int16))
    emit_int16_row_major(lines, f"result_torch_square_{s}", (A.astype(np.int32) @ W.astype(np.int32)).astype(np.int16))


def build_bench_case_dataset(lines, name, m_dim, n_dim, k_dim, prec):
    A = make_bench_case_activation(m_dim, k_dim)
    W = make_bench_case_weight(k_dim, n_dim, prec)
    weight_bits = 2 if prec == 2 else 4
    lines.append(f"/* {name}: shape=({m_dim},{n_dim},{k_dim}), prec={prec} */")
    emit_quad_symbol(lines, f"activation_lp_{name}", pack_activations_lp(A))
    emit_quad_symbol(lines, f"weight_lp_{name}", pack_weights_bitplanes(W, weight_bits))
    emit_int16_col_major(lines, f"result_lp_{name}", np.zeros((m_dim, n_dim), dtype=np.int16))


def generate_lowp_dataset(prec, square_sizes=(32,)):
    lines = [".section .l2,\"aw\",@progbits", f"/* auto-generated, seed={SEED}, prec={prec} */"]
    for s in square_sizes:
        build_square_dataset(lines, s, prec)
    for name, m_dim, n_dim, k_dim in BENCH_CASES:
        build_bench_case_dataset(lines, name, m_dim, n_dim, k_dim, prec)
    return "\n".join(lines) + "\n"
