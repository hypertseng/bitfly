import math
import numpy as np

from bmpmm_case_selection import selected_cases

SEED = 42
NR_LANES = 4
np.random.seed(SEED)

BENCH_CASES = [
    ("case1", 128, 128, 896),
    ("case2", 128, 256, 640),
    ("case3", 128, 256, 1536),
    ("case4", 128, 256, 2048),
    ("case5", 128, 320, 960),
]


def emit_quad_symbol(lines, name, words, align="32*NR_LANES"):
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 4):
        chunk = words[i:i + 4]
        lines.append("    .quad " + ", ".join(f"0x{w:016x}" for w in chunk))


def emit_int16_col_major(lines, name, array, align="32*NR_LANES"):
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


def emit_int16_row_major(lines, name, array, align="32*NR_LANES"):
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


def emit_case_aliases(lines, alias_name, target_name):
    symbol_stems = [
        "activation_lp",
        "weight_lp",
        "result_lp",
        "activation_hp",
        "weight_hp",
        "result_hp",
        "result_torch",
    ]
    lines.append(f"/* alias {alias_name} -> {target_name} */")
    for stem in symbol_stems:
        lines.append(f".global {stem}_{alias_name}")
        lines.append(f".set {stem}_{alias_name}, {stem}_{target_name}")

def emit_int8_row_major(lines, name, array, align="32*NR_LANES"):
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


def _pack_row_chunk(row_int8, k_chunk):
    chunk = np.asarray(row_int8[k_chunk * 8:(k_chunk + 1) * 8], dtype=np.int8)
    if chunk.shape[0] < 8:
        chunk = np.pad(chunk, (0, 8 - chunk.shape[0]), constant_values=0)
    return int(np.frombuffer(chunk.tobytes(), dtype=np.uint64)[0])


def pack_activations_lp(array, mtile):
    m_dim, k_dim = array.shape
    d = math.ceil(k_dim / 8)
    words = []
    for tile_m in range(0, m_dim, mtile):
        tile_rows = min(mtile, m_dim - tile_m)
        m_blocks = max(1, (tile_rows + 7) // 8)
        for m_block in range(m_blocks):
            base = tile_m + m_block * 8
            for k_chunk in range(d):
                even_lane_words = []
                odd_lane_words = []
                for lane in range(NR_LANES):
                    row0 = base + 2 * lane
                    row1 = row0 + 1
                    even_lane_words.append(_pack_row_chunk(array[row0], k_chunk) if row0 < m_dim else 0)
                    odd_lane_words.append(_pack_row_chunk(array[row1], k_chunk) if row1 < m_dim else 0)
                words.extend(even_lane_words)
                words.extend(odd_lane_words)
    return words


def _pack_8x8_bit_block(block_bits):
    word = 0
    for n_col in range(8):
        byte_val = 0
        for k_row in range(8):
            bit = int(block_bits[k_row, n_col]) & 0x1
            byte_val |= bit << (7 - k_row)
        word |= (byte_val & 0xFF) << (n_col * 8)
    return word


def _pack_weight_word(weight_mat, bits, plane, k_blk, n0):
    k_dim, n_dim = weight_mat.shape
    mask = (1 << bits) - 1
    block = np.zeros((8, 8), dtype=np.uint8)
    k_base = k_blk * 8
    for kr in range(8):
        for nc in range(8):
            kg = k_base + kr
            ng = n0 + nc
            if kg < k_dim and ng < n_dim:
                raw = int(weight_mat[kg, ng]) & mask
                block[kr, nc] = (raw >> plane) & 0x1
    return _pack_8x8_bit_block(block)


def pack_weights_bitplanes(weight_mat, bits, ntile):
    k_dim, n_dim = weight_mat.shape
    d = math.ceil(k_dim / 8)
    packed = []
    for tile_n in range(0, n_dim, ntile):
        tile_cols = min(ntile, n_dim - tile_n)
        n_blocks = max(1, (tile_cols + 15) // 16)
        for k_blk in range(d):
            for plane in range(bits):
                for n_block in range(n_blocks):
                    n_base = tile_n + n_block * 16
                    packed.append(_pack_weight_word(weight_mat, bits, plane, k_blk, n_base))
                    packed.append(_pack_weight_word(weight_mat, bits, plane, k_blk, n_base + 8))
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

    emit_quad_symbol(lines, f"activation_lp_square_{s}", pack_activations_lp(A, 8))
    emit_quad_symbol(lines, f"weight_lp_square_{s}", pack_weights_bitplanes(W, weight_bits, 64))
    emit_int16_col_major(lines, f"result_lp_square_{s}", np.zeros((s, s), dtype=np.int16))

    emit_int8_row_major(lines, f"activation_hp_square_{s}", A)
    emit_int8_row_major(lines, f"weight_hp_square_{s}", W)
    emit_int16_row_major(lines, f"result_hp_square_{s}", np.zeros((s, s), dtype=np.int16))
    emit_int16_row_major(lines, f"result_torch_square_{s}", (A.astype(np.int32) @ W.astype(np.int32)).astype(np.int16))


def build_bench_case_dataset(lines, case, prec):
    name = case["name"]
    m_dim, n_dim, k_dim = case["M"], case["N"], case["K"]
    cfg = case["cfg"]
    A = make_bench_case_activation(m_dim, k_dim)
    W = make_bench_case_weight(k_dim, n_dim, prec)
    weight_bits = 2 if prec == 2 else 4
    lines.append(
        f"/* {name}: shape=({m_dim},{n_dim},{k_dim}), "
        f"cfg=(mt={cfg['mtile']},nt={cfg['ntile']},kt={cfg['ktile']},gm={cfg['gm']},gn={cfg['gn']}), prec={prec} */"
    )
    emit_quad_symbol(lines, f"activation_lp_{name}", pack_activations_lp(A, cfg["mtile"]))
    emit_quad_symbol(lines, f"weight_lp_{name}", pack_weights_bitplanes(W, weight_bits, cfg["ntile"]))
    emit_int16_col_major(lines, f"result_lp_{name}", np.zeros((m_dim, n_dim), dtype=np.int16))
    emit_int8_row_major(lines, f"activation_hp_{name}", A)
    emit_int8_row_major(lines, f"weight_hp_{name}", W)
    emit_int16_row_major(lines, f"result_hp_{name}", np.zeros((m_dim, n_dim), dtype=np.int16))
    emit_int16_row_major(lines, f"result_torch_{name}", (A.astype(np.int32) @ W.astype(np.int32)).astype(np.int16))


def generate_lowp_dataset(prec, square_sizes=(32,), model_filter=None):
    lines = [".section .l2,\"aw\",@progbits", f"/* auto-generated, seed={SEED}, prec={prec} */"]
    for s in square_sizes:
        build_square_dataset(lines, s, prec)

    cases = selected_cases(prec, model_filter=model_filter)
    unique_case_by_key = {}
    for case in cases:
        cfg = case["cfg"]
        case_key = (case["M"], case["N"], case["K"], cfg["mtile"], cfg["ntile"], cfg["ktile"], cfg["gm"], cfg["gn"])
        lines.append(f"/* model={case['model']}, scale={case['scale']}, layer={case['layer']} */")
        if case_key in unique_case_by_key:
            emit_case_aliases(lines, case["name"], unique_case_by_key[case_key])
            continue
        unique_case_by_key[case_key] = case["name"]
        build_bench_case_dataset(lines, case, prec)
    return "\n".join(lines) + "\n"
