from pathlib import Path
import sys
import math
import numpy as np

COMMON = Path(__file__).resolve().parents[2] / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bmpmm_gen_common import (
    emit_quad_symbol,
    emit_int16_col_major,
    emit_int16_row_major,
    emit_int8_row_major,
)

SEED = 7
NR_LANES = 4
VRF_BITS = 2 * 64 * 64
CBUF_BITS = 16384
np.random.seed(SEED)

CASES = [
    {"tag": "case1", "name": "binary_mt8_nt16_kt64_g2x2",  "M": 64,  "K": 64, "N": 64,  "mtile": 8,  "ntile": 16, "ktile": 64, "gm": 2, "gn": 2, "prec": 0},
    {"tag": "case2", "name": "binary_mt16_nt16_kt64_g4x1", "M": 128, "K": 64, "N": 64,  "mtile": 16, "ntile": 16, "ktile": 64, "gm": 4, "gn": 1, "prec": 0},
    {"tag": "case3", "name": "binary_mt8_nt64_kt64_g1x2",  "M": 64,  "K": 64, "N": 128, "mtile": 8,  "ntile": 64, "ktile": 64, "gm": 1, "gn": 2, "prec": 0},
    {"tag": "case4", "name": "binary_mt32_nt32_kt32_g1x1", "M": 64,  "K": 32, "N": 64,  "mtile": 32, "ntile": 32, "ktile": 32, "gm": 1, "gn": 1, "prec": 0},
    {"tag": "case5", "name": "binary_mt24_nt32_kt32_g1x1", "M": 48,  "K": 32, "N": 64,  "mtile": 24, "ntile": 32, "ktile": 32, "gm": 1, "gn": 1, "prec": 0},
    {"tag": "case6", "name": "int2_mt8_nt32_kt64_g2x2",    "M": 64,  "K": 64, "N": 128, "mtile": 8,  "ntile": 32, "ktile": 64, "gm": 2, "gn": 2, "prec": 2},
    {"tag": "case7", "name": "int2_mt16_nt16_kt64_g1x4",   "M": 64,  "K": 64, "N": 64,  "mtile": 16, "ntile": 16, "ktile": 64, "gm": 1, "gn": 4, "prec": 2},
    {"tag": "case8", "name": "int4_mt8_nt16_kt32_g2x2",    "M": 64,  "K": 32, "N": 64,  "mtile": 8,  "ntile": 16, "ktile": 32, "gm": 2, "gn": 2, "prec": 3},
]


def weight_bits_from_prec(prec):
    if prec == 0:
        return 1
    if prec in (1, 2):
        return 2
    if prec == 3:
        return 4
    raise ValueError(f"unsupported precision {prec}")


def validate_case(case):
    g = case["gm"] * case["gn"]
    p_weight = weight_bits_from_prec(case["prec"])
    if case["ktile"] % 8 != 0:
        raise ValueError(f"{case['name']}: ktile must be multiple of 8")
    if g >= 8:
        raise ValueError(f"{case['name']}: g must be < 8, got {g}")
    if case["mtile"] * case["ktile"] * 8 > VRF_BITS:
        raise ValueError(f"{case['name']}: activation tile exceeds VRF capacity")
    if case["ntile"] * case["ktile"] * p_weight > VRF_BITS:
        raise ValueError(f"{case['name']}: weight tile exceeds VRF capacity")
    if case["mtile"] * case["ntile"] * g * 16 > CBUF_BITS:
        raise ValueError(f"{case['name']}: output tile exceeds accumulation buffer")


def _pack_row_chunk(row_int8, k_chunk):
    chunk = np.asarray(row_int8[k_chunk * 8:(k_chunk + 1) * 8], dtype=np.int8)
    if chunk.shape[0] < 8:
        chunk = np.pad(chunk, (0, 8 - chunk.shape[0]), constant_values=0)
    return int(np.frombuffer(chunk.tobytes(), dtype=np.uint64)[0])


def pack_activations_bmpu(array, mtile):
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


def pack_weights_bmpu(weight_mat, bits, ntile):
    k_dim, n_dim = weight_mat.shape
    d = math.ceil(k_dim / 8)
    words = []
    for tile_n in range(0, n_dim, ntile):
        tile_cols = min(ntile, n_dim - tile_n)
        n_blocks = max(1, (tile_cols + 15) // 16)
        for n_block in range(n_blocks):
            n_base = tile_n + n_block * 16
            for plane in range(bits):
                for k_blk in range(d):
                    bank2_word = _pack_weight_word(weight_mat, bits, plane, k_blk, n_base)
                    bank3_word = _pack_weight_word(weight_mat, bits, plane, k_blk, n_base + 8)
                    words.extend([bank2_word] * NR_LANES)
                    words.extend([bank3_word] * NR_LANES)
    return words


def make_activation(m_dim, k_dim):
    a = np.zeros((m_dim, k_dim), dtype=np.int16)
    for m in range(m_dim):
        for k in range(k_dim):
            a[m, k] = ((m * 17 + k * 7 + 3) % 255) - 127
    return a.astype(np.int8)


def make_binary_weight(k_dim, n_dim):
    w = np.zeros((k_dim, n_dim), dtype=np.int8)
    for k in range(k_dim):
        for n in range(n_dim):
            w[k, n] = (k * 3 + n * 5 + 1) & 0x1
    return w


def make_int2_weight(k_dim, n_dim):
    lut = np.array([-2, -1, 0, 1], dtype=np.int8)
    w = np.zeros((k_dim, n_dim), dtype=np.int8)
    for k in range(k_dim):
        for n in range(n_dim):
            w[k, n] = lut[(k * 11 + n * 3 + 1) & 0x3]
    return w


def make_int4_weight(k_dim, n_dim):
    w = np.zeros((k_dim, n_dim), dtype=np.int8)
    for k in range(k_dim):
        for n in range(n_dim):
            w[k, n] = np.int8(((k * 11 + n * 5 + 3) & 0xF) - 8)
    return w


def make_weight(case):
    bits = weight_bits_from_prec(case["prec"])
    if bits == 1:
        return make_binary_weight(case["K"], case["N"]), bits
    if bits == 2:
        return make_int2_weight(case["K"], case["N"]), bits
    return make_int4_weight(case["K"], case["N"]), bits


def emit_case(lines, case):
    validate_case(case)
    activation = make_activation(case["M"], case["K"])
    weight, bits = make_weight(case)
    if bits == 1:
        weight_ref = np.where(weight.astype(np.int32) != 0, 1, -1)
    else:
        weight_ref = weight.astype(np.int32)
    result = (activation.astype(np.int32) @ weight_ref).astype(np.int16)
    zeros = np.zeros((case["M"], case["N"]), dtype=np.int16)
    emit_quad_symbol(lines, f"activation_lp_{case['tag']}", pack_activations_bmpu(activation, case["mtile"]))
    emit_quad_symbol(lines, f"weight_lp_{case['tag']}", pack_weights_bmpu(weight, bits, case["ntile"]))
    emit_int16_col_major(lines, f"result_lp_{case['tag']}", zeros)
    emit_int8_row_major(lines, f"activation_hp_{case['tag']}", activation)
    emit_int8_row_major(lines, f"weight_hp_{case['tag']}", weight)
    emit_int16_row_major(lines, f"result_hp_{case['tag']}", zeros)
    emit_int16_col_major(lines, f"result_torch_{case['tag']}", result)


lines = [".section .l2,\"aw\",@progbits", f"/* auto-generated for mytest, seed={SEED} */"]
for case in CASES:
    emit_case(lines, case)
print("\n".join(lines) + "\n", end="")
