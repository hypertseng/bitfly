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

M = 64
K = 64
N = 64
MTILE = 8
NTILE = 64
SEED = 7
NR_LANES = 4
np.random.seed(SEED)


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


def make_activation():
    a = np.zeros((M, K), dtype=np.int16)
    for m in range(M):
        for k in range(K):
            a[m, k] = ((m * 17 + k * 7 + 3) % 255) - 127
    return a.astype(np.int8)


def make_binary_weight():
    w = np.zeros((K, N), dtype=np.int8)
    for k in range(K):
        for n in range(N):
            w[k, n] = (k * 3 + n * 5 + 1) & 0x1
    return w


def make_int2_weight():
    lut = np.array([-2, -1, 0, 1], dtype=np.int8)
    w = np.zeros((K, N), dtype=np.int8)
    for k in range(K):
        for n in range(N):
            w[k, n] = lut[(k * 11 + n * 3 + 1) & 0x3]
    return w


def make_int4_weight():
    w = np.zeros((K, N), dtype=np.int8)
    for k in range(K):
        for n in range(N):
            w[k, n] = np.int8(((k * 11 + n * 5 + 3) & 0xF) - 8)
    return w


def emit_case(lines, tag, activation, weight, bits):
    if bits == 1:
        weight_ref = np.where(weight.astype(np.int32) != 0, 1, -1)
    else:
        weight_ref = weight.astype(np.int32)
    result = (activation.astype(np.int32) @ weight_ref).astype(np.int16)
    emit_quad_symbol(lines, f"activation_lp_{tag}", pack_activations_bmpu(activation, MTILE))
    emit_quad_symbol(lines, f"weight_lp_{tag}", pack_weights_bmpu(weight, bits, NTILE))
    emit_int16_col_major(lines, f"result_lp_{tag}", np.zeros((M, N), dtype=np.int16))
    emit_int8_row_major(lines, f"activation_hp_{tag}", activation)
    emit_int8_row_major(lines, f"weight_hp_{tag}", weight)
    emit_int16_row_major(lines, f"result_hp_{tag}", np.zeros((M, N), dtype=np.int16))
    emit_int16_col_major(lines, f"result_torch_{tag}", result)


lines = [".section .l2,\"aw\",@progbits", f"/* auto-generated for mytest, seed={SEED} */"]
activation = make_activation()
emit_case(lines, "case_binary", activation, make_binary_weight(), 1)
emit_case(lines, "case_int2", activation, make_int2_weight(), 2)
emit_case(lines, "case_int4", activation, make_int4_weight(), 4)
print("\n".join(lines) + "\n", end="")
