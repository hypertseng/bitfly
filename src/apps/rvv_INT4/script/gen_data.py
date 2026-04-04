import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np

COMMON = Path(__file__).resolve().parents[2] / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bmpmm_case_selection import MODEL_LAYER_CASES, infer_model_filter_from_app_name


SEED = 42
np.random.seed(SEED)
RVV_HP_ALIGN = "4096"
SKIP_RESULT_TORCH = os.environ.get("LOWP_SKIP_RESULT_TORCH", "1") == "1"
SKIP_LP_DATA = os.environ.get("LOWP_SKIP_LP_DATA", "1") == "1"
app_name = Path(__file__).resolve().parents[1].name
MODEL_FILTER = os.environ.get("BMPMM_MODEL_FILTER") or infer_model_filter_from_app_name(app_name)
MINIMAL_HP_DATA = os.environ.get("LOWP_MINIMAL_HP_DATA", "1") == "1"
MINIMAL_HP_ROWS = int(os.environ.get("LOWP_MINIMAL_HP_ROWS", "8"))
MINIMAL_HP_K = int(os.environ.get("LOWP_MINIMAL_HP_K", "64"))
USE_INCBIN = os.environ.get("LOWP_USE_INCBIN", "1") == "1"
GEN_DEBUG = os.environ.get("LOWP_GEN_DEBUG", "0") == "1"
_BLOB_DIR = None


def _dbg(msg):
    if GEN_DEBUG:
        print(msg, file=sys.stderr, flush=True)


def _blob_dir():
    global _BLOB_DIR
    if _BLOB_DIR is None:
        _BLOB_DIR = Path.cwd() / "data_blobs"
        _BLOB_DIR.mkdir(parents=True, exist_ok=True)
    return _BLOB_DIR


def _emit_incbin(lines, name, data: bytes, align):
    blob_path = (_blob_dir() / f"{name}.bin").resolve()
    blob_path.write_bytes(data)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    lines.append(f'    .incbin "{blob_path.as_posix()}"')


def selected_rvv_cases(model_filter):
    cases = [case for case in MODEL_LAYER_CASES if model_filter is None or case["model"] == model_filter]
    out = []
    for index, case in enumerate(cases, start=1):
        item = dict(case)
        item["name"] = f"case{index}"
        out.append(item)
    return out


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


def _pack_8x8_bitplane_block(block_bits):
    word = 0
    for n_col in range(8):
        byte_val = 0
        for k_row in range(8):
            bit = int(block_bits[k_row, n_col]) & 0x1
            byte_val |= bit << (7 - k_row)
        word |= (byte_val & 0xFF) << (n_col * 8)
    return word


def pack_weights_int4(weight_mat):
    k_dim, n_dim = weight_mat.shape
    d = math.ceil(k_dim / 8)
    n_blocks = math.ceil(n_dim / 16)
    packed = []
    for k_blk in range(d):
        k0 = k_blk * 8
        for plane in range(4):
            for n_blk in range(n_blocks):
                n0 = n_blk * 16
                for half in range(2):
                    block = np.zeros((8, 8), dtype=np.uint8)
                    for kr in range(8):
                        for nc in range(8):
                            kg = k0 + kr
                            ng = n0 + half * 8 + nc
                            if kg < k_dim and ng < n_dim:
                                raw = int(weight_mat[kg, ng]) & 0xF
                                block[kr, nc] = (raw >> plane) & 0x1
                    packed.append(_pack_8x8_bitplane_block(block))
    return packed, d


def make_top_shape_activation(m_dim, k_dim):
    m_idx = np.arange(m_dim, dtype=np.int32)[:, None]
    k_idx = np.arange(k_dim, dtype=np.int32)[None, :]
    activation = ((m_idx * 13 + k_idx * 7 + 5) % 255) - 127
    return activation.astype(np.int8)


def make_top_shape_weight(k_dim, n_dim):
    k_idx = np.arange(k_dim, dtype=np.int32)[:, None]
    n_idx = np.arange(n_dim, dtype=np.int32)[None, :]
    weight = ((k_idx * 11 + n_idx * 5 + 3) & 0xF) - 8
    return weight.astype(np.int8)


def emit_quad_symbol(lines, name, words, align=8):
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 4):
        chunk = words[i:i + 4]
        lines.append("    .quad " + ", ".join(f"0x{w:016x}" for w in chunk))


def emit_int16_col_major(lines, name, array, align="NR_LANES*4"):
    flat = np.asarray(array, dtype=np.int16).T.flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    if USE_INCBIN:
        _emit_incbin(lines, name, flat.tobytes(), align)
        return
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i:i + 8]))


def emit_int16_row_major(lines, name, array, align="NR_LANES*4"):
    flat = np.asarray(array, dtype=np.int16).flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    if USE_INCBIN:
        _emit_incbin(lines, name, flat.tobytes(), align)
        return
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i:i + 8]))


def emit_int8_row_major(lines, name, array, align="NR_LANES*4"):
    flat = np.asarray(array, dtype=np.int8).flatten()
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    if USE_INCBIN:
        _emit_incbin(lines, name, flat.tobytes(), align)
        return
    words = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    for i in range(0, len(words), 8):
        lines.append("    .word " + ", ".join(f"0x{int(v):08x}" for v in words[i:i + 8]))


def emit_symbol_alias(lines, alias_name, target_name):
    lines.append(f".global {alias_name}")
    lines.append(f".set {alias_name}, {target_name}")


def emit_placeholder_symbol(lines, name, align=8):
    lines.append(f".global {name}")
    lines.append(f".balign {align}")
    lines.append(f"{name}:")
    lines.append("    .word 0x00000000")


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
        emit_symbol_alias(lines, f"{stem}_{alias_name}", f"{stem}_{target_name}")


def build_square_dataset(lines, s):
    activation = make_top_shape_activation(s, s)
    weight = make_top_shape_weight(s, s)
    d = math.ceil(s / 8)
    lines.append(f"/* square={s}: int4, depth={d}, layout=8x16 bitplanes */")

    if SKIP_LP_DATA:
        emit_placeholder_symbol(lines, f"activation_lp_square_{s}")
        emit_placeholder_symbol(lines, f"weight_lp_square_{s}")
        emit_placeholder_symbol(lines, f"result_lp_square_{s}")
    else:
        emit_quad_symbol(lines, f"activation_lp_square_{s}", pack_activations_lp(activation))
        weight_words, _ = pack_weights_int4(weight)
        emit_quad_symbol(lines, f"weight_lp_square_{s}", weight_words)
        emit_int16_col_major(lines, f"result_lp_square_{s}", np.zeros((s, s), dtype=np.int16))

    emit_int8_row_major(lines, f"activation_hp_square_{s}", activation, align=RVV_HP_ALIGN)
    emit_int8_row_major(lines, f"weight_hp_square_{s}", weight, align=RVV_HP_ALIGN)
    emit_int16_row_major(lines, f"result_hp_square_{s}", np.zeros((s, s), dtype=np.int16), align=RVV_HP_ALIGN)
    if SKIP_RESULT_TORCH:
        emit_symbol_alias(lines, f"result_torch_square_{s}", f"result_hp_square_{s}")
    else:
        result = (activation.astype(np.int32) @ weight.astype(np.int32)).astype(np.int16)
        emit_int16_row_major(lines, f"result_torch_square_{s}", result, align=RVV_HP_ALIGN)


def build_top_shape_dataset(lines, case):
    name = case["name"]
    m_dim, n_dim, k_dim = case["M"], case["N"], case["K"]
    _dbg(f"[rvv_int4_gen] begin {name} shape=({m_dim},{n_dim},{k_dim})")
    hp_m_dim = min(m_dim, MINIMAL_HP_ROWS) if MINIMAL_HP_DATA else m_dim
    hp_k_dim = min(k_dim, MINIMAL_HP_K) if MINIMAL_HP_DATA else k_dim
    activation = make_top_shape_activation(hp_m_dim if MINIMAL_HP_DATA else m_dim, k_dim)
    weight = make_top_shape_weight(hp_k_dim if MINIMAL_HP_DATA else k_dim, n_dim)
    _dbg(f"[rvv_int4_gen] arrays_ready {name} hp_m={hp_m_dim} hp_k={hp_k_dim}")

    lines.append(f"/* {name}: shape=({m_dim},{n_dim},{k_dim}), layout=8x16 bitplanes */")
    if SKIP_LP_DATA:
        emit_placeholder_symbol(lines, f"activation_lp_{name}")
        emit_placeholder_symbol(lines, f"weight_lp_{name}")
        emit_placeholder_symbol(lines, f"result_lp_{name}")
    else:
        emit_quad_symbol(lines, f"activation_lp_{name}", pack_activations_lp(activation))
        weight_words, _ = pack_weights_int4(weight)
        emit_quad_symbol(lines, f"weight_lp_{name}", weight_words)
        emit_int16_col_major(lines, f"result_lp_{name}", np.zeros((m_dim, n_dim), dtype=np.int16))
    emit_int8_row_major(lines, f"activation_hp_{name}", activation, align=RVV_HP_ALIGN)
    emit_int8_row_major(lines, f"weight_hp_{name}", weight, align=RVV_HP_ALIGN)
    emit_int16_row_major(lines, f"result_hp_{name}", np.zeros((hp_m_dim, n_dim), dtype=np.int16), align=RVV_HP_ALIGN)
    if SKIP_RESULT_TORCH:
        emit_symbol_alias(lines, f"result_torch_{name}", f"result_hp_{name}")
    else:
        result = (activation.astype(np.int32) @ weight.astype(np.int32)).astype(np.int16)
        emit_int16_row_major(lines, f"result_torch_{name}", result, align=RVV_HP_ALIGN)
    _dbg(f"[rvv_int4_gen] done {name}")


def main():
    parser = argparse.ArgumentParser(description="Generate RVV INT4 datasets")
    parser.add_argument("--out", type=str, default="-", help="Output .S file path, '-' for stdout")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64], help="Square matrix sizes")
    args = parser.parse_args()

    lines = [".section .l2,\"aw\",@progbits", f"/* auto-generated by gen_data.py, seed={SEED} */"]
    for s in args.sizes:
        build_square_dataset(lines, s)

    bench_cases = selected_rvv_cases(MODEL_FILTER)
    unique_case_by_shape = {}
    for case in bench_cases:
        case_key = (case["M"], case["N"], case["K"])
        lines.append(f"/* model={case['model']}, scale={case['scale']}, layer={case['layer']} */")
        if MODEL_FILTER is not None and case_key in unique_case_by_shape:
            emit_case_aliases(lines, case["name"], unique_case_by_shape[case_key])
            continue
        unique_case_by_shape[case_key] = case["name"]
        build_top_shape_dataset(lines, case)

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
