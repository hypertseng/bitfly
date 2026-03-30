from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

PREC_TO_SEARCH_CSV = {
    0: "llm_shape_best_binary.csv",
    1: "llm_shape_best_int2.csv",
    2: "llm_shape_best_int2.csv",
    3: "llm_shape_best_int4.csv",
}

APP_MODEL_FILTERS = {
    "gemma3_270m": "google/gemma-3-270m",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "qwen25_05b": "Qwen/Qwen2.5-0.5B",
    "replit_code_v1_3b": "replit/replit-code-v1-3b",
    "tinyllama_11b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "opt_13b": "facebook/opt-1.3b",
    "qwen25_15b": "Qwen/Qwen2.5-1.5B",
    "stablelm2_16b": "stabilityai/stablelm-2-1_6b",
    "smollm2_17b": "HuggingFaceTB/SmolLM2-1.7B",
    "gemma2_2b": "google/gemma-2-2b",
}

MODEL_LAYER_CASES = [
    {"name": "case1", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 1024, "K": 640},
    {"name": "case2", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 256, "K": 640},
    {"name": "case3", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 256, "K": 640},
    {"name": "case4", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 640, "K": 1024},
    {"name": "case5", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 2048, "K": 640},
    {"name": "case6", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 2048, "K": 640},
    {"name": "case7", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 640, "K": 2048},
    {"name": "case8", "scale": "smol360m", "model": "HuggingFaceTB/SmolLM2-360M", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 960, "K": 960},
    {"name": "case9", "scale": "smol360m", "model": "HuggingFaceTB/SmolLM2-360M", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 320, "K": 960},
    {"name": "case10", "scale": "smol360m", "model": "HuggingFaceTB/SmolLM2-360M", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 320, "K": 960},
    {"name": "case11", "scale": "smol360m", "model": "HuggingFaceTB/SmolLM2-360M", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 960, "K": 960},
    {"name": "case12", "scale": "smol360m", "model": "HuggingFaceTB/SmolLM2-360M", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 2560, "K": 960},
    {"name": "case13", "scale": "smol360m", "model": "HuggingFaceTB/SmolLM2-360M", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 2560, "K": 960},
    {"name": "case14", "scale": "smol360m", "model": "HuggingFaceTB/SmolLM2-360M", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 960, "K": 2560},
    {"name": "case15", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 896, "K": 896},
    {"name": "case16", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 128, "K": 896},
    {"name": "case17", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 128, "K": 896},
    {"name": "case18", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 896, "K": 896},
    {"name": "case19", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 4864, "K": 896},
    {"name": "case20", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 4864, "K": 896},
    {"name": "case21", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 896, "K": 4864},
    {"name": "case22", "scale": "code3b", "model": "replit/replit-code-v1-3b", "layer": "transformer.blocks.0.attn.Wqkv", "M": 128, "N": 7680, "K": 2560},
    {"name": "case23", "scale": "code3b", "model": "replit/replit-code-v1-3b", "layer": "transformer.blocks.0.attn.out_proj", "M": 128, "N": 2560, "K": 2560},
    {"name": "case24", "scale": "code3b", "model": "replit/replit-code-v1-3b", "layer": "transformer.blocks.0.ffn.up_proj", "M": 128, "N": 10240, "K": 2560},
    {"name": "case25", "scale": "code3b", "model": "replit/replit-code-v1-3b", "layer": "transformer.blocks.0.ffn.down_proj", "M": 128, "N": 2560, "K": 10240},
    {"name": "case26", "scale": "tinyllama", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case27", "scale": "tinyllama", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 256, "K": 2048},
    {"name": "case28", "scale": "tinyllama", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 256, "K": 2048},
    {"name": "case29", "scale": "tinyllama", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case30", "scale": "tinyllama", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 5632, "K": 2048},
    {"name": "case31", "scale": "tinyllama", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 5632, "K": 2048},
    {"name": "case32", "scale": "tinyllama", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 2048, "K": 5632},
    {"name": "case33", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.q_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case34", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.k_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case35", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.v_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case36", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.out_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case37", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.fc1", "M": 128, "N": 8192, "K": 2048},
    {"name": "case38", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.fc2", "M": 128, "N": 2048, "K": 8192},
    {"name": "case39", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 1536, "K": 1536},
    {"name": "case40", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 256, "K": 1536},
    {"name": "case41", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 256, "K": 1536},
    {"name": "case42", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 1536, "K": 1536},
    {"name": "case43", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 8960, "K": 1536},
    {"name": "case44", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 8960, "K": 1536},
    {"name": "case45", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 1536, "K": 8960},
    {"name": "case46", "scale": "stablelm", "model": "stabilityai/stablelm-2-1_6b", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case47", "scale": "stablelm", "model": "stabilityai/stablelm-2-1_6b", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case48", "scale": "stablelm", "model": "stabilityai/stablelm-2-1_6b", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case49", "scale": "stablelm", "model": "stabilityai/stablelm-2-1_6b", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case50", "scale": "stablelm", "model": "stabilityai/stablelm-2-1_6b", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 5632, "K": 2048},
    {"name": "case51", "scale": "stablelm", "model": "stabilityai/stablelm-2-1_6b", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 5632, "K": 2048},
    {"name": "case52", "scale": "stablelm", "model": "stabilityai/stablelm-2-1_6b", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 2048, "K": 5632},
    {"name": "case53", "scale": "smol17b", "model": "HuggingFaceTB/SmolLM2-1.7B", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case54", "scale": "smol17b", "model": "HuggingFaceTB/SmolLM2-1.7B", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case55", "scale": "smol17b", "model": "HuggingFaceTB/SmolLM2-1.7B", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case56", "scale": "smol17b", "model": "HuggingFaceTB/SmolLM2-1.7B", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case57", "scale": "smol17b", "model": "HuggingFaceTB/SmolLM2-1.7B", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 8192, "K": 2048},
    {"name": "case58", "scale": "smol17b", "model": "HuggingFaceTB/SmolLM2-1.7B", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 8192, "K": 2048},
    {"name": "case59", "scale": "smol17b", "model": "HuggingFaceTB/SmolLM2-1.7B", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 2048, "K": 8192},
    {"name": "case60", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 2048, "K": 2304},
    {"name": "case61", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 1024, "K": 2304},
    {"name": "case62", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 1024, "K": 2304},
    {"name": "case63", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 2304, "K": 2048},
    {"name": "case64", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 9216, "K": 2304},
    {"name": "case65", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 9216, "K": 2304},
    {"name": "case66", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 2304, "K": 9216},
]


def _default_csv_path_for_prec(prec: int) -> Path:
    csv_name = PREC_TO_SEARCH_CSV.get(prec)
    if csv_name is None:
        raise ValueError(f"unsupported prec {prec}")
    return REPO_ROOT / "tmp" / csv_name


def _load_shape_cfgs(csv_path: Path, prec: int) -> Dict[tuple[int, int, int], Dict[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing shape config csv: {csv_path}")

    configs: Dict[tuple[int, int, int], Dict[str, int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            cfg = {
                "mtile": int(row["mtile"]),
                "ntile": int(row["ntile"]),
                "ktile": int(row["ktile"]),
                "gm": int(row["gm"]),
                "gn": int(row["gn"]),
                "prec": prec,
            }
            if key in configs and configs[key] != cfg:
                raise ValueError(f"conflicting configs for shape {key} in {csv_path}")
            configs[key] = cfg
    return configs


def infer_prec_from_app_name(app_name: str) -> int:
    if "_INT2" in app_name:
        return 2
    if "_INT4" in app_name:
        return 3
    if "_binary" in app_name:
        return 0
    raise ValueError(f"cannot infer precision from app name: {app_name}")


def infer_model_filter_from_app_name(app_name: str) -> str | None:
    for suffix, model_name in APP_MODEL_FILTERS.items():
        if app_name.endswith(f"_{suffix}"):
            return model_name
    return None


def selected_cases(prec: int, csv_path: Path | None = None, count: int | None = None, max_ops: int | None = None, model_filter: str | None = None) -> List[Dict[str, int]]:
    model_filter = model_filter or os.environ.get("BMPMM_MODEL_FILTER")
    csv_path = csv_path or _default_csv_path_for_prec(prec)
    cfg_by_shape = _load_shape_cfgs(csv_path, prec)
    shapes = [shape for shape in MODEL_LAYER_CASES if model_filter is None or shape["model"] == model_filter]
    cases: List[Dict[str, int]] = []
    for index, shape in enumerate(shapes, start=1):
        case = dict(shape)
        case["name"] = f"case{index}"
        shape_key = (shape["M"], shape["N"], shape["K"])
        if shape_key not in cfg_by_shape:
            raise KeyError(f"missing searched config for shape {shape_key} in {csv_path}")
        case["cfg"] = dict(cfg_by_shape[shape_key])
        cases.append(case)
    return cases


def emit_bench_header(app_name: str, prec: int, csv_path: Path | None = None, model_filter: str | None = None) -> str:
    cases = selected_cases(prec, csv_path=csv_path, model_filter=model_filter)
    lines = [
        "#ifndef BMPMM_BENCH_CASES_GENERATED_H",
        "#define BMPMM_BENCH_CASES_GENERATED_H",
        "",
        "#include \"../../common/bmpmm_bench_common.h\"",
        "",
        f"#define BMPMM_BENCH_CASE_COUNT {len(cases)}",
        "",
        "static const bmpmm_bench_case_t kBenchCases[BMPMM_BENCH_CASE_COUNT] = {",
    ]
    for case in cases:
        cfg = case["cfg"]
        lines.append(
            f'    {{"{case["scale"]}", "{case["model"]}", "{case["layer"]}", {case["M"]}UL, {case["N"]}UL, {case["K"]}UL, '
            f'{{{cfg["mtile"]}UL, {cfg["ntile"]}UL, {cfg["ktile"]}UL, {cfg["gm"]}UL, {cfg["gn"]}UL, {cfg["prec"]}UL}}}},'
        )
    lines.extend(["};", "", "#endif", ""])
    return "\n".join(lines)


def write_bench_header(app_dir: Path, prec: int, csv_path: Path | None = None, model_filter: str | None = None) -> None:
    out = app_dir / "kernel" / "bench_cases.h"
    out.write_text(emit_bench_header(app_dir.name, prec, csv_path=csv_path, model_filter=model_filter))


if __name__ == "__main__":
    import sys
    app_dir = Path(sys.argv[1]).resolve()
    app_name = app_dir.name
    prec = int(sys.argv[2]) if len(sys.argv) > 2 else infer_prec_from_app_name(app_name)
    model_filter = sys.argv[3] if len(sys.argv) > 3 else infer_model_filter_from_app_name(app_name)
    csv_path = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
    write_bench_header(app_dir, prec, csv_path=csv_path, model_filter=model_filter)
