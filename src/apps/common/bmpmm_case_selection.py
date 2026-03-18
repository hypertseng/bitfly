from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]

MODEL_LAYER_CASES = [
    {"name": "case1", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 1024, "K": 640},
    {"name": "case2", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 256, "K": 640},
    {"name": "case3", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 256, "K": 640},
    {"name": "case4", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 640, "K": 1024},
    {"name": "case5", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 2048, "K": 640},
    {"name": "case6", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 2048, "K": 640},
    {"name": "case7", "scale": "tiny", "model": "google/gemma-3-270m", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 640, "K": 2048},
    {"name": "case8", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 896, "K": 896},
    {"name": "case9", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 128, "K": 896},
    {"name": "case10", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 128, "K": 896},
    {"name": "case11", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 896, "K": 896},
    {"name": "case12", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 4864, "K": 896},
    {"name": "case13", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 4864, "K": 896},
    {"name": "case14", "scale": "small", "model": "Qwen/Qwen2.5-0.5B", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 896, "K": 4864},
    {"name": "case15", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.q_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case16", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.k_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case17", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.v_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case18", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.self_attn.out_proj", "M": 128, "N": 2048, "K": 2048},
    {"name": "case19", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.fc1", "M": 128, "N": 8192, "K": 2048},
    {"name": "case20", "scale": "medium", "model": "facebook/opt-1.3b", "layer": "model.decoder.layers.0.fc2", "M": 128, "N": 2048, "K": 8192},
    {"name": "case21", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 1536, "K": 1536},
    {"name": "case22", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 256, "K": 1536},
    {"name": "case23", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 256, "K": 1536},
    {"name": "case24", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 1536, "K": 1536},
    {"name": "case25", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 8960, "K": 1536},
    {"name": "case26", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 8960, "K": 1536},
    {"name": "case27", "scale": "midlarge", "model": "Qwen/Qwen2.5-1.5B", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 1536, "K": 8960},
    {"name": "case28", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.q_proj", "M": 128, "N": 2048, "K": 2304},
    {"name": "case29", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.k_proj", "M": 128, "N": 1024, "K": 2304},
    {"name": "case30", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.v_proj", "M": 128, "N": 1024, "K": 2304},
    {"name": "case31", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.self_attn.o_proj", "M": 128, "N": 2304, "K": 2048},
    {"name": "case32", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.mlp.gate_proj", "M": 128, "N": 9216, "K": 2304},
    {"name": "case33", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.mlp.up_proj", "M": 128, "N": 9216, "K": 2304},
    {"name": "case34", "scale": "large", "model": "google/gemma-2-2b", "layer": "model.layers.0.mlp.down_proj", "M": 128, "N": 2304, "K": 9216},
]


def _config_for_prec(prec: int, shape: Dict[str, int]) -> Dict[str, int]:
    if prec == 2:
        return {"mtile": 8, "ntile": 64, "ktile": 64, "gm": 2, "gn": 1, "prec": prec}
    if prec == 3:
        return {"mtile": 8, "ntile": 64, "ktile": 32, "gm": 1, "gn": 1, "prec": prec}
    if prec == 0:
        return {"mtile": 8, "ntile": 64, "ktile": 128, "gm": 1, "gn": 2, "prec": prec}
    raise ValueError(f"unsupported prec {prec}")


def selected_cases(prec: int, csv_path: Path | None = None, count: int | None = None, max_ops: int | None = None, model_filter: str | None = None) -> List[Dict[str, int]]:
    model_filter = model_filter or os.environ.get("BMPMM_MODEL_FILTER")
    shapes = [shape for shape in MODEL_LAYER_CASES if model_filter is None or shape["model"] == model_filter]
    cases: List[Dict[str, int]] = []
    for index, shape in enumerate(shapes, start=1):
        case = dict(shape)
        case["name"] = f"case{index}"
        case["cfg"] = _config_for_prec(prec, shape)
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
    prec = int(sys.argv[2])
    model_filter = sys.argv[3] if len(sys.argv) > 3 else None
    write_bench_header(app_dir, prec, model_filter=model_filter)
