#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoConfig = None
    AutoModelForCausalLM = None


def parse_int_list(expr: str | None) -> List[int]:
    if not expr:
        return []
    out: List[int] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_str_list(expr: str | None) -> List[str]:
    if not expr:
        return []
    out = []
    for part in expr.split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def read_models_file(path: str) -> List[str]:
    models: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                models.append(line)
    return models


def discover_local_hf_models(root: str) -> List[str]:
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return []

    models: List[str] = []
    for child in sorted(root_path.iterdir()):
        name = child.name
        if not child.is_dir() or not name.startswith("models--"):
            continue
        repo_id = name[len("models--"):].replace("--", "/", 1)
        models.append(repo_id)
    return models


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def infer_mnk_from_linear(x: torch.Tensor, weight: torch.Tensor) -> Optional[Tuple[int, int, int]]:
    if x.dim() < 2 or weight.dim() != 2:
        return None
    k_in = x.shape[-1]
    if weight.shape[1] == k_in:
        # Linear: weight [N, K]
        n = weight.shape[0]
        k = k_in
    elif weight.shape[0] == k_in:
        # Conv1D in GPT2 style: weight [K, N]
        n = weight.shape[1]
        k = k_in
    else:
        return None
    m = int(x.numel() // k_in)
    return (m, n, k)


def main() -> None:
    default_models_file = Path(__file__).with_name("models.txt")
    ap = argparse.ArgumentParser(
        description=(
            "Extract unique GEMM shapes (M,N,K) from Linear layers of LLMs at runtime. "
            "Models must be <=3B params and include at least one <=1B model."
        )
    )
    ap.add_argument(
        "--models",
        type=str,
        default="",
        help="comma-separated model names or local paths",
    )
    ap.add_argument(
        "--models-file",
        type=str,
        default=str(default_models_file) if default_models_file.is_file() else "",
        help="text file with one model name/path per line",
    )
    ap.add_argument(
        "--all-local-models",
        action="store_true",
        help="append all locally cached HF models under --local-model-root",
    )
    ap.add_argument(
        "--local-model-root",
        type=str,
        default=os.environ.get("HF_HUB_CACHE", "/data2/zzx/data/model/huggingface/hub"),
        help="root directory containing local HF cache entries such as models--org--name",
    )
    ap.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="batch sizes, e.g. '1,2'",
    )
    ap.add_argument(
        "--seq-lens",
        type=str,
        default="128",
        help="sequence lengths, e.g. '128,256,512'",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run models",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="float16" if torch.cuda.is_available() else "float32",
        choices=["float16", "float32", "bfloat16"],
        help="dtype for model weights",
    )
    ap.add_argument(
        "--max-model-params",
        type=int,
        default=3_000_000_000,
        help="max allowed params per model",
    )
    ap.add_argument(
        "--require-sub1b",
        action="store_true",
        help="error if no model <=1B params is included",
    )
    ap.add_argument(
        "--allow-download",
        action="store_true",
        help="allow downloading from HF (default: local files only)",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow custom model code when loading from HF",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="./tmp/llm_gemm_shapes.csv",
        help="output CSV path for unique shapes",
    )
    ap.add_argument(
        "--out-detail",
        type=str,
        default="",
        help="optional detail CSV (model,layer,M,N,K)",
    )
    args = ap.parse_args()

    if AutoModelForCausalLM is None:
        raise SystemExit("transformers not available; install it to use this script")

    model_list: List[str] = []
    model_list.extend(parse_str_list(args.models))
    if args.models_file:
        model_list.extend(read_models_file(args.models_file))
    if args.all_local_models:
        model_list.extend(discover_local_hf_models(args.local_model_root))
    model_list = [m for m in model_list if m]
    model_list = list(dict.fromkeys(model_list))
    if not model_list:
        raise SystemExit("No models provided. Use --models or --models-file.")

    batch_sizes = parse_int_list(args.batch_sizes)
    seq_lens = parse_int_list(args.seq_lens)
    if not batch_sizes or not seq_lens:
        raise SystemExit("batch-sizes and seq-lens must be non-empty")

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    unique_shapes: Set[Tuple[int, int, int]] = set()
    detail_rows: List[Tuple[str, str, int, int, int]] = []

    saw_sub1b = False

    for model_id in model_list:
        print(f"# loading {model_id}")
        local_only = not args.allow_download
        config = AutoConfig.from_pretrained(
            model_id,
            local_files_only=local_only,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval().to(args.device)

        n_params = count_params(model)
        print(f"# params: {n_params}")
        if n_params <= 1_000_000_000:
            saw_sub1b = True
        if n_params > args.max_model_params:
            print(f"# skip (>{args.max_model_params} params)")
            continue

        hooks = []
        layer_names: Dict[int, str] = {}

        def hook_fn(name):
            def _hook(module, inputs, _output):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                if not hasattr(module, "weight"):
                    return
                w = module.weight
                if not isinstance(w, torch.Tensor):
                    return
                mnk = infer_mnk_from_linear(x, w)
                if mnk is None:
                    return
                m, n, k = mnk
                unique_shapes.add((m, n, k))
                detail_rows.append((model_id, name, m, n, k))
            return _hook

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                layer_names[id(module)] = name
            else:
                # GPT2-style Conv1D
                if module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
                    layer_names[id(module)] = name

        with torch.no_grad():
            vocab = getattr(config, "vocab_size", 32000)
            for b in batch_sizes:
                for s in seq_lens:
                    input_ids = torch.randint(0, vocab, (b, s), device=args.device)
                    attention_mask = torch.ones((b, s), device=args.device)
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        for h in hooks:
            h.remove()

        del model
        torch.cuda.empty_cache()

    if args.require_sub1b and not saw_sub1b:
        raise SystemExit("No model <=1B params was included; add a smaller model")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K"])
        for (m, n, k) in sorted(unique_shapes):
            w.writerow([m, n, k])
    print(f"# exported shapes: {args.out_csv} (unique={len(unique_shapes)})")

    if args.out_detail:
        with open(args.out_detail, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "layer", "M", "N", "K"])
            for row in detail_rows:
                w.writerow(row)
        print(f"# exported detail: {args.out_detail}")


if __name__ == "__main__":
    main()
