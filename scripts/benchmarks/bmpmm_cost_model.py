#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "apps" / "common"))

from bmpmm_case_selection import infer_model_filter_from_app_name, infer_prec_from_app_name, selected_cases


FEATURE_ORDER = [
    "bias",
    "full_windows",
    "tail_windows",
    "store_windows",
    "full_load_a_bytes",
    "full_load_w_bytes",
    "tail_load_a_bytes",
    "tail_load_w_bytes",
    "store_pairs",
]

CALIB_TERM_ORDER = [
    "bias",
    "full_windows",
    "tail_windows",
    "store_windows",
    "full_load_bytes",
    "tail_load_bytes",
    "store_pairs",
]

PREC_NAME_TO_ID = {
    "binary": 0,
    "int2": 2,
    "INT2": 2,
    "int4": 3,
    "INT4": 3,
    "0": 0,
    "2": 2,
    "3": 3,
}

PREC_ID_TO_NAME = {
    0: "binary",
    2: "int2",
    3: "int4",
}

APP_TAG_BY_PREC = {
    0: "bmpmm_binary",
    2: "bmpmm_INT2",
    3: "bmpmm_INT4",
}


@dataclass(frozen=True)
class ExecCfg:
    mtile: int
    ntile: int
    ktile: int
    gm: int
    gn: int
    prec: int


@dataclass(frozen=True)
class BenchCase:
    name: str
    scale: str
    model: str
    layer: str
    M: int
    N: int
    K: int
    cfg: ExecCfg


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(x: int, a: int) -> int:
    return ceil_div(x, a) * a


def weight_bits_from_prec(prec: int) -> int:
    if prec == 0:
        return 1
    if prec == 2:
        return 2
    if prec == 3:
        return 4
    raise ValueError(f"unsupported precision id {prec}")


def planes_from_prec(prec: int) -> int:
    if prec == 0:
        return 1
    if prec == 2:
        return 2
    if prec == 3:
        return 4
    raise ValueError(f"unsupported precision id {prec}")


def window_shape(total_len: int, block_len: int, win_idx: int) -> Tuple[int, int]:
    start = win_idx * block_len
    if start >= total_len:
        return start, 0
    return start, min(block_len, total_len - start)


def next_window(a_windows: int, w_windows: int, row_snake: int, cur_a: int, cur_w: int) -> Tuple[bool, int, int]:
    next_a = cur_a
    next_w = cur_w
    valid = False

    if row_snake:
        if (cur_a & 1) == 0:
            if cur_w + 1 < w_windows:
                next_w = cur_w + 1
                valid = True
            elif cur_a + 1 < a_windows:
                next_a = cur_a + 1
                next_w = 0 if w_windows == 0 else (w_windows - 1)
                valid = True
        else:
            if cur_w > 0:
                next_w = cur_w - 1
                valid = True
            elif cur_a + 1 < a_windows:
                next_a = cur_a + 1
                next_w = 0
                valid = True
    else:
        if (cur_w & 1) == 0:
            if cur_a + 1 < a_windows:
                next_a = cur_a + 1
                valid = True
            elif cur_w + 1 < w_windows:
                next_a = 0 if a_windows == 0 else (a_windows - 1)
                next_w = cur_w + 1
                valid = True
        else:
            if cur_a > 0:
                next_a = cur_a - 1
                valid = True
            elif cur_w + 1 < w_windows:
                next_a = 0
                next_w = cur_w + 1
                valid = True

    return valid, next_a, next_w


def pair_from_ord(a_len: int, w_len: int, reuse_a: int, pair_ord: int) -> Tuple[int, int]:
    ord_idx = 0
    if reuse_a:
        for a_pos in range(a_len):
            if (a_pos & 1) == 0:
                for w_pos in range(w_len):
                    if ord_idx == pair_ord:
                        return a_pos, w_pos
                    ord_idx += 1
            else:
                for w_rev in range(w_len):
                    w_pos = w_len - 1 - w_rev
                    if ord_idx == pair_ord:
                        return a_pos, w_pos
                    ord_idx += 1
    else:
        for w_pos in range(w_len):
            if (w_pos & 1) == 0:
                for a_pos in range(a_len):
                    if ord_idx == pair_ord:
                        return a_pos, w_pos
                    ord_idx += 1
            else:
                for a_rev in range(a_len):
                    a_pos = a_len - 1 - a_rev
                    if ord_idx == pair_ord:
                        return a_pos, w_pos
                    ord_idx += 1
    return 0, 0


def select_group_plan(cfg: ExecCfg, mg_len: int, ng_len: int, k_cfg: int) -> Dict[str, int]:
    weight_bits = weight_bits_from_prec(cfg.prec)
    reuse_a = 1 if (cfg.mtile * k_cfg * 8) >= (cfg.ntile * k_cfg * weight_bits) else 0
    total_slots = min(4, mg_len + ng_len)

    best_load_bytes = None
    best_pref = None
    best_same_a = None
    best_same_w = None
    best_a_slots = 1
    best_w_slots = 1
    best_row_snake = 0

    for a_slots in range(1, min(mg_len, total_slots - 1) + 1):
        w_slots = total_slots - a_slots
        if w_slots == 0 or w_slots > ng_len:
            continue

        a_windows = ceil_div(mg_len, a_slots)
        w_windows = ceil_div(ng_len, w_slots)

        for row_snake in (0, 1):
            cur_a = 0
            cur_w = 0
            prev_a = None
            prev_w = None
            load_a = 0
            load_w = 0
            same_a = 0
            same_w = 0
            prev_pair = None

            while True:
                a_start, a_len = window_shape(mg_len, a_slots, cur_a)
                w_start, w_len = window_shape(ng_len, w_slots, cur_w)

                if cur_a != prev_a:
                    load_a += a_len
                if cur_w != prev_w:
                    load_w += w_len

                pair_count = a_len * w_len
                for pair_ord in range(pair_count):
                    a_pos, w_pos = pair_from_ord(a_len, w_len, reuse_a, pair_ord)
                    abs_a = a_start + a_pos
                    abs_w = w_start + w_pos
                    if prev_pair is not None:
                        if abs_a == prev_pair[0]:
                            same_a += 1
                        if abs_w == prev_pair[1]:
                            same_w += 1
                    prev_pair = (abs_a, abs_w)

                prev_a = cur_a
                prev_w = cur_w
                has_next, cur_a, cur_w = next_window(a_windows, w_windows, row_snake, cur_a, cur_w)
                if not has_next:
                    break

            load_bytes = load_a * cfg.mtile * k_cfg + (load_w * cfg.ntile * k_cfg * weight_bits) // 8
            preferred = same_a if reuse_a else same_w
            better = False

            if best_load_bytes is None or load_bytes < best_load_bytes:
                better = True
            elif load_bytes == best_load_bytes and preferred > best_pref:
                better = True
            elif load_bytes == best_load_bytes and preferred == best_pref and same_a > best_same_a:
                better = True
            elif load_bytes == best_load_bytes and preferred == best_pref and same_a == best_same_a and same_w > best_same_w:
                better = True
            elif load_bytes == best_load_bytes and preferred == best_pref and same_a == best_same_a and same_w == best_same_w and a_slots > best_a_slots:
                better = True
            elif load_bytes == best_load_bytes and preferred == best_pref and same_a == best_same_a and same_w == best_same_w and a_slots == best_a_slots and row_snake < best_row_snake:
                better = True

            if better:
                best_load_bytes = load_bytes
                best_pref = preferred
                best_same_a = same_a
                best_same_w = same_w
                best_a_slots = a_slots
                best_w_slots = w_slots
                best_row_snake = row_snake

    return {
        "a_slots": best_a_slots,
        "w_slots": best_w_slots,
        "a_windows": ceil_div(mg_len, best_a_slots),
        "w_windows": ceil_div(ng_len, best_w_slots),
        "row_snake": best_row_snake,
        "reuse_a": reuse_a,
    }


def collect_window_visit_stats(mg_len: int, ng_len: int, plan: Dict[str, int]) -> Dict[str, int]:
    cur_a = 0
    cur_w = 0
    prev_a = None
    prev_w = None

    stats = {
        "window_count": 0,
        "load_a_count": 0,
        "load_w_count": 0,
        "pair_count": 0,
    }

    while True:
        _, a_len = window_shape(mg_len, plan["a_slots"], cur_a)
        _, w_len = window_shape(ng_len, plan["w_slots"], cur_w)
        stats["window_count"] += 1
        stats["pair_count"] += a_len * w_len

        if cur_a != prev_a:
            stats["load_a_count"] += a_len
        if cur_w != prev_w:
            stats["load_w_count"] += w_len

        prev_a = cur_a
        prev_w = cur_w
        has_next, cur_a, cur_w = next_window(plan["a_windows"], plan["w_windows"], plan["row_snake"], cur_a, cur_w)
        if not has_next:
            break

    return stats


def collect_template_stats(case: BenchCase) -> Dict[str, int]:
    cfg = case.cfg
    m_tiles = ceil_div(case.M, cfg.mtile)
    n_tiles = ceil_div(case.N, cfg.ntile)
    full_tiles = case.K // cfg.ktile
    tail_len = case.K % cfg.ktile

    stats = {
        "full_windows": 0,
        "full_load_a": 0,
        "full_load_w": 0,
        "full_compute": 0,
        "tail_present": 0,
        "tail_k_cfg": 0,
        "tail_windows": 0,
        "tail_load_a": 0,
        "tail_load_w": 0,
        "tail_compute": 0,
        "store_windows": 0,
        "store_count": 0,
    }

    if tail_len:
        stats["tail_present"] = 1
        stats["tail_k_cfg"] = align_up(tail_len, 8)

    for mg in range(0, m_tiles, cfg.gm):
        mg_len = min(cfg.gm, m_tiles - mg)
        for ng in range(0, n_tiles, cfg.gn):
            ng_len = min(cfg.gn, n_tiles - ng)
            if mg_len == 0 or ng_len == 0:
                continue

            plan = select_group_plan(cfg, mg_len, ng_len, cfg.ktile)
            visit = collect_window_visit_stats(mg_len, ng_len, plan)

            if full_tiles:
                stats["full_windows"] += visit["window_count"] * full_tiles
                stats["full_load_a"] += visit["load_a_count"] * full_tiles
                stats["full_load_w"] += visit["load_w_count"] * full_tiles
                stats["full_compute"] += visit["pair_count"] * full_tiles

            if stats["tail_present"]:
                stats["tail_windows"] += visit["window_count"]
                stats["tail_load_a"] += visit["load_a_count"]
                stats["tail_load_w"] += visit["load_w_count"]
                stats["tail_compute"] += visit["pair_count"]

            stats["store_windows"] += visit["window_count"]
            stats["store_count"] += visit["pair_count"]

    return stats


def analytic_compute_cycles(case: BenchCase, stats: Dict[str, int]) -> int:
    cfg = case.cfg
    planes = planes_from_prec(cfg.prec)
    m_blocks = ceil_div(cfg.mtile, 8)
    n_blocks = ceil_div(cfg.ntile, 16)
    phys_blocks = m_blocks * n_blocks

    total = 0
    if stats["full_compute"]:
        full_sa_cycles = 1 + (cfg.ktile // 8) * planes + 2
        total += stats["full_compute"] * phys_blocks * full_sa_cycles

    if stats["tail_present"] and stats["tail_compute"]:
        tail_sa_cycles = 1 + (stats["tail_k_cfg"] // 8) * planes + 2
        total += stats["tail_compute"] * phys_blocks * tail_sa_cycles

    return total


def build_features(case: BenchCase) -> Dict[str, float]:
    stats = collect_template_stats(case)
    cfg = case.cfg
    weight_bits = weight_bits_from_prec(cfg.prec)

    features = {
        "bias": 1.0,
        "full_windows": float(stats["full_windows"]),
        "tail_windows": float(stats["tail_windows"]),
        "store_windows": float(stats["store_windows"]),
        "full_load_a_bytes": float(stats["full_load_a"] * cfg.mtile * cfg.ktile),
        "full_load_w_bytes": float((stats["full_load_w"] * cfg.ntile * cfg.ktile * weight_bits) // 8),
        "tail_load_a_bytes": float(stats["tail_load_a"] * cfg.mtile * stats["tail_k_cfg"]),
        "tail_load_w_bytes": float((stats["tail_load_w"] * cfg.ntile * stats["tail_k_cfg"] * weight_bits) // 8),
        "store_pairs": float(stats["store_count"]),
        "exact_compute_cycles": float(analytic_compute_cycles(case, stats)),
    }
    return features


def build_calibration_terms(case: BenchCase) -> Dict[str, float]:
    features = build_features(case)
    return {
        "bias": 1.0,
        "full_windows": features["full_windows"],
        "tail_windows": features["tail_windows"],
        "store_windows": features["store_windows"],
        "full_load_bytes": features["full_load_a_bytes"] + features["full_load_w_bytes"],
        "tail_load_bytes": features["tail_load_a_bytes"] + features["tail_load_w_bytes"],
        "store_pairs": features["store_pairs"],
    }


def case_from_dict(case_dict: Dict[str, object]) -> BenchCase:
    cfg_dict = case_dict["cfg"]
    cfg = ExecCfg(
        mtile=int(cfg_dict["mtile"]),
        ntile=int(cfg_dict["ntile"]),
        ktile=int(cfg_dict["ktile"]),
        gm=int(cfg_dict["gm"]),
        gn=int(cfg_dict["gn"]),
        prec=int(cfg_dict["prec"]),
    )
    return BenchCase(
        name=str(case_dict["name"]),
        scale=str(case_dict["scale"]),
        model=str(case_dict["model"]),
        layer=str(case_dict["layer"]),
        M=int(case_dict["M"]),
        N=int(case_dict["N"]),
        K=int(case_dict["K"]),
        cfg=cfg,
    )


def load_app_cases(app_name: str) -> Tuple[int, str, List[BenchCase]]:
    prec = infer_prec_from_app_name(app_name)
    model_filter = infer_model_filter_from_app_name(app_name)
    if model_filter is None:
        raise ValueError(f"cannot infer model suffix from app name: {app_name}")
    cases = [case_from_dict(case_dict) for case_dict in selected_cases(prec, model_filter=model_filter)]
    return prec, model_filter, cases


def cfg_key(cfg: ExecCfg) -> str:
    return f"mt{cfg.mtile}_nt{cfg.ntile}_kt{cfg.ktile}_gm{cfg.gm}_gn{cfg.gn}"


def precision_name(value: str | int) -> str:
    if isinstance(value, int):
        return PREC_ID_TO_NAME[value]
    if value not in PREC_NAME_TO_ID:
        raise ValueError(f"unknown precision {value}")
    return PREC_ID_TO_NAME[PREC_NAME_TO_ID[value]]


def load_coefficients(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    feature_order = data.get("feature_order", FEATURE_ORDER)
    if feature_order != FEATURE_ORDER:
        raise ValueError(f"unexpected feature order in {path}")
    return data


def select_precision_coeffs(coeffs: Dict[str, object], prec_name: str, cfg: ExecCfg) -> Dict[str, float]:
    precision_table = coeffs.get("precisions", {})
    if prec_name not in precision_table:
        raise KeyError(f"missing precision entry {prec_name}")
    prec_entry = precision_table[prec_name]
    cfg_overrides = prec_entry.get("cfg_overrides", {})
    key = cfg_key(cfg)
    if key in cfg_overrides:
        return {name: float(cfg_overrides[key].get(name, 0.0)) for name in FEATURE_ORDER}
    base = prec_entry.get("default", {})
    return {name: float(base.get(name, 0.0)) for name in FEATURE_ORDER}


def predict_case_cycles(case: BenchCase, coeffs: Dict[str, object]) -> int:
    prec_name = precision_name(case.cfg.prec)
    weights = select_precision_coeffs(coeffs, prec_name, case.cfg)
    features = build_features(case)
    total = features["exact_compute_cycles"]
    for name in FEATURE_ORDER:
        total += weights[name] * features[name]
    return int(round(total))


def emit_app_log(app_name: str, coeffs: Dict[str, object]) -> int:
    prec, _, cases = load_app_cases(app_name)
    tag = APP_TAG_BY_PREC[prec]
    prec_label = precision_name(prec)

    print(f"[{tag}] precision={prec_label}")
    current_model = None
    model_cycles = 0
    runtime_cache: Dict[Tuple[int, int, int, str], Tuple[int, int]] = {}

    for index, case in enumerate(cases, start=1):
        if current_model != case.model:
            if current_model is not None:
                print(f"[{tag}] model_total model={current_model} bmpmm_cycles={model_cycles}")
            current_model = case.model
            model_cycles = 0
            print()
            print("============================================================")
            print(f"[{tag}] model={case.model} scale={case.scale}")
            print("============================================================")

        print()
        print("------------------------------------------------------------")
        print(
            f"[{tag}] case{index} layer={case.layer} shape=({case.M},{case.N},{case.K}), "
            f"cfg=(mt={case.cfg.mtile},nt={case.cfg.ntile},kt={case.cfg.ktile},gm={case.cfg.gm},gn={case.cfg.gn},p={case.cfg.prec})"
        )

        key = (case.M, case.N, case.K, cfg_key(case.cfg))
        if key in runtime_cache:
            runtime, first_case_index = runtime_cache[key]
            model_cycles += runtime
            print(f"[{tag}] duplicate_shape_skip case{index} reuse_case{first_case_index}")
            print(f"[{tag}] bmpmm_runtime={runtime}")
            continue

        runtime = predict_case_cycles(case, coeffs)
        runtime_cache[key] = (runtime, index)
        model_cycles += runtime
        print(f"[{tag}] bmpmm_runtime={runtime}")

    if current_model is not None:
        print(f"[{tag}] model_total model={current_model} bmpmm_cycles={model_cycles}")

    return 0


def parse_samples_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"precision", "M", "N", "K", "mtile", "ntile", "ktile", "gm", "gn", "measured_total"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns in {path}: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    return rows


def parse_samples_csvs(paths: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in paths:
        rows.extend(parse_samples_csv(path))
    return rows


FAST_ERROR_SHAPE_RE = re.compile(
    r"^\[fast_error_check\]\[(bmpmm_binary|bmpmm_int2|bmpmm_int4)\]\[(?P<tag>[^\]]+)\] "
    r"(?:shape=\((?P<M>\d+),(?P<N>\d+),(?P<K>\d+)\)"
    r"(?: cfg=\(mt=(?P<mt>\d+),nt=(?P<nt>\d+),kt=(?P<kt>\d+),gm=(?P<gm>\d+),gn=(?P<gn>\d+),p=(?P<p>\d+)\))?"
    r"(?: .*)?"
    r"|strict_total=(?P<strict_total>\d+) strict_compute=(?P<strict_compute>\d+) "
    r"strict_warm_total=(?P<strict_warm_total>\d+)(?: strict_warm_compute=(?P<strict_warm_compute>\d+))?"
    r"(?: strict_same_buf_total=(?P<strict_same_buf_total>\d+))? fast_total=(?P<fast_total>\d+) fast_compute=(?P<fast_compute>\d+).*)$"
)


def default_cfg_for_fast_error_tag(tag: str) -> ExecCfg:
    if tag == "bmpmm_int2":
        return ExecCfg(mtile=8, ntile=16, ktile=64, gm=2, gn=1, prec=2)
    if tag == "bmpmm_int4":
        return ExecCfg(mtile=8, ntile=16, ktile=64, gm=2, gn=1, prec=3)
    raise ValueError(f"no default cfg for fast error tag {tag}")


def parse_fast_error_log(path: Path) -> List[Dict[str, object]]:
    records: Dict[Tuple[str, str], Dict[str, object]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = FAST_ERROR_SHAPE_RE.match(line)
            if not match:
                continue

            tag = line.split("]")[1][1:]
            case_tag = match.group("tag")
            key = (tag, case_tag)
            record = records.setdefault(
                key,
                {
                    "name": case_tag,
                    "scale": "fast_error_check",
                    "model": "fast_error_check",
                    "layer": case_tag,
                },
            )

            if match.group("M") is not None:
                record["M"] = int(match.group("M"))
                record["N"] = int(match.group("N"))
                record["K"] = int(match.group("K"))
                if match.group("mt") is not None:
                    record["mtile"] = int(match.group("mt"))
                    record["ntile"] = int(match.group("nt"))
                    record["ktile"] = int(match.group("kt"))
                    record["gm"] = int(match.group("gm"))
                    record["gn"] = int(match.group("gn"))
                    record["precision"] = precision_name(int(match.group("p")))
                else:
                    cfg = default_cfg_for_fast_error_tag(tag)
                    record["mtile"] = cfg.mtile
                    record["ntile"] = cfg.ntile
                    record["ktile"] = cfg.ktile
                    record["gm"] = cfg.gm
                    record["gn"] = cfg.gn
                    record["precision"] = precision_name(cfg.prec)
                continue

            if match.group("strict_total") is not None:
                if tag == "bmpmm_binary":
                    record["measured_total"] = int(match.group("strict_warm_total"))
                else:
                    strict_same_buf = match.group("strict_same_buf_total")
                    if strict_same_buf is None:
                        raise ValueError(f"missing strict_same_buf_total in {path}: {line}")
                    record["measured_total"] = int(strict_same_buf)

    required = {"precision", "M", "N", "K", "mtile", "ntile", "ktile", "gm", "gn", "measured_total"}
    parsed: List[Dict[str, object]] = []
    for key in sorted(records):
        record = records[key]
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"incomplete fast_error_check record {key} in {path}: missing {missing}")
        parsed.append(record)
    return parsed


def case_from_sample_row(row: Dict[str, object]) -> BenchCase:
    prec = PREC_NAME_TO_ID[str(row["precision"])]
    return BenchCase(
        name=str(row.get("name", "sample")),
        scale=str(row.get("scale", "sample")),
        model=str(row.get("model", "sample")),
        layer=str(row.get("layer", "sample")),
        M=int(row["M"]),
        N=int(row["N"]),
        K=int(row["K"]),
        cfg=ExecCfg(
            mtile=int(row["mtile"]),
            ntile=int(row["ntile"]),
            ktile=int(row["ktile"]),
            gm=int(row["gm"]),
            gn=int(row["gn"]),
            prec=prec,
        ),
    )


def sample_feature_record(row: Dict[str, object]) -> Dict[str, object]:
    case = case_from_sample_row(row)
    features = build_features(case)
    terms = build_calibration_terms(case)
    return {
        "row": row,
        "case": case,
        "features": features,
        "terms": terms,
        "target_delta": float(row["measured_total"]) - features["exact_compute_cycles"],
        "measured_total": float(row["measured_total"]),
    }


def fit_term_subset(records: List[Dict[str, object]], term_names: List[str]) -> np.ndarray:
    x = np.asarray([[record["terms"][name] for name in term_names] for record in records], dtype=np.float64)
    y = np.asarray([record["target_delta"] for record in records], dtype=np.float64)
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coeffs


def predict_with_term_subset(record: Dict[str, object], term_names: List[str], coeffs: np.ndarray) -> float:
    total = record["features"]["exact_compute_cycles"]
    for name, coeff in zip(term_names, coeffs):
        total += float(coeff) * float(record["terms"][name])
    return total


def score_subset(records: List[Dict[str, object]], term_names: List[str]) -> Dict[str, float] | None:
    coeffs = fit_term_subset(records, term_names)
    if np.any(coeffs < -1e-9):
        return None

    rel_errs: List[float] = []
    for idx in range(len(records)):
        train = [record for j, record in enumerate(records) if j != idx]
        if not train:
            continue
        loo_coeffs = fit_term_subset(train, term_names)
        if np.any(loo_coeffs < -1e-9):
            return None
        pred = predict_with_term_subset(records[idx], term_names, loo_coeffs)
        exact = records[idx]["measured_total"]
        rel_errs.append(abs(pred - exact) / exact if exact else 0.0)

    if not rel_errs:
        return None

    return {
        "max_rel_err": max(rel_errs),
        "avg_rel_err": sum(rel_errs) / len(rel_errs),
    }


def expand_term_weights(term_names: List[str], coeffs: np.ndarray) -> Dict[str, float]:
    weights = {name: 0.0 for name in FEATURE_ORDER}
    for name, coeff in zip(term_names, coeffs):
        value = float(coeff)
        if name in FEATURE_ORDER:
            weights[name] = value
        elif name == "full_load_bytes":
            weights["full_load_a_bytes"] = value
            weights["full_load_w_bytes"] = value
        elif name == "tail_load_bytes":
            weights["tail_load_a_bytes"] = value
            weights["tail_load_w_bytes"] = value
        else:
            raise ValueError(f"unsupported calibration term {name}")
    return weights


def select_term_subset(records: List[Dict[str, object]]) -> Tuple[List[str], np.ndarray, Dict[str, float]]:
    extra_terms = [name for name in CALIB_TERM_ORDER if name != "bias"]
    # Keep the calibration model low-rank on tiny datasets, but allow one more
    # physical term once we have at least 7 strict samples for a precision.
    max_extra_terms = 3 if len(records) >= 7 else 2
    max_extra_terms = min(max_extra_terms, len(extra_terms), max(0, len(records) - 2))
    best_term_names: List[str] | None = None
    best_score: Dict[str, float] | None = None

    for extra_count in range(0, max_extra_terms + 1):
        for combo in itertools.combinations(extra_terms, extra_count):
            term_names = ["bias", *combo]
            score = score_subset(records, term_names)
            if score is None:
                continue
            if best_score is None:
                best_term_names = term_names
                best_score = score
                continue
            if score["max_rel_err"] < best_score["max_rel_err"] - 1e-12:
                best_term_names = term_names
                best_score = score
                continue
            if abs(score["max_rel_err"] - best_score["max_rel_err"]) <= 1e-12 and score["avg_rel_err"] < best_score["avg_rel_err"] - 1e-12:
                best_term_names = term_names
                best_score = score
                continue
            if (
                abs(score["max_rel_err"] - best_score["max_rel_err"]) <= 1e-12
                and abs(score["avg_rel_err"] - best_score["avg_rel_err"]) <= 1e-12
                and len(term_names) < len(best_term_names or [])
            ):
                best_term_names = term_names
                best_score = score

    if best_term_names is None or best_score is None:
        best_term_names = ["bias", "full_windows"]
        best_coeffs = fit_term_subset(records, best_term_names)
        best_score = score_subset(records, best_term_names) or {"max_rel_err": math.inf, "avg_rel_err": math.inf}
    else:
        best_coeffs = fit_term_subset(records, best_term_names)

    return best_term_names, best_coeffs, best_score


def calibrate_precision(rows: Iterable[Dict[str, object]]) -> Tuple[Dict[str, float], Dict[str, object]]:
    records = [sample_feature_record(row) for row in rows]
    term_names, coeffs, score = select_term_subset(records)
    return expand_term_weights(term_names, coeffs), {
        "selected_terms": term_names,
        "loo_max_rel_err_pct": 100.0 * score["max_rel_err"],
        "loo_avg_rel_err_pct": 100.0 * score["avg_rel_err"],
        "sample_count": len(records),
    }


def cmd_calibrate(args: argparse.Namespace) -> int:
    rows = parse_samples_csvs(Path(path) for path in args.samples)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        prec_name = precision_name(str(row["precision"]))
        grouped.setdefault(prec_name, []).append(row)

    output = {
        "version": 1,
        "feature_order": FEATURE_ORDER,
        "precisions": {},
    }
    for prec_name, prec_rows in grouped.items():
        weights, meta = calibrate_precision(prec_rows)
        output["precisions"][prec_name] = {
            "default": weights,
            "cfg_overrides": {},
            "meta": meta,
        }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def cmd_dump_features(args: argparse.Namespace) -> int:
    _, _, cases = load_app_cases(args.app)
    for case in cases:
        features = build_features(case)
        record = {
            "case": case.name,
            "model": case.model,
            "layer": case.layer,
            "M": case.M,
            "N": case.N,
            "K": case.K,
            "cfg": {
                "mtile": case.cfg.mtile,
                "ntile": case.cfg.ntile,
                "ktile": case.cfg.ktile,
                "gm": case.cfg.gm,
                "gn": case.cfg.gn,
                "prec": case.cfg.prec,
            },
            "features": features,
        }
        print(json.dumps(record, sort_keys=True))
    return 0


def cmd_predict_app(args: argparse.Namespace) -> int:
    coeffs = load_coefficients(Path(args.coeffs))
    return emit_app_log(args.app, coeffs)


def cmd_parse_fast_error_log(args: argparse.Namespace) -> int:
    rows = parse_fast_error_log(Path(args.log))
    out_path = Path(args.out)
    fieldnames = [
        "precision",
        "name",
        "scale",
        "model",
        "layer",
        "M",
        "N",
        "K",
        "mtile",
        "ntile",
        "ktile",
        "gm",
        "gn",
        "measured_total",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
    print(f"wrote {out_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    rows = parse_samples_csvs(Path(path) for path in args.samples)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        prec_name = precision_name(str(row["precision"]))
        grouped.setdefault(prec_name, []).append(row)

    for prec_name in sorted(grouped):
        weights, meta = calibrate_precision(grouped[prec_name])
        print(json.dumps({
            "precision": prec_name,
            "selected_terms": meta["selected_terms"],
            "loo_max_rel_err_pct": round(meta["loo_max_rel_err_pct"], 4),
            "loo_avg_rel_err_pct": round(meta["loo_avg_rel_err_pct"], 4),
            "sample_count": meta["sample_count"],
            "weights": weights,
        }, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python cost model for BMPMM app runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    predict = sub.add_parser("predict-app", help="emit app-style bmpmm runtime logs from a coefficient file")
    predict.add_argument("--app", required=True, help="app name such as bmpmm_binary_gemma3_270m")
    predict.add_argument("--coeffs", required=True, help="json coefficient file")
    predict.set_defaults(func=cmd_predict_app)

    dump_features = sub.add_parser("dump-features", help="dump per-case template features for one bmpmm app")
    dump_features.add_argument("--app", required=True, help="app name such as bmpmm_INT4_qwen25_05b")
    dump_features.set_defaults(func=cmd_dump_features)

    calibrate = sub.add_parser("calibrate", help="fit per-precision coefficients from small strict samples")
    calibrate.add_argument("--samples", required=True, nargs="+", help="csv files with precision/M/N/K/cfg/measured_total columns")
    calibrate.add_argument("--out", required=True, help="output json path")
    calibrate.set_defaults(func=cmd_calibrate)

    parse_fast = sub.add_parser("parse-fast-error-log", help="extract calibration samples from fast_error_check output")
    parse_fast.add_argument("--log", required=True, help="fast_error_check log path")
    parse_fast.add_argument("--out", required=True, help="output csv path")
    parse_fast.set_defaults(func=cmd_parse_fast_error_log)

    validate = sub.add_parser("validate", help="fit each precision and print leave-one-out error summary")
    validate.add_argument("--samples", required=True, nargs="+", help="csv files with precision/M/N/K/cfg/measured_total columns")
    validate.set_defaults(func=cmd_validate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
