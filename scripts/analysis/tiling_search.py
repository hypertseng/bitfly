#!/usr/bin/env python3
"""Per-shape search for BMPMM execution parameters.

This version follows the latest constraints:
- No global anchor search; each shape is searched independently.
- Hardware granularity is fixed in code.
- The search range allows g to reach 8 by default.
- Config legality uses capacity constraints instead of exact-fill constraints.
- Compute model follows the implemented BSPA wavefront latency.
- The software-side scheduling model follows src/apps/common/bmpmm_operator_template.h.
- BMPMM/BMPSE are modeled as single-pair instructions; gm/gn only affect the
  group traversal and the resident-tile reuse policy.
- The resident working set is modeled as four VRF bank-pair slots. The search
  automatically chooses how many resident slots are spent on A vs. W tiles for
  each group shape, then traverses the group in a canonical windowed snake
  order under that policy.
- g=(gm*gn) is evaluated against the implemented legality rules.
- K-tile is capped by the currently generated software path limit.
- Store overlap is evaluated as background drain during memory-free
  compute/control time, with only the remaining tail exposed.
- Search granularity follows the conservative executable backend path: mtile is
  modeled at 8-row granularity and ntile at 16-column granularity.

Notation:
- Shape: (M, N, K)
- Config: (g, gm, gn, mtile, ntile)
- Constraint: mtile * ntile * g * 16 <= buffer_bits
- Reuse decomposition: g = gm * gn
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# Fixed hardware constants.
R_ROWS = 2
C_COLS = 2
N_VECTOR = 8
M_TILE_MIN = 8
M_TILE_STEP = 8
N_TILE_MIN = 16
N_TILE_STEP = 16
G_MAX_HW = 8
BUFFER_ELEM_BITS = 16
ACT_BITS = 8
ARRAY_M = 8
ARRAY_N = 16
K_STEP = 8
BANK_CAP_BITS = 8192
# Software-search cap for ktile. With dispatch-table enforcement enabled, the
# checked-in bmpcfg table may further reduce the executable k values.
K_TILE_MAX_SW = 256
STORE_QUEUE_DRAIN_BYTES_PER_CYCLE = 16.0
BMPCFG_DISPATCH_TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "src",
    "apps",
    "common",
    "bmpcfg_dispatch_table.inc",
)

# Software-template control costs.
EMIT_CFG_CYCLES = 1.0
# One BMPSE issue on the software side. Internal 8x16 block traversal happens in
# the BMPU/VSTU path, not in software.
STORE_BLOCK_CTRL_CYCLES = 1.0


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class Config:
    g: int
    gm: int
    gn: int
    mtile: int
    ntile: int


@dataclass
class ScoreDetail:
    cycles_total: float
    cycles_compute: float
    cycles_memory: float
    cycles_control: float
    cycles_exposed_load: float
    cycles_exposed_store: float
    hidden_load_cycles: float
    hidden_store_cycles: float
    hidden_load_share: float
    hidden_store_share: float
    hidden_memory_share: float
    util: float
    group_g: int
    preferred_reuse_transitions: int
    pair_same_a_transitions: int
    pair_same_w_transitions: int
    emit_cfg_count: int
    load_a_count: int
    load_w_count: int
    store_instr_count: int
    resident_a_slots: int
    resident_w_slots: int
    resident_row_snake: int
    ktile: int
    k_iters: int
    reuse_a: int


@dataclass
class GroupExecutionPlan:
    pipeline_cycles: float
    compute_cycles: float
    load_cycles: float
    load_bytes: float
    control_cycles: float
    hidden_load_cycles: float
    window_count: int
    emit_cfg_count: int
    load_a_count: int
    load_w_count: int
    preferred_reuse_transitions: int
    pair_same_a_transitions: int
    pair_same_w_transitions: int
    resident_a_slots: int
    resident_w_slots: int
    resident_row_snake: int


_WORKER_CONFIGS: Sequence[Config] = ()
_WORKER_PREC_BITS: int = 4
_WORKER_OUT_BITS: int = 16
_WORKER_BW_BYTES_PER_CYCLE: float = 16.0
_WORKER_SUPPORTED_KEYS: set[tuple[int, int, int, int, int, int]] = set()
_WORKER_GROUP_LOAD_MODEL: str = "resident_pair"


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_down(x: int, a: int) -> int:
    return (x // a) * a


def align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def prec_to_planes(prec_bits: int) -> int:
    if prec_bits in (1, 2, 4):
        return prec_bits
    return max(1, prec_bits)


def prec_bits_to_code(prec_bits: int) -> int:
    if prec_bits == 1:
        return 0
    if prec_bits == 2:
        return 2
    if prec_bits == 4:
        return 3
    raise ValueError(f"Unsupported precision bits for current BMPU path: {prec_bits}")


def decode_bmpcfg_key(imm25: int) -> tuple[int, int, int, int, int, int]:
    prec = (imm25 >> 22) & 0x7
    k = (((imm25 >> 17) & 0x1F) + 1) * 8
    mtile = (((imm25 >> 13) & 0xF) * 4) + 8
    ntile = (((imm25 >> 10) & 0x7) * 16) + 16
    gm = ((imm25 >> 7) & 0x7) + 1
    gn = ((imm25 >> 4) & 0x7) + 1
    return prec, k, mtile, ntile, gm, gn


def load_supported_bmpcfg_keys(path: str) -> set[tuple[int, int, int, int, int, int]]:
    if not os.path.exists(path):
        return set()
    out: set[tuple[int, int, int, int, int, int]] = set()
    pat = re.compile(r"\{0x([0-9A-Fa-f]+)u,\s*bmpcfg_emit_stub_")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            out.add(decode_bmpcfg_key(int(m.group(1), 16)))
    return out


def load_shapes(path: str) -> List[Shape]:
    out: List[Shape] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("Input CSV is missing header")
        names = {x.strip().lower(): x for x in reader.fieldnames if x}
        for col in ("m", "n", "k"):
            if col not in names:
                raise SystemExit("Input CSV must contain columns: M,N,K")
        for row in reader:
            out.append(
                Shape(
                    m=int(row[names["m"]]),
                    n=int(row[names["n"]]),
                    k=int(row[names["k"]]),
                )
            )
    if not out:
        raise SystemExit("No shape rows found in input CSV")
    return out


def gm_gn_triplets(g_max: int, allowed_g: Sequence[int] | None = None) -> List[Tuple[int, int, int]]:
    allowed = set(allowed_g) if allowed_g is not None else None
    out: List[Tuple[int, int, int]] = []
    for gm in range(1, g_max + 1):
        for gn in range(1, g_max + 1):
            g = gm * gn
            if g > g_max:
                continue
            if allowed is not None and g not in allowed:
                continue
            out.append((gm, gn, g))
    return out


def legal_configs(buffer_bits: int) -> List[Config]:
    # Capacity-legal: mtile * ntile * g * 16 <= buffer_bits
    if buffer_bits <= 0 or buffer_bits % BUFFER_ELEM_BITS != 0:
        return []

    target_area = buffer_bits // BUFFER_ELEM_BITS
    g_max_buffer = max(1, target_area // max(1, M_TILE_MIN * N_TILE_MIN))
    g_max = min(g_max_buffer, G_MAX_HW)

    out: List[Config] = []
    for gm, gn, g in gm_gn_triplets(g_max):
        tile_area_max = target_area // g
        mt = M_TILE_MIN
        while mt <= tile_area_max:
            nt = N_TILE_MIN
            while nt <= tile_area_max:
                if mt * nt <= tile_area_max:
                    out.append(Config(g=g, gm=gm, gn=gn, mtile=mt, ntile=nt))
                nt += N_TILE_STEP
            mt += M_TILE_STEP

    # Remove duplicates while preserving deterministic order.
    seen: set[Config] = set()
    uniq: List[Config] = []
    for c in out:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def auto_m_max(shapes: Sequence[Shape], buffer_bits: int) -> int:
    shape_cap = max(s.m for s in shapes)
    buffer_cap = max(M_TILE_STEP, buffer_bits // max(1, N_TILE_MIN * BUFFER_ELEM_BITS))
    return max(M_TILE_STEP, align_down(min(shape_cap, buffer_cap), M_TILE_STEP))


def auto_n_max(shapes: Sequence[Shape], buffer_bits: int) -> int:
    shape_cap = max(s.n for s in shapes)
    buffer_cap = max(N_TILE_STEP, buffer_bits // max(1, M_TILE_MIN * BUFFER_ELEM_BITS))
    return max(N_TILE_STEP, align_down(min(shape_cap, buffer_cap), N_TILE_STEP))


def auto_g_candidates(buffer_bits: int) -> List[int]:
    g_max_buffer = max(1, buffer_bits // max(1, M_TILE_MIN * N_TILE_MIN * BUFFER_ELEM_BITS))
    g_max = min(g_max_buffer, G_MAX_HW)
    vals: set[int] = set()
    for g in range(1, g_max + 1):
        # Under hardware cap (<=8), keep dense enumeration.
        vals.add(g)
    vals.add(1)
    vals.add(g_max)
    return sorted(vals)


def auto_tile_candidates(max_value: int, min_value: int, step: int) -> List[int]:
    vals: set[int] = set()
    v = min_value
    while v <= max_value:
        vals.add(v)
        if v < 128:
            v += step
        else:
            grown = int(math.ceil(v * 1.5 / step) * step)
            if grown <= v:
                grown = v + step
            v = grown
    vals.add(max_value)
    return sorted(x for x in vals if min_value <= x <= max_value and x % step == 0)


def iter_configs(g_list: Sequence[int], m_tiles: Sequence[int], n_tiles: Sequence[int], buffer_bits: int) -> Iterable[Config]:
    g_cap = max(g_list) if g_list else 0
    for gm, gn, g in gm_gn_triplets(g_cap, allowed_g=g_list):
        for mt in m_tiles:
            for nt in n_tiles:
                if mt * nt * g * BUFFER_ELEM_BITS <= buffer_bits:
                    yield Config(g=g, gm=gm, gn=gn, mtile=mt, ntile=nt)


def max_feasible_ktile(shape_k: int, mtile: int, ntile: int, prec_bits: int) -> int | None:
    # Activation: mtile * ktile * ACT_BITS <= BANK_CAP_BITS
    k_cap_act = BANK_CAP_BITS // max(1, mtile * ACT_BITS)
    # Weight: ntile * ktile * prec <= 8192 bits
    k_cap_wgt = BANK_CAP_BITS // max(1, ntile * prec_bits)
    k_cap = min(shape_k, K_TILE_MAX_SW, k_cap_act, k_cap_wgt)
    k_cap = align_down(k_cap, 8)
    if k_cap < 8:
        return None
    return k_cap


def prune_configs_for_shape(shape: Shape, configs: Sequence[Config]) -> List[Config]:
    # Do not evaluate mtile/ntile larger than one padded tile covering this shape side.
    m_cap = max(M_TILE_MIN, align_up(shape.m, M_TILE_STEP))
    n_cap = max(N_TILE_MIN, align_up(shape.n, N_TILE_STEP))
    out: List[Config] = []
    for c in configs:
        if c.mtile > m_cap or c.ntile > n_cap:
            continue
        out.append(c)
    return out


def single_compute_cycles(mtile: int, ntile: int, k_block: int, prec_bits: int) -> float:
    # Hardware-aligned compute model from sa.sv:
    # T_bspa = (ROWS - 1) + ceil(k/8) * planes + COLS
    # T_compute = ceil(m/8) * ceil(n/16) * T_bspa
    m_steps = ceil_div(mtile, ARRAY_M)
    n_steps = ceil_div(ntile, ARRAY_N)
    k_steps = ceil_div(k_block, K_STEP)
    planes = prec_to_planes(prec_bits)
    t_bspa = (R_ROWS - 1) + k_steps * planes + C_COLS
    return float(m_steps * n_steps * t_bspa)


def template_reuse_a(mtile: int, ntile: int, prec_bits: int) -> bool:
    act_bits = mtile * ACT_BITS
    wgt_bits = ntile * prec_bits
    return act_bits >= wgt_bits


def simulate_load_compute_pipeline(prep_cycles: Sequence[float], compute_cycles: Sequence[float]) -> Tuple[float, float]:
    """Two-stage pipeline matching the common operator template.

    The software template issues, for each (pair, k-tile), one control+load stage
    followed by one compute stage. Successive load stages may overlap with the
    previous compute stage.
    """
    if len(prep_cycles) != len(compute_cycles):
        raise ValueError("prep_cycles and compute_cycles must have identical lengths")
    if not prep_cycles:
        return 0.0, 0.0

    load_finish = 0.0
    compute_finish = 0.0
    total_prep = 0.0
    total_compute = 0.0

    for prep, comp in zip(prep_cycles, compute_cycles):
        load_finish += prep
        compute_start = max(compute_finish, load_finish)
        compute_finish = compute_start + comp
        total_prep += prep
        total_compute += comp

    makespan = max(load_finish, compute_finish)
    exposed_prep = max(0.0, makespan - total_compute)
    hidden_prep = max(0.0, total_prep - exposed_prep)
    return makespan, hidden_prep


def score_store_overlap(stages: Sequence[Tuple[float, float, float, float]], *, store_bw_bytes_per_cycle: float) -> Tuple[float, float]:
    pending_store_bytes = 0.0
    total_issue_cycles = 0.0

    for pipeline_cycles, load_cycles, store_bytes, store_ctrl_cycles in stages:
        # Pair pipeline can only hide older store traffic during the non-load
        # portion because loads and stores share the same modeled bandwidth.
        store_drain_cycles = max(0.0, pipeline_cycles - load_cycles)
        pending_store_bytes = max(0.0, pending_store_bytes - store_drain_cycles * store_bw_bytes_per_cycle)
        total_issue_cycles += pipeline_cycles

        # Store data becomes pending after the pair's compute completes and BMPSE
        # issue starts. Those control cycles can hide part of the pending store tail.
        pending_store_bytes += store_bytes
        pending_store_bytes = max(0.0, pending_store_bytes - store_ctrl_cycles * store_bw_bytes_per_cycle)
        total_issue_cycles += store_ctrl_cycles

    exposed_store_cycles = pending_store_bytes / store_bw_bytes_per_cycle
    return total_issue_cycles + exposed_store_cycles, exposed_store_cycles


def preferred_reuse_transitions(m_tiles: int, n_tiles: int, gm: int, gn: int, *, reuse_a: bool) -> int:
    total = 0
    for mg in range(0, m_tiles, gm):
        mg_len = min(gm, m_tiles - mg)
        for ng in range(0, n_tiles, gn):
            ng_len = min(gn, n_tiles - ng)
            if reuse_a:
                total += mg_len * max(0, ng_len - 1)
            else:
                total += ng_len * max(0, mg_len - 1)
    return total


def pair_transition_stats(m_tiles: int, n_tiles: int, gm: int, gn: int, *, reuse_a: bool) -> Tuple[int, int]:
    same_a = 0
    same_w = 0
    for mg in range(0, m_tiles, gm):
        mg_len = min(gm, m_tiles - mg)
        for ng in range(0, n_tiles, gn):
            ng_len = min(gn, n_tiles - ng)
            prev_ai = None
            prev_wi = None
            for pair_idx in range(mg_len * ng_len):
                if reuse_a:
                    ai = pair_idx // ng_len
                    wi = pair_idx % ng_len
                else:
                    wi = pair_idx // mg_len
                    ai = pair_idx % mg_len
                if prev_ai is not None:
                    if ai == prev_ai:
                        same_a += 1
                    if wi == prev_wi:
                        same_w += 1
                prev_ai = ai
                prev_wi = wi
    return same_a, same_w


def iter_group_shapes(m_tiles: int, n_tiles: int, gm: int, gn: int) -> Iterable[Tuple[int, int]]:
    for mg in range(0, m_tiles, gm):
        mg_len = min(gm, m_tiles - mg)
        for ng in range(0, n_tiles, gn):
            ng_len = min(gn, n_tiles - ng)
            if mg_len > 0 and ng_len > 0:
                yield mg_len, ng_len


def chunk_indices(length: int, block: int) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    start = 0
    while start < length:
        end = min(length, start + block)
        out.append(tuple(range(start, end)))
        start = end
    return out


def iter_window_order(a_windows: int, w_windows: int, *, row_snake: bool) -> Iterable[Tuple[int, int]]:
    if row_snake:
        for a_idx in range(a_windows):
            if a_idx % 2 == 0:
                w_iter = range(w_windows)
            else:
                w_iter = range(w_windows - 1, -1, -1)
            for w_idx in w_iter:
                yield a_idx, w_idx
    else:
        for w_idx in range(w_windows):
            if w_idx % 2 == 0:
                a_iter = range(a_windows)
            else:
                a_iter = range(a_windows - 1, -1, -1)
            for a_idx in a_iter:
                yield a_idx, w_idx


def iter_window_pairs(a_block: Sequence[int], w_block: Sequence[int], *, reuse_a: bool) -> Iterable[Tuple[int, int]]:
    if reuse_a:
        for a_pos, a_idx in enumerate(a_block):
            if a_pos % 2 == 0:
                w_iter = w_block
            else:
                w_iter = tuple(reversed(w_block))
            for w_idx in w_iter:
                yield a_idx, w_idx
    else:
        for w_pos, w_idx in enumerate(w_block):
            if w_pos % 2 == 0:
                a_iter = a_block
            else:
                a_iter = tuple(reversed(a_block))
            for a_idx in a_iter:
                yield a_idx, w_idx


def build_slot_cache_group_plan(
    mg_len: int,
    ng_len: int,
    *,
    cfg: Config,
    k_tiles: Sequence[int],
    prec_bits: int,
    bw_bytes_per_cycle: float,
    reuse_a: bool,
) -> GroupExecutionPlan:
    """Plan one gm/gn group under the single-pair BMPMM/BMPSE protocol.

    A group is executed as a sequence of single-pair compute instructions. The
    only group-level reuse comes from keeping up to four resident bank-pair
    slots in VRF and traversing the group in a deterministic windowed order.
    """
    total_slots = min(4, mg_len + ng_len)
    best_plan: GroupExecutionPlan | None = None

    for a_slots in range(1, min(mg_len, total_slots - 1) + 1):
        w_slots = total_slots - a_slots
        if w_slots < 1 or w_slots > ng_len:
            continue

        a_windows = chunk_indices(mg_len, a_slots)
        w_windows = chunk_indices(ng_len, w_slots)
        if not a_windows or not w_windows:
            continue

        for row_snake in (True, False):
            prep_cycles: List[float] = []
            compute_cycles: List[float] = []
            total_load_cycles = 0.0
            total_load_bytes = 0.0
            total_control_cycles = 0.0
            emit_cfg_count = 0
            load_a_count = 0
            load_w_count = 0
            pair_same_a = 0
            pair_same_w = 0

            for k_idx, k_exec in enumerate(k_tiles):
                a_tile_bytes = cfg.mtile * k_exec * ACT_BITS / 8.0
                w_tile_bytes = k_exec * cfg.ntile * prec_bits / 8.0
                compute_cycle = single_compute_cycles(cfg.mtile, cfg.ntile, k_exec, prec_bits)
                prev_a_window: frozenset[int] | None = None
                prev_w_window: frozenset[int] | None = None
                prev_pair: Tuple[int, int] | None = None

                for a_win_idx, w_win_idx in iter_window_order(len(a_windows), len(w_windows), row_snake=row_snake):
                    a_block = a_windows[a_win_idx]
                    w_block = w_windows[w_win_idx]
                    a_window = frozenset(a_block)
                    w_window = frozenset(w_block)

                    new_a_count = len(a_window if prev_a_window is None else (a_window - prev_a_window))
                    new_w_count = len(w_window if prev_w_window is None else (w_window - prev_w_window))
                    load_cycle = (new_a_count * a_tile_bytes + new_w_count * w_tile_bytes) / bw_bytes_per_cycle
                    ctrl_cycle = EMIT_CFG_CYCLES

                    emit_cfg_count += 1
                    total_control_cycles += ctrl_cycle
                    load_a_count += new_a_count
                    load_w_count += new_w_count
                    total_load_cycles += load_cycle
                    total_load_bytes += new_a_count * a_tile_bytes + new_w_count * w_tile_bytes

                    for pair_ord, pair in enumerate(iter_window_pairs(a_block, w_block, reuse_a=reuse_a)):
                        prep_cycles.append(load_cycle + ctrl_cycle if pair_ord == 0 else 0.0)
                        compute_cycles.append(compute_cycle)
                        if k_idx == 0:
                            if prev_pair is not None:
                                if pair[0] == prev_pair[0]:
                                    pair_same_a += 1
                                if pair[1] == prev_pair[1]:
                                    pair_same_w += 1
                            prev_pair = pair

                    prev_a_window = a_window
                    prev_w_window = w_window

            pipeline_cycles, hidden_load_cycles = simulate_load_compute_pipeline(prep_cycles, compute_cycles)
            preferred_transitions = pair_same_a if reuse_a else pair_same_w
            plan = GroupExecutionPlan(
                pipeline_cycles=pipeline_cycles,
                compute_cycles=sum(compute_cycles),
                load_cycles=total_load_cycles,
                load_bytes=total_load_bytes,
                control_cycles=total_control_cycles,
                hidden_load_cycles=hidden_load_cycles,
                window_count=len(a_windows) * len(w_windows),
                emit_cfg_count=emit_cfg_count,
                load_a_count=load_a_count,
                load_w_count=load_w_count,
                preferred_reuse_transitions=preferred_transitions,
                pair_same_a_transitions=pair_same_a,
                pair_same_w_transitions=pair_same_w,
                resident_a_slots=a_slots,
                resident_w_slots=w_slots,
                resident_row_snake=1 if row_snake else 0,
            )
            if best_plan is None or better_group_plan(plan, best_plan):
                best_plan = plan

    if best_plan is None:
        raise ValueError(f"No slot-cache plan for group shape ({mg_len}, {ng_len})")
    return best_plan


def better_group_plan(a: GroupExecutionPlan, b: GroupExecutionPlan) -> bool:
    if a.pipeline_cycles != b.pipeline_cycles:
        return a.pipeline_cycles < b.pipeline_cycles
    if a.load_bytes != b.load_bytes:
        return a.load_bytes < b.load_bytes
    if a.hidden_load_cycles != b.hidden_load_cycles:
        return a.hidden_load_cycles > b.hidden_load_cycles
    if a.preferred_reuse_transitions != b.preferred_reuse_transitions:
        return a.preferred_reuse_transitions > b.preferred_reuse_transitions
    if a.resident_a_slots != b.resident_a_slots:
        return a.resident_a_slots > b.resident_a_slots
    return a.resident_row_snake < b.resident_row_snake


def _init_worker(
    configs: Sequence[Config],
    prec_bits: int,
    out_bits: int,
    bw_bytes_per_cycle: float,
    supported_keys: set[tuple[int, int, int, int, int, int]],
    group_load_model: str,
) -> None:
    global _WORKER_CONFIGS
    global _WORKER_PREC_BITS
    global _WORKER_OUT_BITS
    global _WORKER_BW_BYTES_PER_CYCLE
    global _WORKER_SUPPORTED_KEYS
    global _WORKER_GROUP_LOAD_MODEL
    _WORKER_CONFIGS = configs
    _WORKER_PREC_BITS = prec_bits
    _WORKER_OUT_BITS = out_bits
    _WORKER_BW_BYTES_PER_CYCLE = bw_bytes_per_cycle
    _WORKER_SUPPORTED_KEYS = supported_keys
    _WORKER_GROUP_LOAD_MODEL = group_load_model


def _search_one_shape(task: Tuple[int, Shape]) -> Tuple[int, Dict[str, object], Tuple[int, int, int, int, int, int], float, int]:
    idx, shp = task
    shape_configs = prune_configs_for_shape(shp, _WORKER_CONFIGS)
    if not shape_configs:
        raise RuntimeError(f"No feasible config after pruning for shape {shp}")

    best_cfg: Config | None = None
    best_score: ScoreDetail | None = None

    for cfg in shape_configs:
        ktile = max_feasible_ktile(shp.k, cfg.mtile, cfg.ntile, _WORKER_PREC_BITS)
        if ktile is None:
            continue
        if _WORKER_SUPPORTED_KEYS:
            prec_code = prec_bits_to_code(_WORKER_PREC_BITS)
            k_rem = shp.k
            supported = True
            while k_rem > 0:
                cur_k = min(ktile, k_rem)
                k_exec = align_up(cur_k, K_STEP)
                if (prec_code, k_exec, cfg.mtile, cfg.ntile, cfg.gm, cfg.gn) not in _WORKER_SUPPORTED_KEYS:
                    supported = False
                    break
                k_rem -= cur_k
            if not supported:
                continue

        d = score_config(
            shp,
            cfg,
            k_block=ktile,
            prec_bits=_WORKER_PREC_BITS,
            out_bits=_WORKER_OUT_BITS,
            bw_bytes_per_cycle=_WORKER_BW_BYTES_PER_CYCLE,
            group_load_model=_WORKER_GROUP_LOAD_MODEL,
        )
        if best_score is None or better(d, best_score):
            best_cfg, best_score = cfg, d

    if best_cfg is None or best_score is None:
        raise RuntimeError(f"No best config found for shape {shp}")

    row = {
        "M": shp.m,
        "N": shp.n,
        "K": shp.k,
        "g": best_cfg.g,
        "gm": best_cfg.gm,
        "gn": best_cfg.gn,
        "act_reuse_groups": best_cfg.gn,
        "wgt_reuse_groups": best_cfg.gm,
        "mtile": best_cfg.mtile,
        "ntile": best_cfg.ntile,
        "ktile": best_score.ktile,
        "k_iters": best_score.k_iters,
        "area_x_g": best_cfg.mtile * best_cfg.ntile * best_cfg.g,
        "group_g": best_score.group_g,
        "reuse_a": best_score.reuse_a,
        "preferred_reuse_transitions": best_score.preferred_reuse_transitions,
        "pair_same_a_transitions": best_score.pair_same_a_transitions,
        "pair_same_w_transitions": best_score.pair_same_w_transitions,
        "emit_cfg_count": best_score.emit_cfg_count,
        "load_a_count": best_score.load_a_count,
        "load_w_count": best_score.load_w_count,
        "store_instr_count": best_score.store_instr_count,
        "resident_a_slots": best_score.resident_a_slots,
        "resident_w_slots": best_score.resident_w_slots,
        "resident_row_snake": best_score.resident_row_snake,
        "cycles_total": round(best_score.cycles_total, 3),
        "cycles_compute": round(best_score.cycles_compute, 3),
        "cycles_memory": round(best_score.cycles_memory, 3),
        "cycles_control": round(best_score.cycles_control, 3),
        "cycles_exposed_load": round(best_score.cycles_exposed_load, 3),
        "cycles_exposed_store": round(best_score.cycles_exposed_store, 3),
        "hidden_load_cycles": round(best_score.hidden_load_cycles, 3),
        "hidden_store_cycles": round(best_score.hidden_store_cycles, 3),
        "hidden_load_share": round(best_score.hidden_load_share, 6),
        "hidden_store_share": round(best_score.hidden_store_share, 6),
        "hidden_memory_share": round(best_score.hidden_memory_share, 6),
        "utilization": round(best_score.util, 4),
    }

    key = (best_cfg.g, best_cfg.gm, best_cfg.gn, best_cfg.mtile, best_cfg.ntile, best_score.ktile)
    return idx, row, key, best_score.cycles_total, len(shape_configs)


def score_config(
    shape: Shape,
    cfg: Config,
    *,
    k_block: int,
    prec_bits: int,
    out_bits: int,
    bw_bytes_per_cycle: float,
    group_load_model: str = "resident_pair",
) -> ScoreDetail:
    m_tiles = ceil_div(shape.m, cfg.mtile)
    n_tiles = ceil_div(shape.n, cfg.ntile)
    total_pairs = m_tiles * n_tiles
    group_shape_counts = Counter(iter_group_shapes(m_tiles, n_tiles, cfg.gm, cfg.gn))
    k_tiles: List[int] = []
    k_rem = shape.k
    while k_rem > 0:
        cur_k = min(k_block, k_rem)
        k_tiles.append(align_up(cur_k, K_STEP))
        k_rem -= cur_k
    k_iters = len(k_tiles)

    # Utilization for edge tiles.
    padded_m = m_tiles * cfg.mtile
    padded_n = n_tiles * cfg.ntile
    util = (shape.m * shape.n) / float(padded_m * padded_n)

    reuse_a = template_reuse_a(cfg.mtile, cfg.ntile, prec_bits)
    store_instrs_per_pair = 1
    store_bytes_per_pair = cfg.mtile * cfg.ntile * out_bits / 8.0

    total_mem_bytes = 0.0
    total_compute_cycles = 0.0
    total_control_cycles = 0.0
    total_load_cycles = 0.0
    total_hidden_load_cycles = 0.0
    total_store_cycles = 0.0
    emit_cfg_count = 0
    load_a_count = 0
    load_w_count = 0
    preferred_transitions_total = 0
    pair_same_a_total = 0
    pair_same_w_total = 0
    resident_a_slots = 0
    resident_w_slots = 0
    resident_row_snake = 0
    dominant_group_key: Tuple[int, int] | None = None
    dominant_group_occ = -1
    store_instr_count = total_pairs * store_instrs_per_pair
    store_overlap_stages: List[Tuple[float, float, float, float]] = []

    for (mg_len, ng_len), group_occ in group_shape_counts.items():
        group_pairs = mg_len * ng_len
        if group_load_model in ("resident_pair", "slot_cache"):
            group_plan = build_slot_cache_group_plan(
                mg_len,
                ng_len,
                cfg=cfg,
                k_tiles=k_tiles,
                prec_bits=prec_bits,
                bw_bytes_per_cycle=bw_bytes_per_cycle,
                reuse_a=reuse_a,
            )
            group_store_bytes = group_pairs * store_bytes_per_pair
            group_store_emit_cfg_count = group_plan.window_count
            group_store_ctrl_cycles = (
                group_store_emit_cfg_count * EMIT_CFG_CYCLES
                + group_pairs * store_instrs_per_pair * STORE_BLOCK_CTRL_CYCLES
            )
            for _ in range(group_occ):
                store_overlap_stages.append(
                    (group_plan.pipeline_cycles, group_plan.load_cycles, group_store_bytes, group_store_ctrl_cycles)
                )
            total_mem_bytes += group_occ * (group_plan.load_bytes + group_store_bytes)
            total_compute_cycles += group_occ * group_plan.compute_cycles
            total_control_cycles += group_occ * (group_plan.control_cycles + group_store_ctrl_cycles)
            total_load_cycles += group_occ * group_plan.load_cycles
            total_hidden_load_cycles += group_occ * group_plan.hidden_load_cycles
            total_store_cycles += group_occ * (group_store_bytes / STORE_QUEUE_DRAIN_BYTES_PER_CYCLE)
            emit_cfg_count += group_occ * (group_plan.emit_cfg_count + group_store_emit_cfg_count)
            load_a_count += group_occ * group_plan.load_a_count
            load_w_count += group_occ * group_plan.load_w_count
            preferred_transitions_total += group_occ * group_plan.preferred_reuse_transitions
            pair_same_a_total += group_occ * group_plan.pair_same_a_transitions
            pair_same_w_total += group_occ * group_plan.pair_same_w_transitions
            if group_occ > dominant_group_occ or (
                group_occ == dominant_group_occ
                and dominant_group_key is not None
                and (mg_len * ng_len) > (dominant_group_key[0] * dominant_group_key[1])
            ):
                dominant_group_occ = group_occ
                dominant_group_key = (mg_len, ng_len)
                resident_a_slots = group_plan.resident_a_slots
                resident_w_slots = group_plan.resident_w_slots
                resident_row_snake = group_plan.resident_row_snake
        elif group_load_model == "bank_limited":
            prep_cycles: List[float] = []
            compute_cycles: List[float] = []
            pair_load_cycles = 0.0
            pair_compute_cycles = 0.0
            pair_control_cycles = 0.0
            pair_load_bytes = 0.0

            for k_exec in k_tiles:
                a_tile_bytes = cfg.mtile * k_exec * ACT_BITS / 8.0
                w_tile_bytes = k_exec * cfg.ntile * prec_bits / 8.0
                a_block_cycles = a_tile_bytes / bw_bytes_per_cycle
                w_block_cycles = w_tile_bytes / bw_bytes_per_cycle
                compute_cycle = single_compute_cycles(cfg.mtile, cfg.ntile, k_exec, prec_bits)
                prep_cycle = EMIT_CFG_CYCLES + a_block_cycles + w_block_cycles

                prep_cycles.append(prep_cycle)
                compute_cycles.append(compute_cycle)

                pair_load_cycles += a_block_cycles + w_block_cycles
                pair_compute_cycles += compute_cycle
                pair_control_cycles += EMIT_CFG_CYCLES
                pair_load_bytes += a_tile_bytes + w_tile_bytes

            pair_pipeline_cycles, _pair_hidden_prep_cycles = simulate_load_compute_pipeline(prep_cycles, compute_cycles)
            pair_store_bytes = store_bytes_per_pair
            pair_store_ctrl_cycles = store_instrs_per_pair * STORE_BLOCK_CTRL_CYCLES
            exposed_load_cycles_pair = max(0.0, pair_pipeline_cycles - pair_compute_cycles - pair_control_cycles)
            hidden_load_cycles_pair = max(0.0, pair_load_cycles - exposed_load_cycles_pair)

            for _ in range(group_occ * group_pairs):
                store_overlap_stages.append(
                    (pair_pipeline_cycles, pair_load_cycles, pair_store_bytes, pair_store_ctrl_cycles)
                )

            total_mem_bytes += group_occ * group_pairs * (pair_load_bytes + pair_store_bytes)
            total_compute_cycles += group_occ * group_pairs * pair_compute_cycles
            total_control_cycles += group_occ * group_pairs * (pair_control_cycles + pair_store_ctrl_cycles)
            total_load_cycles += group_occ * group_pairs * pair_load_cycles
            total_hidden_load_cycles += group_occ * group_pairs * hidden_load_cycles_pair
            total_store_cycles += group_occ * group_pairs * (pair_store_bytes / STORE_QUEUE_DRAIN_BYTES_PER_CYCLE)
            emit_cfg_count += group_occ * group_pairs * k_iters
            load_a_count += group_occ * group_pairs * k_iters
            load_w_count += group_occ * group_pairs * k_iters
        elif group_load_model == "optimistic_group":
            prep_cycles = []
            compute_cycles = []
            group_load_cycles = 0.0
            group_compute_cycles = 0.0
            group_control_cycles = 0.0
            group_load_bytes = 0.0
            group_load_a_count = 0
            group_load_w_count = 0

            for k_exec in k_tiles:
                a_tile_bytes = cfg.mtile * k_exec * ACT_BITS / 8.0
                w_tile_bytes = k_exec * cfg.ntile * prec_bits / 8.0
                group_a_bytes = mg_len * a_tile_bytes
                group_w_bytes = ng_len * w_tile_bytes
                a_block_cycles = group_a_bytes / bw_bytes_per_cycle
                w_block_cycles = group_w_bytes / bw_bytes_per_cycle
                compute_cycle = group_pairs * single_compute_cycles(cfg.mtile, cfg.ntile, k_exec, prec_bits)
                prep_cycle = EMIT_CFG_CYCLES + a_block_cycles + w_block_cycles

                prep_cycles.append(prep_cycle)
                compute_cycles.append(compute_cycle)

                group_load_cycles += a_block_cycles + w_block_cycles
                group_compute_cycles += compute_cycle
                group_control_cycles += EMIT_CFG_CYCLES
                group_load_bytes += group_a_bytes + group_w_bytes
                group_load_a_count += mg_len
                group_load_w_count += ng_len

            group_pipeline_cycles, _group_hidden_prep_cycles = simulate_load_compute_pipeline(prep_cycles, compute_cycles)
            group_store_bytes = group_pairs * store_bytes_per_pair
            group_store_ctrl_cycles = group_pairs * store_instrs_per_pair * STORE_BLOCK_CTRL_CYCLES
            exposed_load_cycles_group = max(0.0, group_pipeline_cycles - group_compute_cycles - group_control_cycles)
            hidden_load_cycles_group = max(0.0, group_load_cycles - exposed_load_cycles_group)

            for _ in range(group_occ):
                store_overlap_stages.append(
                    (group_pipeline_cycles, group_load_cycles, group_store_bytes, group_store_ctrl_cycles)
                )
            total_mem_bytes += group_occ * (group_load_bytes + group_store_bytes)
            total_compute_cycles += group_occ * group_compute_cycles
            total_control_cycles += group_occ * (group_control_cycles + group_store_ctrl_cycles)
            total_load_cycles += group_occ * group_load_cycles
            total_hidden_load_cycles += group_occ * hidden_load_cycles_group
            total_store_cycles += group_occ * (group_store_bytes / STORE_QUEUE_DRAIN_BYTES_PER_CYCLE)
            emit_cfg_count += group_occ * k_iters
            load_a_count += group_occ * group_load_a_count
            load_w_count += group_occ * group_load_w_count
        else:
            raise ValueError(f"Unsupported group_load_model: {group_load_model}")

    total_cycles, exposed_store_cycles = score_store_overlap(
        store_overlap_stages,
        store_bw_bytes_per_cycle=STORE_QUEUE_DRAIN_BYTES_PER_CYCLE,
    )

    total_memory_cycles = total_load_cycles + total_store_cycles
    hidden_load_cycles = min(total_hidden_load_cycles, total_load_cycles)
    hidden_store_cycles = max(0.0, total_store_cycles - exposed_store_cycles)
    hidden_load_share = hidden_load_cycles / total_load_cycles if total_load_cycles > 0 else 0.0
    hidden_store_share = hidden_store_cycles / total_store_cycles if total_store_cycles > 0 else 0.0
    hidden_memory_share = (hidden_load_cycles + hidden_store_cycles) / total_memory_cycles if total_memory_cycles > 0 else 0.0
    cycles_exposed_load = max(0.0, total_load_cycles - hidden_load_cycles)
    if group_load_model in ("resident_pair", "slot_cache"):
        preferred_transitions = preferred_transitions_total
        pair_same_a_transitions = pair_same_a_total
        pair_same_w_transitions = pair_same_w_total
    else:
        preferred_transitions = preferred_reuse_transitions(m_tiles, n_tiles, cfg.gm, cfg.gn, reuse_a=reuse_a)
        pair_same_a_transitions, pair_same_w_transitions = pair_transition_stats(
            m_tiles,
            n_tiles,
            cfg.gm,
            cfg.gn,
            reuse_a=reuse_a,
        )

    return ScoreDetail(
        cycles_total=total_cycles,
        cycles_compute=total_compute_cycles,
        cycles_memory=total_memory_cycles,
        cycles_control=total_control_cycles,
        cycles_exposed_load=cycles_exposed_load,
        cycles_exposed_store=exposed_store_cycles,
        hidden_load_cycles=hidden_load_cycles,
        hidden_store_cycles=hidden_store_cycles,
        hidden_load_share=hidden_load_share,
        hidden_store_share=hidden_store_share,
        hidden_memory_share=hidden_memory_share,
        util=util,
        group_g=cfg.g,
        preferred_reuse_transitions=preferred_transitions,
        pair_same_a_transitions=pair_same_a_transitions,
        pair_same_w_transitions=pair_same_w_transitions,
        emit_cfg_count=emit_cfg_count,
        load_a_count=load_a_count,
        load_w_count=load_w_count,
        store_instr_count=store_instr_count,
        resident_a_slots=resident_a_slots if group_load_model in ("resident_pair", "slot_cache") else 0,
        resident_w_slots=resident_w_slots if group_load_model in ("resident_pair", "slot_cache") else 0,
        resident_row_snake=resident_row_snake if group_load_model in ("resident_pair", "slot_cache") else 0,
        ktile=k_block,
        k_iters=k_iters,
        reuse_a=1 if reuse_a else 0,
    )


def better(a: ScoreDetail, b: ScoreDetail) -> bool:
    if a.cycles_total != b.cycles_total:
        return a.cycles_total < b.cycles_total
    a_exposed_mem = a.cycles_exposed_load + a.cycles_exposed_store
    b_exposed_mem = b.cycles_exposed_load + b.cycles_exposed_store
    if a_exposed_mem != b_exposed_mem:
        return a_exposed_mem < b_exposed_mem
    return a.hidden_memory_share > b.hidden_memory_share


def write_shape_best_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "M",
        "N",
        "K",
        "g",
        "gm",
        "gn",
        "act_reuse_groups",
        "wgt_reuse_groups",
        "mtile",
        "ntile",
        "ktile",
        "k_iters",
        "area_x_g",
        "group_g",
        "reuse_a",
        "preferred_reuse_transitions",
        "pair_same_a_transitions",
        "pair_same_w_transitions",
        "emit_cfg_count",
        "load_a_count",
        "load_w_count",
        "store_instr_count",
        "resident_a_slots",
        "resident_w_slots",
        "resident_row_snake",
        "cycles_total",
        "cycles_compute",
        "cycles_memory",
        "cycles_control",
        "cycles_exposed_load",
        "cycles_exposed_store",
        "hidden_load_cycles",
        "hidden_store_cycles",
        "hidden_load_share",
        "hidden_store_share",
        "hidden_memory_share",
        "utilization",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_config_rank_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "rank",
        "g",
        "gm",
        "gn",
        "act_reuse_groups",
        "wgt_reuse_groups",
        "mtile",
        "ntile",
        "ktile",
        "area_x_g",
        "picked_shapes",
        "avg_total_cycles",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-shape exhaustive search aligned to the current BMPMM software template")
    ap.add_argument("--shapes-csv", required=True, help="CSV with columns M,N,K")
    ap.add_argument("--out-best-csv", default="tmp/llm_shape_best_innovative.csv")
    ap.add_argument(
        "--out-anchor-csv",
        default="tmp/llm_shape_anchor_innovative.csv",
        help="Compatibility output: config ranking aggregated from per-shape winners",
    )
    ap.add_argument("--buffer-bits", type=int, default=16384, help="Buffer capacity in bits; legal configs satisfy mtile*ntile*g*16<=buffer_bits")
    ap.add_argument("--prec-bits", type=int, default=1, choices=[1, 2, 4], help="Weight bit-width supported by the current BMPU RTL/software path")
    ap.add_argument("--out-bits", type=int, default=16)
    ap.add_argument("--bw-bytes-per-cycle", type=float, default=16.0, help="AXI payload bandwidth in Bytes/cycle (128bit/cycle = 16)")
    ap.add_argument(
        "--group-load-model",
        default="resident_pair",
        choices=["resident_pair", "slot_cache", "bank_limited", "optimistic_group"],
        help="Operand-load reuse model for gm/gn. 'resident_pair' models four resident VRF bank-pair slots under single-pair BMPMM/BMPSE execution; 'slot_cache' is kept as a compatibility alias.",
    )
    ap.add_argument(
        "--enforce-dispatch-table",
        action="store_true",
        help="Also require configs to exist in bmpcfg dispatch table; disabled by default for pure performance search",
    )
    ap.add_argument("--progress-every-shapes", type=int, default=5, help="Print progress every N shapes")
    ap.add_argument("--jobs", type=int, default=0, help="Parallel worker processes; <=0 uses CPU count")
    ap.add_argument("--chunksize", type=int, default=0, help="Multiprocessing chunksize; <=0 uses auto")
    args = ap.parse_args()

    shapes = load_shapes(args.shapes_csv)
    if args.bw_bytes_per_cycle <= 0:
        raise SystemExit("bw-bytes-per-cycle must be positive")
    supported_keys = load_supported_bmpcfg_keys(BMPCFG_DISPATCH_TABLE) if args.enforce_dispatch_table else set()

    configs = legal_configs(args.buffer_bits)
    if not configs:
        raise SystemExit("No feasible config under capacity constraint mtile*ntile*g*16<=buffer_bits")

    g_list = sorted({c.g for c in configs})
    m_vals = sorted({c.mtile for c in configs})
    n_vals = sorted({c.ntile for c in configs})
    m_max = max(m_vals)
    n_max = max(n_vals)

    best_rows: List[Dict[str, object]] = []
    picked_stats: Dict[Tuple[int, int, int, int, int, int], List[float]] = {}

    t0 = time.time()
    total_eval = 0
    done = 0
    results_by_idx: Dict[int, Dict[str, object]] = {}

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    tasks = list(enumerate(shapes, start=1))
    chunksize = args.chunksize if args.chunksize > 0 else max(1, len(tasks) // max(1, jobs * 8))

    if jobs <= 1:
        _init_worker(configs, args.prec_bits, args.out_bits, args.bw_bytes_per_cycle, supported_keys, args.group_load_model)
        iterator = map(_search_one_shape, tasks)
        for idx, row, key, score, eval_count in iterator:
            results_by_idx[idx] = row
            picked_stats.setdefault(key, []).append(score)
            total_eval += eval_count
            done += 1
            if args.progress_every_shapes > 0 and (done % args.progress_every_shapes == 0 or done == len(shapes)):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                remain = len(shapes) - done
                eta = remain / rate if rate > 0 else 0.0
                print(
                    f"[progress] shapes {done}/{len(shapes)} ({done*100.0/len(shapes):.1f}%), "
                    f"elapsed={elapsed:.1f}s, eta={eta:.1f}s, "
                    f"avg-cfg/shape={total_eval/max(1, done):.1f}"
                )
    else:
        with mp.Pool(
            processes=jobs,
            initializer=_init_worker,
            initargs=(configs, args.prec_bits, args.out_bits, args.bw_bytes_per_cycle, supported_keys, args.group_load_model),
        ) as pool:
            for idx, row, key, score, eval_count in pool.imap_unordered(_search_one_shape, tasks, chunksize=chunksize):
                results_by_idx[idx] = row
                picked_stats.setdefault(key, []).append(score)
                total_eval += eval_count
                done += 1
                if args.progress_every_shapes > 0 and (done % args.progress_every_shapes == 0 or done == len(shapes)):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remain = len(shapes) - done
                    eta = remain / rate if rate > 0 else 0.0
                    print(
                        f"[progress] shapes {done}/{len(shapes)} ({done*100.0/len(shapes):.1f}%), "
                        f"elapsed={elapsed:.1f}s, eta={eta:.1f}s, "
                        f"avg-cfg/shape={total_eval/max(1, done):.1f}"
                    )

    for idx in range(1, len(shapes) + 1):
        best_rows.append(results_by_idx[idx])

    write_shape_best_csv(args.out_best_csv, best_rows)

    rank_rows: List[Dict[str, object]] = []
    ranked = sorted(picked_stats.items(), key=lambda kv: (-len(kv[1]), sum(kv[1]) / len(kv[1])))
    for idx, (k, vals) in enumerate(ranked, start=1):
        g, gm, gn, mt, nt, kt = k
        rank_rows.append(
            {
                "rank": idx,
                "g": g,
                "gm": gm,
                "gn": gn,
                "act_reuse_groups": gn,
                "wgt_reuse_groups": gm,
                "mtile": mt,
                "ntile": nt,
                "ktile": kt,
                "area_x_g": g * mt * nt,
                "picked_shapes": len(vals),
                "avg_total_cycles": round(sum(vals) / len(vals), 3),
            }
        )
    write_config_rank_csv(args.out_anchor_csv, rank_rows)

    print(f"Loaded shapes: {len(shapes)}")
    print(f"Hardware mtile granularity: min={M_TILE_MIN}, step={M_TILE_STEP}")
    print(f"Hardware ntile granularity: min={N_TILE_MIN}, step={N_TILE_STEP}")
    print(f"Software-supported ktile range: 8..{K_TILE_MAX_SW} (aligned to {K_STEP})")
    print(f"Group load model: {args.group_load_model}")
    if args.enforce_dispatch_table:
        print(f"Loaded bmpcfg tuples from dispatch table: {len(supported_keys)}")
    else:
        print("Dispatch-table filtering: disabled")
    print(f"Legal mtile values: {m_vals}")
    print(f"Legal ntile values: {n_vals}")
    g_max_buffer = max(1, args.buffer_bits // max(1, M_TILE_MIN * N_TILE_MIN * BUFFER_ELEM_BITS))
    print(f"Auto g range: {g_list[0]}..{g_list[-1]} ({len(g_list)} candidates)")
    print(f"g upper bound sources: buffer={g_max_buffer}, implementation={G_MAX_HW}, effective={min(g_max_buffer, G_MAX_HW)}")
    print(f"Global feasible configs: {len(configs)}")
    print(f"Parallel jobs: {jobs}")
    print(f"Parallel chunksize: {chunksize}")
    print(f"Average evaluated configs per shape (after pruning): {total_eval/max(1, len(shapes)):.1f}")
    print(f"Wrote per-shape best: {args.out_best_csv}")
    print(f"Wrote config ranking: {args.out_anchor_csv}")


if __name__ == "__main__":
    main()
