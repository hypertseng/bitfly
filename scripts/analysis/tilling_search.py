#!/usr/bin/env python3
"""Per-shape exhaustive search for reuse/tile parameters.

This version follows the latest constraints:
- No global anchor search; each shape is searched independently.
- Hardware granularity is fixed in code.
- g range is auto-derived from buffer budget.
- Compute model follows the provided single-compute equation.
- Memory/compute overlap is estimated via explicit 2-engine pipeline simulation.

Notation:
- Shape: (M, N, K)
- Config: (g, gm, gn, mtile, ntile)
- Constraint: mtile * ntile * g * 16 == buffer_bits
- Reuse decomposition: g = gm * gn
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple


# Fixed hardware constants.
R_ROWS = 2
C_COLS = 2
N_VECTOR = 8
M_TILE_MIN = 8
M_TILE_STEP = 8
N_TILE_MIN = 16
N_TILE_STEP = 16
TOTAL_BANKS = 8
BANKS_PER_BLOCK = 2
MAX_RESIDENT_BLOCKS = TOTAL_BANKS // BANKS_PER_BLOCK
PREFETCH_BANKS = 4
PREFETCH_BLOCKS = PREFETCH_BANKS // BANKS_PER_BLOCK
G_MAX_HW = 8
BUFFER_ELEM_BITS = 16
ACT_BITS = 8
ARRAY_M = 8
ARRAY_N = 16
K_STEP = 8
PIPE_LAT = 2
BANK_CAP_BITS = 8192

# Fixed control overhead knobs (can be calibrated with real traces).
TILE_SETUP_CYCLES = 32.0
PHASE_SETUP_BASE_CYCLES = 8.0
PHASE_SETUP_GM_GN_COEFF = 2.0
PHASE_SETUP_PHASE_COEFF = 4.0


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
    overlap_hidden_cycles: float
    overlap_hidden_ratio: float
    overlap_hidden_load_ratio: float
    overlap_ratio: float
    util: float
    phase_count: int
    resident_blocks: int
    ktile: int
    k_iters: int


_WORKER_CONFIGS: Sequence[Config] = ()
_WORKER_PREC_BITS: int = 4
_WORKER_OUT_BITS: int = 16
_WORKER_BW_BYTES_PER_CYCLE: float = 16.0


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_down(x: int, a: int) -> int:
    return (x // a) * a


def align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


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


def divisors(x: int) -> List[int]:
    out: List[int] = []
    for a in range(1, int(math.sqrt(x)) + 1):
        if x % a == 0:
            out.append(a)
            if a * a != x:
                out.append(x // a)
    return sorted(out)


def g_pairs(g: int) -> List[Tuple[int, int]]:
    return [(gm, g // gm) for gm in divisors(g)]


def exact_fit_configs(buffer_bits: int) -> List[Config]:
    # Exact-fit: mtile * ntile * g * 16 == buffer_bits
    if buffer_bits <= 0 or buffer_bits % BUFFER_ELEM_BITS != 0:
        return []

    target_area = buffer_bits // BUFFER_ELEM_BITS
    g_max_buffer = max(1, target_area // max(1, M_TILE_MIN * N_TILE_MIN))
    g_max = min(g_max_buffer, G_MAX_HW)

    out: List[Config] = []
    for g in range(1, g_max + 1):
        if target_area % g != 0:
            continue
        tile_area = target_area // g

        # Enumerate mtile by hardware granularity; ntile is determined uniquely.
        mt = M_TILE_MIN
        while mt <= tile_area:
            if tile_area % mt == 0:
                nt = tile_area // mt
                if nt >= N_TILE_MIN and nt % N_TILE_STEP == 0:
                    for gm, gn in g_pairs(g):
                        out.append(Config(g=g, gm=gm, gn=gn, mtile=mt, ntile=nt))
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
    for g in g_list:
        for gm, gn in g_pairs(g):
            for mt in m_tiles:
                for nt in n_tiles:
                    if mt * nt * g * BUFFER_ELEM_BITS == buffer_bits:
                        yield Config(g=g, gm=gm, gn=gn, mtile=mt, ntile=nt)


def max_feasible_ktile(shape_k: int, mtile: int, ntile: int, prec_bits: int) -> int | None:
    # Activation: mtile * ktile * ACT_BITS <= BANK_CAP_BITS
    k_cap_act = BANK_CAP_BITS // max(1, mtile * ACT_BITS)
    # Weight: ntile * ktile * prec <= 8192 bits
    k_cap_wgt = BANK_CAP_BITS // max(1, ntile * prec_bits)
    k_cap = min(shape_k, k_cap_act, k_cap_wgt)
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
    # Hardware-aligned compute model:
    # T_compute = ceil(m/8) * ceil(n/16) * (ceil(k/8) + PIPE_LAT - 1)
    # `prec_bits` is intentionally not part of this compute formula.
    m_steps = ceil_div(mtile, ARRAY_M)
    n_steps = ceil_div(ntile, ARRAY_N)
    k_steps = ceil_div(k_block, K_STEP)
    return float(m_steps * n_steps * (k_steps + PIPE_LAT - 1))


def phase_setup_cycles(gm_eff: int, gn_eff: int, phase_count: int) -> float:
    # f(gm, gn, phase_count): dynamic setup overhead per phase.
    # Tunable form: base + coeff1*(gm+gn) + coeff2*(phase_count-1).
    return (
        PHASE_SETUP_BASE_CYCLES
        + PHASE_SETUP_GM_GN_COEFF * float(gm_eff + gn_eff)
        + PHASE_SETUP_PHASE_COEFF * float(max(0, phase_count - 1))
    )


def choose_phase_major(a_cnt: int, w_cnt: int, load_a: bool, load_w: bool) -> str:
    if load_a and not load_w:
        return "W_REUSE"
    if load_w and not load_a:
        return "A_REUSE"
    if a_cnt == 1 and w_cnt > 1:
        return "A_REUSE"
    if w_cnt == 1 and a_cnt > 1:
        return "W_REUSE"
    if w_cnt >= a_cnt:
        return "A_REUSE"
    return "W_REUSE"


def simulate_phase_cycles(
    a_cnt: int,
    w_cnt: int,
    *,
    load_a: bool,
    load_w: bool,
    a_block_cycles: float,
    w_block_cycles: float,
    compute_cycles_per_pair: float,
) -> Tuple[float, float, float]:
    # Event-driven phase simulation with one shared load channel and one compute engine.
    # Loads are scheduled dynamically (just-in-time) against the compute order.
    total_pairs = a_cnt * w_cnt
    if total_pairs == 0:
        return 0.0, 0.0, 0.0

    major = choose_phase_major(a_cnt, w_cnt, load_a, load_w)
    pair_order: List[Tuple[int, int]] = []
    if major == "A_REUSE":
        for a_idx in range(a_cnt):
            for w_idx in range(w_cnt):
                pair_order.append((a_idx, w_idx))
    else:
        for w_idx in range(w_cnt):
            for a_idx in range(a_cnt):
                pair_order.append((a_idx, w_idx))

    a_loaded = [not load_a for _ in range(a_cnt)]
    w_loaded = [not load_w for _ in range(w_cnt)]

    load_intervals: List[Tuple[float, float]] = []
    comp_intervals: List[Tuple[float, float]] = []

    t = 0.0
    next_pair = 0
    load_task: Tuple[str, int] | None = None
    load_busy_until = 0.0
    comp_busy_until = 0.0

    def choose_next_load(from_pair: int) -> Tuple[str, int] | None:
        # Prefer the earliest missing dependency in execution order.
        for p in range(from_pair, total_pairs):
            ai, wi = pair_order[p]
            if not a_loaded[ai]:
                return ("A", ai)
            if not w_loaded[wi]:
                return ("W", wi)
        return None

    while next_pair < total_pairs:
        # If load channel is free, issue next required load.
        if load_task is None:
            ld = choose_next_load(next_pair)
            if ld is not None:
                kind, idx = ld
                dur = a_block_cycles if kind == "A" else w_block_cycles
                s = t
                e = s + dur
                load_intervals.append((s, e))
                load_busy_until = e
                load_task = (kind, idx)

        # If compute is free and next pair deps are ready, start compute.
        if t >= comp_busy_until and next_pair < total_pairs:
            ai, wi = pair_order[next_pair]
            if a_loaded[ai] and w_loaded[wi]:
                cs = t
                ce = cs + compute_cycles_per_pair
                comp_intervals.append((cs, ce))
                comp_busy_until = ce
                next_pair += 1
                # Continue immediately to allow zero-gap issuing.
                continue

        # Determine next event time.
        next_events: List[float] = []
        if load_task is not None:
            next_events.append(load_busy_until)
        if t < comp_busy_until:
            next_events.append(comp_busy_until)

        if not next_events:
            # Compute is idle and no in-flight load. Force-issue one missing load.
            ld = choose_next_load(next_pair)
            if ld is None:
                break
            kind, idx = ld
            dur = a_block_cycles if kind == "A" else w_block_cycles
            s = t
            e = s + dur
            load_intervals.append((s, e))
            load_busy_until = e
            load_task = (kind, idx)
            next_events.append(load_busy_until)

        t = min(next_events)

        # Commit finished load.
        if load_task is not None and abs(t - load_busy_until) < 1e-12:
            kind, idx = load_task
            if kind == "A":
                a_loaded[idx] = True
            else:
                w_loaded[idx] = True
            load_task = None

    overlap_hidden = 0.0
    i = 0
    j = 0
    while i < len(load_intervals) and j < len(comp_intervals):
        ls, le = load_intervals[i]
        cs, ce = comp_intervals[j]
        overlap_hidden += max(0.0, min(le, ce) - max(ls, cs))
        if le <= ce:
            i += 1
        else:
            j += 1

    last_load_end = load_intervals[-1][1] if load_intervals else 0.0
    last_comp_end = comp_intervals[-1][1] if comp_intervals else 0.0
    makespan = max(last_load_end, last_comp_end)
    return makespan, last_load_end, overlap_hidden


@lru_cache(maxsize=None)
def phase_chunks(gm: int, gn: int) -> Tuple[Tuple[int, int, bool, bool], ...]:
    """Return phase plan tuples: (a_cnt, w_cnt, load_a, load_w).

    load_a/load_w indicate whether this phase introduces new A/W blocks that must be loaded.
    """
    if gm + gn <= MAX_RESIDENT_BLOCKS:
        return ((gm, gn, True, True),)

    plan: List[Tuple[int, int, bool, bool]] = []
    if gm <= MAX_RESIDENT_BLOCKS and gn > MAX_RESIDENT_BLOCKS:
        # Keep A resident, stream W in chunks.
        w_chunk = max(1, MAX_RESIDENT_BLOCKS - gm)
        remain = gn
        first = True
        while remain > 0:
            cur_w = min(w_chunk, remain)
            plan.append((gm, cur_w, first, True))
            first = False
            remain -= cur_w
        return tuple(plan)

    if gn <= MAX_RESIDENT_BLOCKS and gm > MAX_RESIDENT_BLOCKS:
        # Keep W resident, stream A in chunks.
        a_chunk = max(1, MAX_RESIDENT_BLOCKS - gn)
        remain = gm
        first = True
        while remain > 0:
            cur_a = min(a_chunk, remain)
            plan.append((cur_a, gn, True, first))
            first = False
            remain -= cur_a
        return tuple(plan)

    # Both sides too large: use balanced 2+2 blocking.
    a_chunk = max(1, MAX_RESIDENT_BLOCKS // 2)
    w_chunk = max(1, MAX_RESIDENT_BLOCKS - a_chunk)
    a_rem = gm
    while a_rem > 0:
        cur_a = min(a_chunk, a_rem)
        w_rem = gn
        while w_rem > 0:
            cur_w = min(w_chunk, w_rem)
            plan.append((cur_a, cur_w, True, True))
            w_rem -= cur_w
        a_rem -= cur_a
    return tuple(plan)


def _init_worker(configs: Sequence[Config], prec_bits: int, out_bits: int, bw_bytes_per_cycle: float) -> None:
    global _WORKER_CONFIGS
    global _WORKER_PREC_BITS
    global _WORKER_OUT_BITS
    global _WORKER_BW_BYTES_PER_CYCLE
    _WORKER_CONFIGS = configs
    _WORKER_PREC_BITS = prec_bits
    _WORKER_OUT_BITS = out_bits
    _WORKER_BW_BYTES_PER_CYCLE = bw_bytes_per_cycle


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

        d = score_config(
            shp,
            cfg,
            k_block=ktile,
            prec_bits=_WORKER_PREC_BITS,
            out_bits=_WORKER_OUT_BITS,
            bw_bytes_per_cycle=_WORKER_BW_BYTES_PER_CYCLE,
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
        "act_reuse_groups": best_cfg.gm,
        "wgt_reuse_groups": best_cfg.gn,
        "mtile": best_cfg.mtile,
        "ntile": best_cfg.ntile,
        "ktile": best_score.ktile,
        "k_iters": best_score.k_iters,
        "area_x_g": best_cfg.mtile * best_cfg.ntile * best_cfg.g,
        "resident_blocks": best_score.resident_blocks,
        "phase_count": best_score.phase_count,
        "cycles_total": round(best_score.cycles_total, 3),
        "cycles_compute": round(best_score.cycles_compute, 3),
        "cycles_memory": round(best_score.cycles_memory, 3),
        "cycles_control": round(best_score.cycles_control, 3),
        "cycles_exposed_load": round(best_score.cycles_exposed_load, 3),
        "overlap_hidden_cycles": round(best_score.overlap_hidden_cycles, 3),
        "overlap_hidden_ratio": round(best_score.overlap_hidden_ratio, 6),
        "overlap_hidden_load_ratio": round(best_score.overlap_hidden_load_ratio, 6),
        "overlap_ratio": round(best_score.overlap_ratio, 6),
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
) -> ScoreDetail:
    m_tiles = ceil_div(shape.m, cfg.mtile)
    n_tiles = ceil_div(shape.n, cfg.ntile)
    k_iters = ceil_div(shape.k, k_block)

    # Utilization for edge tiles.
    padded_m = m_tiles * cfg.mtile
    padded_n = n_tiles * cfg.ntile
    util = (shape.m * shape.n) / float(padded_m * padded_n)

    # Activation payload uses fixed 8-bit elements; weight payload uses `prec_bits`.
    a_tile_bytes = cfg.mtile * k_block * ACT_BITS / 8.0
    w_tile_bytes = k_block * cfg.ntile * prec_bits / 8.0
    c_tile_bytes = cfg.mtile * cfg.ntile * out_bits / 8.0

    # Single shared load channel + compute engine.
    timeline_t = 0.0
    total_mem_bytes = 0.0
    total_compute_cycles = 0.0
    total_control_cycles = 0.0
    total_load_cycles = 0.0
    total_hidden_cycles = 0.0

    gm_count_map: Dict[int, int] = {}
    for mg in range(0, m_tiles, cfg.gm):
        gm_eff = min(cfg.gm, m_tiles - mg)
        gm_count_map[gm_eff] = gm_count_map.get(gm_eff, 0) + 1

    gn_count_map: Dict[int, int] = {}
    for ng in range(0, n_tiles, cfg.gn):
        gn_eff = min(cfg.gn, n_tiles - ng)
        gn_count_map[gn_eff] = gn_count_map.get(gn_eff, 0) + 1

    a_block_cycles = a_tile_bytes / bw_bytes_per_cycle
    w_block_cycles = w_tile_bytes / bw_bytes_per_cycle
    compute_cycles_per_pair_base = single_compute_cycles(cfg.mtile, cfg.ntile, k_block, prec_bits) / max(util, 1e-6)

    for gm_eff, gm_cnt in gm_count_map.items():
        for gn_eff, gn_cnt in gn_count_map.items():
            repeat_groups = k_iters * gm_cnt * gn_cnt
            if repeat_groups <= 0:
                continue

            plan = phase_chunks(gm_eff, gn_eff)
            phase_count = len(plan)

            group_makespan = 0.0
            group_last_load_end = 0.0
            group_mem_bytes = 0.0
            group_load_cycles = 0.0
            group_load_blocks = 0
            group_hidden_cycles = 0.0
            group_compute_cycles = 0.0
            group_control_cycles = 0.0
            prefix_makespan = 0.0

            for (a_cnt, w_cnt, load_a, load_w) in plan:
                a_total_bytes = (a_cnt * a_tile_bytes) if load_a else 0.0
                w_total_bytes = (w_cnt * w_tile_bytes) if load_w else 0.0

                compute_pairs = a_cnt * w_cnt
                compute_cycles = compute_pairs * compute_cycles_per_pair_base

                phase_makespan, phase_load_end, phase_hidden = simulate_phase_cycles(
                    a_cnt,
                    w_cnt,
                    load_a=load_a,
                    load_w=load_w,
                    a_block_cycles=a_block_cycles,
                    w_block_cycles=w_block_cycles,
                    compute_cycles_per_pair=compute_cycles_per_pair_base,
                )

                group_mem_bytes += (a_total_bytes + w_total_bytes)
                group_load_cycles += (a_total_bytes + w_total_bytes) / bw_bytes_per_cycle
                group_load_blocks += (a_cnt if load_a else 0) + (w_cnt if load_w else 0)
                group_hidden_cycles += phase_hidden
                group_compute_cycles += compute_cycles
                # Control overhead intentionally removed from the model.
                group_control_cycles += 0.0

                group_last_load_end = prefix_makespan + phase_load_end
                prefix_makespan += phase_makespan

            group_makespan = prefix_makespan

            # Account for cross-group prefetch overlap:
            # next group's exposed load can overlap with previous group's compute.
            group_hidden_one = min(group_hidden_cycles, group_load_cycles)
            group_exposed_load_one = max(0.0, group_load_cycles - group_hidden_one)
            prefetch_fraction = min(1.0, PREFETCH_BLOCKS / float(max(1, group_load_blocks)))
            prefetch_load_cap = group_load_cycles * prefetch_fraction
            cross_hidden_per_transition = min(group_exposed_load_one, group_compute_cycles, prefetch_load_cap)
            cross_hidden_total = max(0, repeat_groups - 1) * cross_hidden_per_transition

            if group_makespan > 0.0:
                timeline_t += repeat_groups * group_makespan - cross_hidden_total

            total_mem_bytes += repeat_groups * group_mem_bytes
            total_load_cycles += repeat_groups * group_load_cycles
            total_hidden_cycles += repeat_groups * group_hidden_cycles + cross_hidden_total
            total_compute_cycles += repeat_groups * group_compute_cycles
            total_control_cycles += 0.0

    # Output store: once after all K accumulation for each output tile.
    store_bytes = m_tiles * n_tiles * c_tile_bytes
    store_cycles = store_bytes / bw_bytes_per_cycle
    total_mem_bytes += store_bytes

    total_control_cycles = 0.0
    total_cycles = timeline_t + store_cycles
    total_memory_cycles = total_load_cycles + store_cycles
    overlap_hidden_ratio = total_hidden_cycles / total_cycles if total_cycles > 0 else 0.0
    overlap_hidden_load_ratio = total_hidden_cycles / total_load_cycles if total_load_cycles > 0 else 0.0
    # Primary overlap metric: hidden load over all memory-side time (load+store).
    overlap_ratio = total_hidden_cycles / total_memory_cycles if total_memory_cycles > 0 else 0.0
    cycles_exposed_load = max(0.0, total_load_cycles - min(total_hidden_cycles, total_load_cycles))

    resident_blocks = cfg.gm + cfg.gn
    phase_count = ceil_div(resident_blocks, MAX_RESIDENT_BLOCKS)

    return ScoreDetail(
        cycles_total=total_cycles,
        cycles_compute=total_compute_cycles,
        cycles_memory=total_mem_bytes / bw_bytes_per_cycle,
        cycles_control=total_control_cycles,
        cycles_exposed_load=cycles_exposed_load,
        overlap_hidden_cycles=min(total_hidden_cycles, total_load_cycles),
        overlap_hidden_ratio=overlap_hidden_ratio,
        overlap_hidden_load_ratio=overlap_hidden_load_ratio,
        overlap_ratio=overlap_ratio,
        util=util,
        phase_count=phase_count,
        resident_blocks=resident_blocks,
        ktile=k_block,
        k_iters=k_iters,
    )


def better(a: ScoreDetail, b: ScoreDetail) -> bool:
    if a.cycles_total != b.cycles_total:
        return a.cycles_total < b.cycles_total
    if a.cycles_exposed_load != b.cycles_exposed_load:
        return a.cycles_exposed_load < b.cycles_exposed_load
    return a.overlap_ratio > b.overlap_ratio


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
        "resident_blocks",
        "phase_count",
        "cycles_total",
        "cycles_compute",
        "cycles_memory",
        "cycles_control",
        "cycles_exposed_load",
        "overlap_hidden_cycles",
        "overlap_hidden_ratio",
        "overlap_hidden_load_ratio",
        "overlap_ratio",
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
    ap = argparse.ArgumentParser(description="Per-shape exhaustive search with pipeline overlap simulation")
    ap.add_argument("--shapes-csv", required=True, help="CSV with columns M,N,K")
    ap.add_argument("--out-best-csv", default="tmp/llm_shape_best_innovative.csv")
    ap.add_argument(
        "--out-anchor-csv",
        default="tmp/llm_shape_anchor_innovative.csv",
        help="Compatibility output: config ranking aggregated from per-shape winners",
    )
    ap.add_argument("--buffer-bits", type=int, default=16384, help="Buffer capacity in bits; exact-fit uses mtile*ntile*g*16=buffer_bits")
    ap.add_argument("--prec-bits", type=int, default=1, choices=[1, 2, 4, 8, 16], help="Weight bit-width")
    ap.add_argument("--out-bits", type=int, default=16)
    ap.add_argument("--bw-bytes-per-cycle", type=float, default=16.0, help="AXI payload bandwidth in Bytes/cycle (128bit/cycle = 16)")
    ap.add_argument("--progress-every-shapes", type=int, default=5, help="Print progress every N shapes")
    ap.add_argument("--jobs", type=int, default=0, help="Parallel worker processes; <=0 uses CPU count")
    ap.add_argument("--chunksize", type=int, default=0, help="Multiprocessing chunksize; <=0 uses auto")
    args = ap.parse_args()

    shapes = load_shapes(args.shapes_csv)
    if args.bw_bytes_per_cycle <= 0:
        raise SystemExit("bw-bytes-per-cycle must be positive")

    # Closed-form exact-fit enumeration is much faster than grid-scan + filter.
    configs = exact_fit_configs(args.buffer_bits)
    if not configs:
        raise SystemExit("No feasible config under exact-fit constraint mtile*ntile*g*16=buffer_bits")

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
        _init_worker(configs, args.prec_bits, args.out_bits, args.bw_bytes_per_cycle)
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
            initargs=(configs, args.prec_bits, args.out_bits, args.bw_bytes_per_cycle),
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
                "act_reuse_groups": gm,
                "wgt_reuse_groups": gn,
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
    print(f"Exact-fit mtile values: {m_vals}")
    print(f"Exact-fit ntile values: {n_vals}")
    g_max_buffer = max(1, args.buffer_bits // max(1, M_TILE_MIN * N_TILE_MIN * BUFFER_ELEM_BITS))
    print(f"Auto g range: {g_list[0]}..{g_list[-1]} ({len(g_list)} candidates)")
    print(f"g upper bound sources: buffer={g_max_buffer}, hardware={G_MAX_HW}, effective={min(g_max_buffer, G_MAX_HW)}")
    print(f"Global feasible configs: {len(configs)}")
    print(f"Parallel jobs: {jobs}")
    print(f"Parallel chunksize: {chunksize}")
    print(f"Average evaluated configs per shape (after pruning): {total_eval/max(1, len(shapes)):.1f}")
    print(f"Wrote per-shape best: {args.out_best_csv}")
    print(f"Wrote config ranking: {args.out_anchor_csv}")


if __name__ == "__main__":
    main()
