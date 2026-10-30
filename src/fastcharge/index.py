"""ISU-ILCC zip 直读索引器。

为什么不解压:
    原始 zip 解压后约 29.7 GB，而仓库只需 cycle 级标量摘要（约 1.5 M 行 × 十几列）。
    逐 member 流式读取 → 抽取标量 → 丢弃数组，峰值内存只与单个电芯大小相关（< 300 MB）。

产出两张表:
    cells.parquet   一电芯一行（协议参数 + 寿命结局 + 删失标记）
    cycles.parquet  一循环一行（容量、时长、倍率、效率）
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .config import EOL_SOH_THRESHOLD, METRICS, NOMINAL_CAPACITY_AH

_MEMBER_RE = re.compile(r"ISU_ILCC/ISU-ILCC_G(?P<group>\d+)C(?P<cell>\d+)\.pkl$")


# --------------------------------------------------------------------------
# 协议解析
# --------------------------------------------------------------------------
def _seg(seg: dict[str, Any]) -> dict[str, Any]:
    """把一段 protocol 描述拉平成标量（None 保持 None，不臆造）。"""
    return {
        "rate_C": None if seg.get("rate_in_C") is None else float(seg["rate_in_C"]),
        "current_A": None if seg.get("current_in_A") is None else float(seg["current_in_A"]),
        "power_W": None if seg.get("power_W") is None else float(seg["power_W"]),
        "voltage_V": None if seg.get("voltage_in_V") is None else float(seg["voltage_in_V"]),
        "start_soc": None if seg.get("start_soc") is None else float(seg["start_soc"]),
        "end_soc": None if seg.get("end_soc") is None else float(seg["end_soc"]),
        "end_voltage_V": None
        if seg.get("end_voltage_in_V") is None
        else float(seg["end_voltage_in_V"]),
    }


def parse_protocol(obj: dict[str, Any]) -> dict[str, Any]:
    """从单个电芯 dict 抽出协议层面的结构化参数。

    ISU-ILCC 的统一格式把协议写在 charge_protocol / discharge_protocol 列表里
    （每段带 rate_in_C 与 start/end SOC），这是本项目的核心自变量来源。
    """
    ch = [_seg(s) for s in (obj.get("charge_protocol") or [])]
    dc = [_seg(s) for s in (obj.get("discharge_protocol") or [])]
    soc_lo = float(obj["SOC_interval"][0])
    soc_hi = float(obj["SOC_interval"][1])

    out: dict[str, Any] = {
        "n_charge_segments": len(ch),
        "n_discharge_segments": len(dc),
        "soc_lo": soc_lo,
        "soc_hi": soc_hi,
        "soc_span": max(soc_hi - soc_lo, 1e-6),
        "soc_window_width": soc_hi - soc_lo,
    }
    # 充电：峰值倍率、最低倍率、段内倍率序列
    ch_rates = [s["rate_C"] for s in ch if s["rate_C"] is not None]
    dc_rates = [s["rate_C"] for s in dc if s["rate_C"] is not None]
    out["charge_rate_peak"] = max(ch_rates) if ch_rates else np.nan
    out["charge_rate_min"] = min(ch_rates) if ch_rates else np.nan
    out["charge_rate_mean"] = float(np.mean(ch_rates)) if ch_rates else np.nan
    out["charge_rate_median"] = float(np.median(ch_rates)) if ch_rates else np.nan
    out["discharge_rate_mean"] = float(np.mean(dc_rates)) if dc_rates else np.nan
    out["discharge_rate_peak"] = max(dc_rates) if dc_rates else np.nan

    # 高倍率段落在哪个 SOC 区间（协议结构里最伤寿命的候选因素之一）
    if ch:
        peaks = [s for s in ch if s["rate_C"] is not None and s["rate_C"] == out["charge_rate_peak"]]
        s0 = [s["start_soc"] for s in peaks if s["start_soc"] is not None]
        out["peak_rate_soc_start"] = float(np.mean(s0)) if s0 else np.nan
    else:
        out["peak_rate_soc_start"] = np.nan

    out["charge_rate_is_constant"] = bool(len(set(ch_rates)) <= 1)
    out["nominal_capacity_Ah"] = (
        None if obj.get("nominal_capacity_in_Ah") is None else float(obj["nominal_capacity_in_Ah"])
    )
    return out


# --------------------------------------------------------------------------
# 循环级抽取
# --------------------------------------------------------------------------
@dataclass
class CellSummary:
    cell_id: str
    group: int
    cell_index: int
    n_cycles: int
    q_ref_Ah: float
    q_final_Ah: float
    eol_index: int | None
    eol_cycle: int | None
    censored: bool
    soh_final: float
    q_est_Ah: float


def _phase_durations(t: np.ndarray, i: np.ndarray) -> tuple[float, float, float]:
    """按电流符号切出充电/放电/静置段的时长（秒）。

    BatteryLife 统一格式的电流符号: 正 = 充电。用首个符号跳变定位相位边界。
    """
    if t.size < 3:
        return (np.nan, np.nan, np.nan)
    pos = i > 0
    neg = i < 0
    zero = ~pos & ~neg
    dt_charge = dt_discharge = dt_rest = 0.0
    for mask, acc in ((pos, "c"), (neg, "d"), (zero, "r")):
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        # 只统计连续段（把同符号的多个段合并计总时长，逐步累加相邻采样间隔）
        diffs = np.diff(t)
        seg_mask = mask[:-1] & mask[1:]
        d = float(diffs[seg_mask].sum()) if seg_mask.any() else 0.0
        if acc == "c":
            dt_charge = d
        elif acc == "d":
            dt_discharge = d
        else:
            dt_rest = d
    return (dt_charge, dt_discharge, dt_rest)


def iter_cells(zip_path: Path) -> Iterator[tuple[str, int, int, dict[str, Any]]]:
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            m = _MEMBER_RE.search(name)
            if not m:
                continue
            with zf.open(name) as fh:
                import pickle

                obj = pickle.loads(fh.read())
            yield (
                Path(name).stem,
                int(m.group("group")),
                int(m.group("cell")),
                obj,
            )


def summarize_cell(
    obj: dict[str, Any],
    *,
    soh_threshold: float = EOL_SOH_THRESHOLD,
    smooth_window: int = 11,
) -> tuple[CellSummary, pd.DataFrame]:
    """单电芯 → (cell 级摘要, cycle 级表)。"""
    prot = parse_protocol(obj)
    span = prot["soc_span"]
    cycles = obj["cycle_data"]
    rows = []
    q_charge = np.full(len(cycles), np.nan)
    q_discharge = np.full(len(cycles), np.nan)
    q_norm = np.full(len(cycles), np.nan)

    for k, c in enumerate(cycles):
        arrs = {}
        for key in ("current_in_A", "voltage_in_V", "time_in_s", "charge_capacity_in_Ah",
                    "discharge_capacity_in_Ah"):
            v = c.get(key)
            arrs[key] = None if v is None else np.asarray(v, dtype=float)
        qc = arrs["charge_capacity_in_Ah"]
        qd = arrs["discharge_capacity_in_Ah"]
        i = arrs["current_in_A"]
        t = arrs["time_in_s"]

        cap_c = float(np.nanmax(qc)) if qc is not None and qc.size else np.nan
        cap_d = float(np.nanmax(qd)) if qd is not None and qd.size else np.nan
        q_charge[k] = cap_c
        q_discharge[k] = cap_d
        # 只充了一段 SOC 窗口，按窗口跨度归一化到满容量
        q_norm[k] = cap_c / span if np.isfinite(cap_c) else np.nan

        dur_c = dur_d = dur_r = np.nan
        i_max = i_mean = np.nan
        if i is not None and t is not None and i.size and t.size:
            i_max = float(np.nanmax(np.abs(i)))
            i_mean = float(np.nanmean(np.abs(i)))
            dur_c, dur_d, dur_r = _phase_durations(t, i)
        total_dur = float(t[-1] - t[0]) if t is not None and t.size else np.nan

        rows.append(
            {
                "cycle_index": k,
                "cycle_number": int(c.get("cycle_number", k + 1)),
                "q_charge_Ah": cap_c,
                "q_discharge_Ah": cap_d,
                "q_norm_Ah": q_norm[k],
                "ce": (cap_d / cap_c) if (np.isfinite(cap_d) and np.isfinite(cap_c) and cap_c > 0) else np.nan,
                "i_max_A": i_max,
                "i_abs_mean_A": i_mean,
                "t_charge_s": dur_c,
                "t_discharge_s": dur_d,
                "t_rest_s": dur_r,
                "t_total_s": total_dur,
                "v_max_V": float(np.nanmax(arrs["voltage_in_V"])) if arrs["voltage_in_V"] is not None and arrs["voltage_in_V"].size else np.nan,
                "v_min_V": float(np.nanmin(arrs["voltage_in_V"])) if arrs["voltage_in_V"] is not None and arrs["voltage_in_V"].size else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    cell_cols = {k: v for k, v in prot.items()}
    for k, v in cell_cols.items():
        df[k] = v

    # ---- 寿命结局 ----
    q = pd.Series(q_norm)
    q_smooth = q.rolling(smooth_window, min_periods=1, center=True).median()
    q_ref = float(np.nanmedian(q_norm[:5])) if np.isfinite(q_norm[:5]).any() else np.nan
    if not np.isfinite(q_ref) or q_ref <= 0:
        q_ref = NOMINAL_CAPACITY_AH

    below = (q_smooth / q_ref) < soh_threshold
    eol_idx = int(np.flatnonzero(below.to_numpy())[0]) if below.any() else None
    censored = eol_idx is None
    eol_cycle = int(df.loc[eol_idx, "cycle_number"]) if eol_idx is not None else None
    summary = CellSummary(
        cell_id=str(obj.get("cell_id")),
        group=-1,
        cell_index=-1,
        n_cycles=len(df),
        q_ref_Ah=q_ref,
        q_final_Ah=float(np.nanmedian(q_norm[-5:])),
        eol_index=eol_idx,
        eol_cycle=eol_cycle,
        censored=censored,
        soh_final=float(np.nanmedian(q_norm[-5:]) / q_ref) if np.isfinite(q_ref) else np.nan,
        q_est_Ah=float(q_ref),
    )
    return summary, df


# --------------------------------------------------------------------------
# 批处理入口
# --------------------------------------------------------------------------
def build_tables(
    zip_path: Path,
    *,
    soh_threshold: float = EOL_SOH_THRESHOLD,
    out_dir: Path | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把整个 zip 扫一遍，产出 cells / cycles 两表并落盘 parquet。"""
    out_dir = out_dir or METRICS
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_rows: list[dict[str, Any]] = []
    cyc_frames: list[pd.DataFrame] = []

    for n, (cell_id, group, cell_idx, obj) in enumerate(iter_cells(zip_path), start=1):
        summ, df = summarize_cell(obj, soh_threshold=soh_threshold)
        summ.group = group
        summ.cell_index = cell_idx
        d = asdict(summ)
        d["cell_id"] = cell_id
        # 协议参数合并进 cell 表
        d.update({k: v for k, v in parse_protocol(obj).items()})
        d["cathode"] = obj.get("cathode_material")
        d["anode"] = obj.get("anode_material")
        d["form_factor"] = obj.get("form_factor")
        cell_rows.append(d)

        df.insert(0, "cell_id", cell_id)
        df.insert(1, "group", group)
        df.insert(2, "cell_index", cell_idx)
        cyc_frames.append(df)

        if verbose and (n % 20 == 0 or n == 1):
            print(f"  [{n}] {cell_id} cycles={len(df)} eol={summ.eol_cycle}", flush=True)
        if limit and n >= limit:
            break

    cells = pd.DataFrame(cell_rows)
    cycles = pd.concat(cyc_frames, ignore_index=True)

    cells.to_parquet(out_dir / "cells.parquet", index=False)
    cycles.to_parquet(out_dir / "cycles.parquet", index=False)
    return cells, cycles
