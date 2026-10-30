"""数据层与协议解析单元测试（不依赖本地大文件）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fastcharge.index import parse_protocol, summarize_cell  # noqa: E402
from fastcharge.lifetime import (  # noqa: E402
    dual_objective_report,
    group_summary,
    kruskal_by_bins,
    rate_effect_within_window,
)


# --------------------------------------------------------------------------
# 合成一只电芯（结构照抄 ISU-ILCC 的 dict 形态）
# --------------------------------------------------------------------------
def make_cell(cell_id="X1", soc_lo=0.5, rate=1.0, n_cycles=200, decay=0.002, seed=0):
    rng = np.random.default_rng(seed)
    span = 1.0 - soc_lo
    cycles = []
    for k in range(n_cycles):
        q = 0.25 * span * max(0.05, 1 - decay * k) + rng.normal(0, 1e-4)
        t_c = span / rate * 3600 * 1.5
        t_d = span / 0.5 * 3600
        n = 200
        t = np.concatenate([np.linspace(0, t_c, n), np.linspace(t_c + 30, t_c + 30 + t_d, n)])
        i = np.concatenate([np.full(n, rate * 0.25), np.full(n, -0.5 * 0.25)])
        qc = np.linspace(0, q, n)
        qd = np.linspace(0, q, n)
        cycles.append({
            "cycle_number": k + 1,
            "current_in_A": i.tolist(),
            "voltage_in_V": (np.linspace(3.0, 4.2, n).tolist() * 2),
            "charge_capacity_in_Ah": qc.tolist() + [q] * n,
            "discharge_capacity_in_Ah": [q] * n + qd.tolist(),
            "time_in_s": t.tolist(),
            "temperature_in_C": None,
            "internal_resistance_in_ohm": None,
        })
    return {
        "cell_id": cell_id,
        "cycle_data": cycles,
        "nominal_capacity_in_Ah": 0.25,
        "depth_of_charge": 1.0,
        "depth_of_discharge": 1.0,
        "SOC_interval": [soc_lo, 1.0],
        "charge_protocol": [{"rate_in_C": rate, "start_soc": soc_lo, "end_soc": 1.0,
                             "current_in_A": None, "power_in_W": None,
                             "voltage_in_V": None, "end_voltage_in_V": None}],
        "discharge_protocol": [{"rate_in_C": 0.5, "start_soc": 1.0, "end_soc": soc_lo,
                                "current_in_A": None, "power_in_W": None,
                                "voltage_in_V": None, "end_voltage_in_V": None}],
    }


# --------------------------------------------------------------------------
# 协议解析
# --------------------------------------------------------------------------
def test_parse_protocol_basic():
    p = parse_protocol(make_cell(soc_lo=0.5, rate=1.0))
    assert p["soc_lo"] == 0.5
    assert p["soc_hi"] == 1.0
    assert abs(p["soc_span"] - 0.5) < 1e-9
    assert p["charge_rate_peak"] == 1.0
    assert p["discharge_rate_mean"] == 0.5
    assert p["charge_rate_is_constant"] is True


def test_parse_protocol_missing_rate_kept_none():
    """协议里 rate_in_C 缺失时不能臆造数值。"""
    cell = make_cell()
    cell["charge_protocol"][0]["rate_in_C"] = None
    p = parse_protocol(cell)
    assert np.isnan(p["charge_rate_peak"]) or p["charge_rate_peak"] is None


# --------------------------------------------------------------------------
# 寿命结局
# --------------------------------------------------------------------------
def test_summarize_cell_finds_eol():
    cell = make_cell(n_cycles=400, decay=0.003, soc_lo=0.5)
    summ, df = summarize_cell(cell, soh_threshold=0.8, smooth_window=5)
    assert df.shape[0] == 400
    assert not summ.censored
    assert summ.eol_cycle is not None and summ.eol_cycle > 0
    assert summ.eol_cycle < 400


def test_summarize_cell_censored_when_never_fails():
    cell = make_cell(n_cycles=60, decay=0.0005, soc_lo=0.5)
    summ, df = summarize_cell(cell, soh_threshold=0.8, smooth_window=5)
    assert summ.censored is True
    assert summ.eol_cycle is None


def test_summarize_cell_q_ref_scales_by_span():
    """只充 50% 窗口时，q_norm 应归一到满容量 ≈0.25 Ah。"""
    cell = make_cell(soc_lo=0.5, rate=1.0, n_cycles=40, decay=0.0001)
    summ, df = summarize_cell(cell)
    assert 0.20 < summ.q_ref_Ah < 0.30, f"q_ref={summ.q_ref_Ah}"


# --------------------------------------------------------------------------
# 统计函数
# --------------------------------------------------------------------------
def _synthetic_cells(n_groups=6, per_group=4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        span = 0.3 + 0.12 * g
        rate = 0.5 + 0.4 * g
        eol = int(3000 * span / (1 + rate))
        for c in range(per_group):
            e = eol * rng.normal(1, 0.1)
            rows.append({
                "cell_id": f"G{g}C{c}", "group": g, "cell_index": c,
                "eol_cycle": e, "censored": False, "n_cycles": e * 1.2,
                "q_ref_Ah": 0.25, "soc_lo": 1 - span, "soc_hi": 1.0,
                "soc_span": span, "charge_rate_peak": rate,
                "charge_rate_min": rate, "charge_rate_mean": rate,
                "charge_rate_median": rate, "discharge_rate_mean": 0.5,
                "peak_rate_soc_start": 1 - span, "charge_rate_is_constant": True,
            })
    return pd.DataFrame(rows)


def test_group_summary_columns():
    cells = _synthetic_cells()
    gs = group_summary(cells)
    assert len(gs) == 6
    for col in ("eol_median", "censored_frac", "n_cells", "eol_cv"):
        assert col in gs.columns
    assert gs["censored_frac"].between(0, 1).all()


def test_kruskal_detects_strong_effect():
    cells = _synthetic_cells()
    r = kruskal_by_bins(cells, "soc_span", [0.0, 0.45, 0.6, 0.8, 0.9, 1.01])
    assert r["p"] < 0.05 or r.get("epsilon_sq", 0) > 0.5


def test_dual_objective_report_has_both_targets():
    cells = _synthetic_cells()
    d = dual_objective_report(cells, ["soc_span", "charge_rate_peak"])
    assert "efc" in d.columns and "eol_cycle" in d.columns
    assert set(d.index) == {"soc_span", "charge_rate_peak"}
    # 两个口径给出不同的相关性强度 → 口径选择确实改变结论
    assert d.loc["soc_span", "efc"] != d.loc["soc_span", "eol_cycle"]
    assert "delta" in d.columns


def test_rate_effect_within_window_handles_empty():
    """样本不足时必须返回空表而不是崩。"""
    cells = _synthetic_cells()
    out = rate_effect_within_window(cells, min_span=0.99)
    assert isinstance(out, pd.DataFrame)
