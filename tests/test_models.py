"""A3 充电时长模型与优化器单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fastcharge.chargetime import charge_time_h, fit_charge_time, time_savings_law  # noqa: E402
from fastcharge.lifetime import add_throughput_columns  # noqa: E402
from fastcharge.optimize import ApplicabilityDomain, hypervolume_2d, pareto_front  # noqa: E402
from fastcharge.protocol import build_feature_matrix, protocol_features  # noqa: E402


# --------------------------------------------------------------------------
# 充电时长模型
# --------------------------------------------------------------------------
def _fake_cycles(n_cells: int = 12, seed: int = 0) -> pd.DataFrame:
    """合成一个符合 t = a·span/rate + c·span + d 的 cycles 表。"""
    rng = np.random.default_rng(seed)
    spans = rng.choice([0.3, 0.5, 0.7, 1.0], n_cells)
    rates = rng.choice([0.5, 1.0, 2.0, 3.0], n_cells)
    a, c, d = 0.87, 0.43, 0.17
    true_h = a * spans / rates + c * spans + d
    rows = []
    for i in range(n_cells):
        for k in range(20):
            rows.append({
                "cell_id": f"F{i}", "group": i, "cycle_index": k,
                "soc_span": spans[i], "charge_rate_peak": rates[i],
                "t_charge_s": true_h[i] * 3600 * rng.normal(1, 0.01),
                "t_discharge_s": spans[i] / 0.5 * 3600,
            })
    return pd.DataFrame(rows)


def test_charge_time_fit_recovers_parameters():
    cyc = _fake_cycles()
    res = fit_charge_time(cyc)
    assert res["r2"] > 0.95, f"R² 太低: {res['r2']}"
    assert 0.7 < res["a_cc"] < 1.05
    assert 0.3 < res["c_cv"] < 0.55


def test_charge_time_monotone_decreasing_in_rate():
    """倍率越高时长越短，但收益递减（二阶差分为正）。"""
    v = charge_time_h(0.6, np.array([0.5, 1, 2, 3, 4]), 0.87, 0.43, 0.17)
    assert np.all(np.diff(v) < 0)
    gains = -np.diff(v)
    assert np.all(np.diff(gains) < 0), "时间收益必须递减"


def test_cv_floor_exists():
    """即便倍率趋于无穷，时长也不会低于 CV 地板。"""
    res = fit_charge_time(_fake_cycles())
    floor = res["c_cv"] * 0.6 + res["d_tail"]
    t_inf = charge_time_h(0.6, np.array([1e4]), res["a_cc"], res["c_cv"], res["d_tail"])[0]
    assert t_inf >= floor * 0.999
    assert t_inf < floor * 1.05


def test_savings_law_shape():
    res = fit_charge_time(_fake_cycles())
    law = time_savings_law(res, span=0.6)
    assert len(law) == 7
    assert law["cv_share"].is_monotonic_increasing
    assert law["cv_share"].iloc[-1] > 0.7


# --------------------------------------------------------------------------
# Pareto / 超体积
# --------------------------------------------------------------------------
def test_pareto_front_basic():
    F = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [2.5, 2.5], [1.5, 3.5]])
    idx = pareto_front(F)
    assert set(idx.tolist()) == {0, 1, 2}, f"前三个才是非支配点，得到 {idx}"


def test_pareto_front_all_dominated():
    F = np.array([[5.0, 5.0], [3.0, 3.0], [1.0, 1.0]])
    idx = pareto_front(F)
    assert idx.tolist() == [2]


def test_hypervolume_monotone():
    ref = np.array([10.0, 10.0])
    single = np.array([[2.0, 8.0]])
    better = np.array([[2.0, 5.0]])
    assert hypervolume_2d(single, ref) < hypervolume_2d(better, ref)
    assert hypervolume_2d(np.zeros((0, 2)), ref) == 0.0


def test_hypervolume_vs_dense_grid():
    """前沿点越多，超体积（对同一参照点）不会下降。"""
    ref = np.array([10.0, 10.0])
    a = np.array([[3.0, 6.0]])
    b = np.array([[3.0, 6.0], [6.0, 3.0]])
    assert hypervolume_2d(b, ref) > hypervolume_2d(a, ref)


# --------------------------------------------------------------------------
# 适用域
# --------------------------------------------------------------------------
def test_applicability_domain_rejects_far_points():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)) * np.array([0.2, 0.8, 0.5]) + np.array([0.5, 1.5, 1.0])
    ad = ApplicabilityDomain(X, quantile=0.95)
    inside = ad(X)
    assert inside.mean() > 0.9, "训练点自身必须绝大多数在域内"
    far = np.array([[5.0, 20.0, 10.0]])
    assert not ad(far)[0], "极端外推点必须被拒绝"


def test_applicability_domain_empty_edge():
    X = np.zeros((5, 3))
    ad = ApplicabilityDomain(X, quantile=0.95)
    assert ad(X).all()


# --------------------------------------------------------------------------
# 协议特征
# --------------------------------------------------------------------------
def test_protocol_features_high_rate_high_soc():
    row = {"charge_rate_peak": 2.0, "charge_rate_min": 2.0, "charge_rate_mean": 2.0,
           "charge_rate_median": 2.0, "soc_lo": 0.75, "soc_hi": 1.0, "soc_span": 0.25,
           "peak_rate_soc_start": 0.75, "discharge_rate_mean": 0.5,
           "charge_rate_is_constant": True}
    f = protocol_features(row)
    assert f["f_highrate_highsoc"] == 1.0
    assert f["f_highsoc_exposure"] > 0

    row2 = {**row, "soc_lo": 0.0, "peak_rate_soc_start": 0.0, "soc_span": 1.0}
    f2 = protocol_features(row2)
    assert f2["f_highrate_highsoc"] == 0.0
    assert f2["f_soc_start_deficit"] == 1.0


def test_feature_matrix_shape():
    df = pd.DataFrame([{
        "charge_rate_peak": 1.0, "charge_rate_min": 1.0, "charge_rate_mean": 1.0,
        "charge_rate_median": 1.0, "soc_lo": 0.5, "soc_hi": 1.0, "soc_span": 0.5,
        "peak_rate_soc_start": 0.5, "discharge_rate_mean": 0.5,
        "charge_rate_is_constant": True,
    }] * 3)
    X, names = build_feature_matrix(df)
    assert X.shape == (3, len(names))
    assert "f_soc_span" in names and "f_rate_peak" in names


# --------------------------------------------------------------------------
# EFC 口径
# --------------------------------------------------------------------------
def test_throughput_columns():
    df = pd.DataFrame({
        "eol_cycle": [1000.0, 400.0], "soc_span": [0.3, 1.0],
        "q_ref_Ah": [0.25, 0.25], "censored": [False, False],
    })
    out = add_throughput_columns(df)
    assert out["efc"].tolist() == [300.0, 400.0]
    assert out["q_total_Ah"].tolist() == [75.0, 100.0]
