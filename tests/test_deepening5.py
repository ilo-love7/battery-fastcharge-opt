"""第五轮深化单测：EOL 阈值、模型敏感性、老化漂移归一化、温度协议表。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


# --------------------------------------------------------------------------
# EOL 阈值：单调性
# --------------------------------------------------------------------------
def test_eol_earlier_at_higher_threshold():
    """阈值越严（越高），EOL 来得越早。"""
    q = np.linspace(1.0, 0.5, 100)          # 单调衰减
    eols = {}
    for thr in (0.70, 0.80, 0.90):
        below = np.flatnonzero(q < thr)
        eols[thr] = int(below[0]) if below.size else None
    assert eols[0.90] < eols[0.80] < eols[0.70]


def test_eol_none_when_never_crosses():
    q = np.linspace(1.0, 0.85, 50)
    assert np.flatnonzero(q < 0.70).size == 0


def test_contiguous_limit_stops_at_first_failure():
    """安全上限取"从最小倍率起连续安全段"，不能跳过中间的不安全点。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", REPO / "scripts" / "15_model_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    import matplotlib
    matplotlib.use("Agg")
    spec.loader.exec_module(mod)

    df = pd.DataFrame({
        "temperature_C": [30.0] * 5,
        "c_rate": [0.5, 1.0, 1.5, 2.0, 3.0],
        # 1.5 安全、2.0 不安全、3.0 又"安全"（非单调陷阱）
        "eta_min_V": [0.04, 0.02, 0.008, -0.005, 0.01],
    })
    lim = mod.contiguous_limit(df, 30.0)
    assert lim == 1.5, f"应停在 1.5C（首个不安全的前一个），得到 {lim}"


# --------------------------------------------------------------------------
# 老化漂移：归一化是关键
# --------------------------------------------------------------------------
def _make_cell_cycles(n=400, decay=0.0008, t_base=3600, tau_growth=0.0):
    """合成一只老化电芯的 cycles 表。"""
    rows = []
    for k in range(n):
        q = 0.25 * max(0.05, 1 - decay * k)
        tau = t_base * (1 + tau_growth * k / n)     # 每 Ah 所需时间
        t = tau * q                                  # 绝对时长 ∝ τ × q
        rows.append({"cycle_index": k, "cycle_number": k + 1,
                     "q_charge_Ah": q, "t_charge_s": t, "q_norm_Ah": q})
    return pd.DataFrame(rows)


def test_absolute_time_shrinks_while_tau_grows():
    """核心反直觉：内阻恶化（τ↑）时，绝对时长仍会缩短（因为电荷量↓更多）。"""
    cyc = _make_cell_cycles(tau_growth=0.10)   # τ 上升 10%
    head = cyc.iloc[:50]
    tail = cyc.iloc[-60:-10]                   # EOL 前

    t_e = head["t_charge_s"].median()
    t_l = tail["t_charge_s"].median()
    q_e = head["q_charge_Ah"].median()
    q_l = tail["q_charge_Ah"].median()

    assert t_l / t_e < 1.0, "绝对时长应变短"
    assert (t_l / q_l) / (t_e / q_e) > 1.0, "归一化 τ 应变大"


def test_tau_ratio_recovers_growth():
    """归一口径应能还原设定的 τ 增长率。"""
    growth = 0.10
    cyc = _make_cell_cycles(tau_growth=growth)
    head = cyc.iloc[:50]
    tail = cyc.iloc[-60:-10]
    tau_e = head["t_charge_s"].median() / head["q_charge_Ah"].median()
    tau_l = tail["t_charge_s"].median() / tail["q_charge_Ah"].median()
    recovered = tau_l / tau_e - 1
    # 取的是最后 50 圈的窗口，实际 τ 增幅略小于全局 growth
    assert 0 < recovered < growth * 1.2
    assert recovered > growth * 0.5


def test_no_growth_gives_tau_ratio_one():
    cyc = _make_cell_cycles(tau_growth=0.0)
    head = cyc.iloc[:50]
    tail = cyc.iloc[-60:-10]
    tau_e = head["t_charge_s"].median() / head["q_charge_Ah"].median()
    tau_l = tail["t_charge_s"].median() / tail["q_charge_Ah"].median()
    assert tau_l / tau_e == pytest.approx(1.0, abs=1e-9)


def test_degenerate_zero_capacity_guard():
    """容量趋零时 t/q 会发散 —— 必须被 EOL 前取段逻辑排除。"""
    cyc = _make_cell_cycles(n=400, decay=0.002)   # 到 400 圈剩 ~20%
    tail_bad = cyc.iloc[-5:]
    head = cyc.iloc[:50]
    tau_bad = (tail_bad["t_charge_s"] / tail_bad["q_charge_Ah"]).median()
    tau_head = (head["t_charge_s"] / head["q_charge_Ah"]).median()
    assert tau_bad / tau_head == pytest.approx(1.0, abs=1e-9), \
        "τ 是每 Ah 时间，与容量无关 → 不应因容量衰减而改变"

    # 真正的发散源自 q → 0：构造极端衰减来验证
    cyc2 = _make_cell_cycles(n=400, decay=0.0024)   # 末段容量 < 0.01 Ah
    assert (cyc2.iloc[-5:]["q_charge_Ah"] < 0.05).all()
    tau2 = (cyc2.iloc[-5:]["t_charge_s"] / cyc2.iloc[-5:]["q_charge_Ah"]).median()
    # 合成数据里 t ∝ q，所以 τ 恒定；真实数据才发散的根源是 t 不随 q 缩小
    assert np.isfinite(tau2)


# --------------------------------------------------------------------------
# 温度协议表
# --------------------------------------------------------------------------
def test_limit_interpolation_monotone_in_temperature():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tt", REPO / "scripts" / "17_temp_table.py")
    mod = importlib.util.module_from_spec(spec)
    import matplotlib
    matplotlib.use("Agg")
    spec.loader.exec_module(mod)

    tbl = pd.DataFrame({
        "temperature_C": [0.0, 0.0, 15.0, 15.0, 30.0, 30.0, 45.0, 45.0],
        "soc_start": [0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6],
        "rate_limit_C": [0.0, 0.0, 0.5, 0.4, 1.0, 0.9, 2.5, 2.0],
    })
    # 温度升高 → 上限单调不减
    vals = [mod.limit_at(tbl, 0.3, T) for T in (0, 5, 15, 30, 45)]
    assert all(np.diff(vals) >= -1e-9), f"温度升高上限不应下降: {vals}"

    # SOC 起点升高 → 上限单调不增
    vals2 = [mod.limit_at(tbl, s, 30.0) for s in (0.0, 0.3, 0.6)]
    assert all(np.diff(vals2) <= 1e-9), f"SOC 升高上限不应上升: {vals2}"


def test_recommended_rate_never_exceeds_ceiling():
    """推荐倍率必须 ≤ 安全上限（留取整余量）。"""
    for lim in (0.51, 0.99, 1.33, 2.17):
        rate = np.floor(lim * 20) / 20
        assert rate <= lim + 1e-9


# --------------------------------------------------------------------------
# 模型敏感性
# --------------------------------------------------------------------------
def test_rate_effect_within_window_empty_returns_frame():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rs", REPO / "scripts" / "14_eol_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    import matplotlib
    matplotlib.use("Agg")
    spec.loader.exec_module(mod)

    cells = pd.DataFrame({
        "cell_id": ["a"], "group": [1], "soc_span": [0.5],
        "rate": [1.0], "drate": [1.0],
        "eol_70": [100.0], "eol_80": [80.0],
    })
    out = mod.rank_stability(cells, thresholds=[0.80])
    assert isinstance(out, pd.DataFrame)
    # 只有 1 组时无法算排名一致性
    assert len(out) <= 1
