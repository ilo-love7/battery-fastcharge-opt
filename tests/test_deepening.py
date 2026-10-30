"""深化模块单测：阶梯协议、RL/DP、早期预测、稳健性。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fastcharge.chargetime import cascade_time_h, multistage_time_h, stage_time_h  # noqa: E402
from fastcharge.damage import (  # noqa: E402
    damage_constant_rate,
    damage_stages,
    efc_constant_rate,
    stage_damage_breakdown,
)
from fastcharge.early import _fit_poly, feature_cols  # noqa: E402
from fastcharge.multistage import (  # noqa: E402
    cascade_time,
    make_stages,
    pareto_non_dominated,
)
from fastcharge.rl import (  # noqa: E402
    ChargeEnv,
    evaluate_policy,
    policy_to_stages,
    q_learning,
    solve_dp,
)

CT = {"a_cc": 0.87, "c_cv": 0.43, "d_tail": 0.17}
DMG = {"alpha": 0.0007, "p_rate": 0.12, "beta_soc": 2.0}


# --------------------------------------------------------------------------
# 阶梯协议时长
# --------------------------------------------------------------------------
def test_stage_time_proportional_to_dq():
    t1 = stage_time_h(1.0, 0.0, 0.2, CT)
    t2 = stage_time_h(1.0, 0.0, 0.4, CT)
    assert t2 == pytest.approx(2 * t1)


def test_stage_time_zero_span():
    assert stage_time_h(1.0, 0.5, 0.5, CT) == 0.0


def test_multistage_adds_cv_once():
    """阶梯协议只付一次 CV 尾巴，不是每段都付。"""
    one = multistage_time_h([(1.0, 0.0, 0.5)], CT)
    two = multistage_time_h([(1.0, 0.0, 0.25), (1.0, 0.25, 0.5)], CT)
    # 同倍率拆成两段，CC 部分相同，CV 只算一次 → 时长应相等
    assert two == pytest.approx(one, rel=1e-9)


def test_multistage_tiered_faster_than_slow_constant():
    """低段高倍率 + 高段低倍率，比全程按低倍率快。"""
    tier = multistage_time_h([(2.0, 0.0, 0.3), (1.0, 0.3, 0.6)], CT)
    slow = multistage_time_h([(1.0, 0.0, 0.6)], CT)
    fast = multistage_time_h([(2.0, 0.0, 0.6)], CT)
    assert fast < tier < slow


def test_cascade_time_h_matches_manual():
    rates = [(0.0, 2.0), (0.3, 1.0)]
    t = cascade_time_h(rates, 0.0, 0.6, CT)
    manual = 0.87 * (0.3 / 2.0 + 0.3 / 1.0) + 0.43 * 0.6 + 0.17
    assert t == pytest.approx(manual, rel=1e-9)


def test_cascade_time_equal_rate_reduces_to_constant():
    flat = cascade_time_h([(0.0, 1.0), (0.3, 1.0)], 0.0, 0.6, CT)
    const = multistage_time_h([(1.0, 0.0, 0.6)], CT)
    assert flat == pytest.approx(const, rel=1e-9)


# --------------------------------------------------------------------------
# 损伤模型
# --------------------------------------------------------------------------
def test_damage_constant_rate_increases_with_rate():
    a = damage_constant_rate(0.6, 1.0, 0.2, DMG)
    b = damage_constant_rate(0.6, 3.0, 0.2, DMG)
    assert b > a


def test_damage_high_soc_more_damaging():
    """同样 dQ，高 SOC 段损伤更大（β>0）。"""
    low = damage_stages([(1.0, 0.0, 0.2)], DMG)
    high = damage_stages([(1.0, 0.8, 1.0)], DMG)
    assert high > low


def test_stage_breakdown_shares_sum_to_one():
    stages = [(2.0, 0.0, 0.3), (1.0, 0.3, 0.6)]
    df = stage_damage_breakdown(stages, DMG)
    assert df["damage_share"].sum() == pytest.approx(1.0, rel=1e-6)


def test_efc_inverse_of_damage():
    d = damage_constant_rate(0.5, 1.0, 0.5, DMG)
    assert efc_constant_rate(0.5, 1.0, 0.5, DMG) == pytest.approx(1 / d, rel=1e-9)


# --------------------------------------------------------------------------
# Pareto / 阶梯搜索
# --------------------------------------------------------------------------
def test_make_stages_covers_window():
    st = make_stages(0.2, 0.8, 3, (2.0, 1.0, 0.5))
    assert st[0][1] == pytest.approx(0.2)
    assert st[-1][2] == pytest.approx(0.8)
    assert len(st) == 3


def test_pareto_non_dominated_basic():
    df = pd.DataFrame({
        "time_per_fec_h": [1.0, 1.2, 1.5, 2.0],
        "pred_efc": [500, 800, 800, 1500],
    })
    pf = pareto_non_dominated(df)
    # (1.5, 800) 被 (1.2, 800) 支配 → 去掉
    assert 1.5 not in pf["time_per_fec_h"].tolist()
    # (1.0,500) 与 (2.0,1500) 都是端点，保留
    assert len(pf) == 3


# --------------------------------------------------------------------------
# RL / DP
# --------------------------------------------------------------------------
def _env(lam=0.002, rate_limit=None, sigma=0.0):
    return ChargeEnv(0.0, 0.8, DMG, CT, lam=lam, n_states=8,
                     rate_limit=rate_limit, temp_sigma=sigma)


def test_dp_returns_valid_policy():
    env = _env()
    dp = solve_dp(env)
    assert len(dp["policy_rates"]) == env.n
    assert np.all(np.isfinite(dp["policy_rates"]))
    assert np.all(dp["policy_rates"] > 0)


def test_dp_value_matches_manual_cost():
    """DP 的 V[0] 应等于按最优策略走完全程的总成本。"""
    env = _env()
    dp = solve_dp(env)
    ev = evaluate_policy(env, dp["policy_rates"])
    span = env.edges[-1] - env.edges[0]
    manual = env.lam * ev["time_per_fec_h"] * span + ev["damage"]
    assert manual == pytest.approx(dp["value"], rel=1e-6)


def test_q_learning_matches_dp_on_deterministic():
    """确定性 MDP 上 Q-learning 应收敛到 DP 的最优策略（核心验证）。"""
    env = _env(lam=0.002)
    dp = solve_dp(env)
    rl = q_learning(env, episodes=3000, seed=0)
    assert np.allclose(dp["policy_rates"], rl["policy_rates"])


def test_dp_respects_rate_limit():
    env = _env(rate_limit=lambda s, T: 1.0)
    dp = solve_dp(env)
    assert dp["policy_rates"].max() <= 1.0 + 1e-9


def test_policy_to_stages_compresses():
    edges = np.linspace(0, 0.8, 9)
    rates = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    st = policy_to_stages(rates, edges)
    assert len(st) == 3
    assert [r for r, _, _ in st] == [2.0, 1.0, 0.5]


def test_policy_to_stages_skips_zero():
    edges = np.linspace(0, 0.8, 5)
    rates = np.array([1.0, 0.0, 0.0, 1.0])
    st = policy_to_stages(rates, edges)
    assert all(r > 0 for r, _, _ in st)


def test_evaluate_policy_positive():
    env = _env()
    dp = solve_dp(env)
    ev = evaluate_policy(env, dp["policy_rates"])
    assert ev["time_per_fec_h"] > 0
    assert ev["pred_efc"] > 0
    assert ev["n_stages"] >= 1


# --------------------------------------------------------------------------
# 早期预测特征
# --------------------------------------------------------------------------
def test_fit_poly_linear_recovery():
    y = 1.0 - 0.001 * np.arange(50, dtype=float)
    slope, quad, resid = _fit_poly(y, 2)
    assert slope == pytest.approx(-0.001, rel=1e-6)
    assert abs(quad) < 1e-12
    assert resid < 1e-9


def test_fit_poly_handles_nan():
    y = np.array([1.0, 0.99, np.nan, 0.97, 0.96, 0.95])
    slope, quad, resid = _fit_poly(y, 2)
    assert np.isfinite(slope)


def test_fit_poly_short_series():
    slope, quad, resid = _fit_poly(np.array([1.0, 0.9]), 2)
    assert np.isnan(slope)


def test_feature_cols_naming():
    cols = feature_cols(50)
    assert "n50_slope" in cols and "n50_ce_mean" in cols
    assert all(c.startswith("n50_") for c in cols)
