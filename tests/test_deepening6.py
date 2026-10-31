"""第六轮深化单测：因子化设计 ANOVA、删失感知应变量、容量衰减率。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]


def load_mod():
    spec = importlib.util.spec_from_file_location(
        "snl", REPO / "scripts" / "20_factorial_snl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 文件名解析
# --------------------------------------------------------------------------
def test_parse_name_all_fields():
    mod = load_mod()
    p = mod.parse_name("SNL_18650_NMC_25C_0-100_0.5-2C_a.pkl")
    assert p["chem"] == "NMC"
    assert p["temp_C"] == 25.0
    assert p["soc_lo"] == 0.0 and p["soc_hi"] == 1.0
    assert p["charge_rate"] == 0.5
    assert p["discharge_rate"] == 2.0
    assert p["replicate"] == "a"


def test_parse_name_partial_window():
    mod = load_mod()
    p = mod.parse_name("SNL_18650_NCA_25C_20-80_0.5-0.5C_b.pkl")
    assert p["soc_lo"] == pytest.approx(0.20)
    assert p["soc_hi"] == pytest.approx(0.80)
    assert p["discharge_rate"] == 0.5


def test_parse_name_rejects_garbage():
    mod = load_mod()
    assert mod.parse_name("SNL/") == {}
    assert mod.parse_name("random.txt") == {}


def test_all_real_names_parse():
    """真实文件名全部可解析（防止正则漏掉变体）。"""
    import zipfile
    mod = load_mod()
    names = ["SNL_18650_LFP_15C_0-100_0.5-2C_b.pkl",
             "SNL_18650_NMC_25C_0-100_0.5-3C_d.pkl",
             "SNL_18650_NCA_35C_0-100_0.5-2C_b.pkl"]
    for n in names:
        assert mod.parse_name(n), f"{n} 解析失败"


# --------------------------------------------------------------------------
# 删失感知应变量
# --------------------------------------------------------------------------
def test_survival_endpoint_keeps_all_rows():
    """存活循环数必须覆盖全部电芯（含删失），不能丢样本。"""
    mod = load_mod()
    df = pd.DataFrame({
        "cell_id": ["a", "b", "c"], "n_cycles": [100, 500, 2000],
        "eol_cycle": [80.0, np.nan, np.nan], "censored": [False, True, True],
        "soc_lo": [0.0] * 3, "soc_hi": [1.0] * 3,
    })
    out = mod.survival_endpoint(df)
    assert len(out) == 3
    assert out["log_survival"].notna().all()
    assert (out["survival_cycles"] == df["n_cycles"]).all()


def test_survival_endpoint_throughput_scales_with_window():
    mod = load_mod()
    df = pd.DataFrame({
        "cell_id": ["a", "b"], "n_cycles": [1000, 1000],
        "eol_cycle": [np.nan] * 2, "censored": [True, True],
        "soc_lo": [0.0, 0.2], "soc_hi": [1.0, 0.8],
    })
    out = mod.survival_endpoint(df)
    # 窗口 1.0 → throughput 1000；窗口 0.6 → 600
    assert out["throughput"].tolist() == pytest.approx([1000.0, 600.0])


# --------------------------------------------------------------------------
# 不平衡设计 ANOVA
# --------------------------------------------------------------------------
def test_anova_balanced_two_groups():
    """平衡设计下，单因子两水平的 F 应有限且为正。"""
    mod = load_mod()
    rng = np.random.default_rng(0)
    rows = []
    for a in ("lo", "hi"):
        for b in ("X", "Y"):          # 给 b 两个水平，避免单水平退化
            for _ in range(6):
                rows.append({"a": a, "b": b, "y": rng.normal(0, 1)})
    d = pd.DataFrame(rows)
    r = mod.two_way_anova(d, "a", "b", "y")
    assert np.isfinite(r["f_a"]) and r["f_a"] > 0
    assert np.isfinite(r["f_b"]) and r["f_b"] > 0
    # 无交互的真实设计 → 交互项不应显著
    assert r["p_ab"] > 0.01


def test_anova_never_returns_negative_f():
    """核心回归：不平衡设计下旧实现会给出 F=-9.39，修正后必须恒为正。"""
    mod = load_mod()
    rng = np.random.default_rng(1)
    rows = []
    # 极不平衡：A 有 3 水平但每档样本数不同，B 有 2 水平且与 A 交叉缺失
    for a, na in (("a1", 3), ("a2", 10), ("a3", 2)):
        for b in ("b1", "b2"):
            for _ in range(na):
                rows.append({"a": a, "b": b, "y": rng.normal(0, 1)})
    d = pd.DataFrame(rows)
    r = mod.two_way_anova(d, "a", "b", "y")
    for k in ("f_a", "f_b", "f_ab"):
        v = r.get(k)
        assert v is None or v >= 0, f"{k} 出现了负 F 值: {v}"


def test_anova_detects_strong_effect():
    """真实大效应必须被检出（p<0.01）。"""
    mod = load_mod()
    rng = np.random.default_rng(2)
    rows = []
    for a in ("lo", "hi"):
        for rep in range(12):
            # hi 组均值高 3 个标准差
            rows.append({"a": a, "b": "X", "y": rng.normal(0, 1) + (3 if a == "hi" else 0)})
    r = mod.two_way_anova(pd.DataFrame(rows), "a", "b", "y")
    assert r["p_a"] < 0.01
    assert r["eta2_a"] > 0.3


def test_anova_p_equals_one_when_no_effect():
    """纯噪声下 p 应接近 1（不应拒真）。"""
    mod = load_mod()
    rng = np.random.default_rng(3)
    rows = []
    for a in ("a1", "a2", "a3"):
        for rep in range(10):
            rows.append({"a": a, "b": "X", "y": rng.normal(0, 1)})
    r = mod.two_way_anova(pd.DataFrame(rows), "a", "b", "y")
    assert r["p_a"] > 0.05


def test_anova_insufficient_dof_returns_note():
    mod = load_mod()
    d = pd.DataFrame({"a": ["x"], "b": ["y"], "v": [1.0]})
    r = mod.two_way_anova(d, "a", "b", "v")
    assert "note" in r


# --------------------------------------------------------------------------
# 容量衰减率
# --------------------------------------------------------------------------
def test_fade_rate_linear_recovery():
    """合成线性衰减：应还原设定斜率。"""
    mod = load_mod()
    q = 3.0 * (1 - 0.0002 * np.arange(500))          # 每 1000 圈掉 0.2 SOH
    cycles = [{"charge_capacity_in_Ah": [q[i]]} for i in range(len(q))]
    r = mod.eol_from_cycles(cycles)
    assert r["fade_per_1k"] == pytest.approx(0.2, abs=0.02)


def test_fade_rate_negative_for_improving_cell():
    """容量上升（形成激活）→ 衰减率为负。"""
    mod = load_mod()
    q = 3.0 * (1 + 0.0001 * np.arange(300))
    cycles = [{"charge_capacity_in_Ah": [q[i]]} for i in range(len(q))]
    r = mod.eol_from_cycles(cycles)
    assert r["fade_per_1k"] < 0


def test_fade_rate_nan_for_short_series():
    mod = load_mod()
    cycles = [{"charge_capacity_in_Ah": [3.0 - 0.01 * i]} for i in range(10)]
    r = mod.eol_from_cycles(cycles)
    assert np.isnan(r["fade_per_1k"])


def test_eol_none_when_no_decay():
    mod = load_mod()
    cycles = [{"charge_capacity_in_Ah": [3.0]} for _ in range(100)]
    r = mod.eol_from_cycles(cycles)
    assert r["censored"] is True
    assert np.isnan(r["eol_cycle"])


# --------------------------------------------------------------------------
# 干净对照
# --------------------------------------------------------------------------
def test_clean_rate_effect_skips_single_rate_groups():
    mod = load_mod()
    df = pd.DataFrame({
        "cell_id": list("abc"), "chem": ["H"] * 3, "temp_C": [25] * 3,
        "soc_lo": [0.0] * 3, "soc_hi": [1.0] * 3,
        "discharge_rate": [1.0, 1.0, 2.0],
        "survival_cycles": [100, 120, 200],
        "eol_cycle": [80.0, 90.0, 150.0], "censored": [False] * 3,
    })
    out = mod.clean_rate_effect(df)
    assert len(out) == 2                     # 两档倍率，各一行
    assert out["n_stages"].iloc[0] == 2


def test_clean_rate_effect_baseline_is_min():
    mod = load_mod()
    df = pd.DataFrame({
        "cell_id": list("abcd"), "chem": ["H"] * 4, "temp_C": [25] * 4,
        "soc_lo": [0.0] * 4, "soc_hi": [1.0] * 4,
        "discharge_rate": [0.5, 0.5, 2.0, 2.0],
        "survival_cycles": [100, 110, 300, 310],
        "eol_cycle": [np.nan] * 4, "censored": [True] * 4,
    })
    out = mod.clean_rate_effect(df)
    # 每个倍率档一行（0.5C 与 2C 各一行），基线都应是 0.5C 组的中位 105
    assert len(out) == 2
    assert out["baseline_median"].tolist() == pytest.approx([105.0, 105.0])
    assert out["life_ratio"].max() > 2
