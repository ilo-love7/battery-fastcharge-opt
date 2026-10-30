"""07 — 结论收敛：机理上限 vs 实测数据的交叉检验 + 推荐协议卡。

为什么单独一步
--------------
第 6 步的析锂上限（30°C → 1.0C）来自 OKane2022（NMC811/石墨-SiO2 商用电芯），
而 ISU-ILCC 是 NMC/石墨 0.25 Ah 软包。两者**不是同一只电芯**。若直接把 1.0C 当
ISU-ILCC 的硬上限，会过度保守。本脚本做两件事：
    1. 用 ISU-ILCC 实测数据检验"超 1C 是否真的更差"（固定 SOC 窗口分层对比）；
    2. 给出两套推荐协议：机理保守版 / 数据实证版，并明确标注各自依据与风险。

产出:
    results/metrics/physics_vs_data.csv        机理上限 × 实测数据 交叉检验
    results/metrics/recommendation_card.csv    推荐协议卡（两套口径）
    results/metrics/rate_effect_within_window.csv  固定窗口下的倍率效应
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fastcharge.chargetime import charge_time_h, fit_charge_time  # noqa: E402
from fastcharge.config import METRICS, ensure_dirs  # noqa: E402
from fastcharge.lifetime import add_throughput_columns  # noqa: E402

WINDOWS = [(0.90, 1.01, "满窗口 90–100%"),
           (0.70, 0.90, "深充 70–90%"),
           (0.55, 0.70, "中窗 55–70%"),
           (0.35, 0.55, "浅充 35–55%"),
           (0.25, 0.35, "极浅 25–35%")]


def rate_effect_table(cells: pd.DataFrame, rate_cut: float = 1.0) -> pd.DataFrame:
    """固定 SOC 窗口后，比较"≤rate_cut"与">rate_cut"的寿命，含 EFC 口径。"""
    c = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    rows = []
    for lo, hi, label in WINDOWS:
        s = c[(c["soc_span"] >= lo) & (c["soc_span"] < hi)]
        a = s[s["charge_rate_peak"] <= rate_cut]
        b = s[s["charge_rate_peak"] > rate_cut]
        if len(a) < 3 or len(b) < 3:
            rows.append({"window": label, "n_low_rate": len(a), "n_high_rate": len(b),
                         "note": "样本不足"})
            continue
        u, p = stats.mannwhitneyu(a["eol_cycle"], b["eol_cycle"])
        rows.append({
            "window": label,
            "n_low_rate": len(a),
            "n_high_rate": len(b),
            "median_cycles_low_rate": float(a["eol_cycle"].median()),
            "median_cycles_high_rate": float(b["eol_cycle"].median()),
            "cycles_delta_pct": float((b["eol_cycle"].median() / a["eol_cycle"].median() - 1) * 100),
            "median_efc_low_rate": float(a["efc"].median()),
            "median_efc_high_rate": float(b["efc"].median()),
            "efc_delta_pct": float((b["efc"].median() / a["efc"].median() - 1) * 100),
            "mannwhitney_p": float(p),
            "significant_at_5pct": bool(p < 0.05),
            "note": "",
        })
    return pd.DataFrame(rows)


def physics_vs_data(limits: pd.DataFrame, effect: pd.DataFrame) -> pd.DataFrame:
    """机理上限与实测证据对照表。"""
    lim30 = float(limits.loc[limits["temperature_C"] == 30.0, "rate_limit_C"].iloc[0])
    rows = []
    for _, r in effect.iterrows():
        if r.get("note") == "样本不足":
            continue
        rows.append({
            "window": r["window"],
            "pybamm_safe_rate_30C": lim30,
            "measured_cycles_delta_pct_if_rate>limit": r["cycles_delta_pct"],
            "measured_efc_delta_pct": r["efc_delta_pct"],
            "p_value": r["mannwhitney_p"],
            "verdict": (
                "实测支持机理上限" if (r["cycles_delta_pct"] < -10 and r["mannwhitney_p"] < 0.05)
                else "实测未显示显著恶化（机理判据偏保守）"
            ),
        })
    return pd.DataFrame(rows)


def recommendation_card(gp, ct, cells: pd.DataFrame) -> pd.DataFrame:
    """两套推荐协议 + 理由 + 与基线对比。"""
    from fastcharge.optimize import objectives, feature_rows

    c = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    base_cycles = float(c[(c["soc_span"] > 0.95) & (c["charge_rate_peak"] < 0.6)]["eol_cycle"].median())
    base_t = charge_time_h(1.0, 0.5, ct["a_cc"], ct["c_cv"], ct["d_tail"])

    cands = [
        # (名称, span, rate, drate, 依据)
        ("A 机理保守版（1C 上限）", 0.50, 0.99, 1.0,
         "PyBaMM 30°C 连续安全段上限 1.0C；窗口 50% 兼顾日常补能与寿命"),
        ("B 数据实证版（1.5C）", 0.45, 1.50, 1.2,
         "ISU-ILCC 实测：固定窗口下 1–1.5C 未见显著寿命恶化，且仍在设计空间覆盖内"),
        ("C 折中版（1.25C）", 0.50, 1.25, 1.1,
         "介于机理上限与实测支持区间，留 25% 安全余量"),
        ("D 基线对照 0.5C 满窗口", 1.00, 0.50, 0.5, "传统家充基线"),
    ]
    rows = []
    for name, span, rate, drate, why in cands:
        X = np.array([[span, rate, drate]])
        F = objectives(X, gp, ct)
        t_win = charge_time_h(np.array([span]), np.array([rate]),
                              ct["a_cc"], ct["c_cv"], ct["d_tail"])[0]
        rows.append({
            "protocol": name,
            "soc_window": span,
            "peak_rate_C": rate,
            "discharge_rate_C": drate,
            "charge_time_window_h": float(t_win),
            "time_per_fec_h": float(t_win / span),
            "pred_cycles": float(np.exp(-F[0, 1])),
            "output_per_hour_Ah": float(span * 0.25 / t_win),
            "time_saved_vs_baseline_pct": float((1 - (t_win / span) / (base_t / 1.0)) * 100),
            "note": why,
        })
    df = pd.DataFrame(rows)
    df["baseline_cycles_measured"] = base_cycles
    df["cycles_ratio_vs_baseline"] = df["pred_cycles"] / df["pred_cycles"].iloc[-1]
    return df


def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")
    ct = fit_charge_time(cycles)

    import pickle

    from fastcharge.config import MODELS
    with open(MODELS / "gp_cycles.pkl", "rb") as fh:
        gp = pickle.load(fh)["model"]

    print("=== 1) 固定 SOC 窗口下的倍率效应（实测数据）===")
    eff = rate_effect_table(cells, rate_cut=1.0)
    eff.to_csv(METRICS / "rate_effect_within_window.csv", index=False, encoding="utf-8-sig")
    print(eff.to_string(index=False))

    print("\n=== 2) 机理上限 vs 实测 ===")
    limits = pd.read_csv(METRICS / "rate_limit_table.csv")
    pvd = physics_vs_data(limits, eff)
    pvd.to_csv(METRICS / "physics_vs_data.csv", index=False, encoding="utf-8-sig")
    print(pvd.round(2).to_string(index=False))

    print("\n=== 3) 推荐协议卡 ===")
    card = recommendation_card(gp, ct, cells)
    card.to_csv(METRICS / "recommendation_card.csv", index=False, encoding="utf-8-sig")
    print(card.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
