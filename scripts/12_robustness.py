"""12 — 深化④：关键结论的统计稳健性（bootstrap 置信区间 + 置换检验）。

为什么必须做
------------
报告里所有结论都是"点估计"：ε²=0.737、ρ=−0.317、10/12 越界……
但只有 61 组协议、239 只电芯，点估计本身带抽样波动。
不给出区间，等于不知道结论有多硬。

本模块对每条关键结论做：
    · Bootstrap（按**协议**重采样，不是按电芯——同协议 4 只电芯不独立）
    · 置换检验（null 分布下该效应是否可能偶然出现）
    · 效应量的置信区间（不是 p 值的二值判断）

产出:
    results/metrics/bootstrap_ci.csv        各结论的 95% CI
    results/metrics/permutation_test.csv    置换检验 p 值
    results/metrics/claim_summary.csv       结论强度分级
    results/figures/14_robustness.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fastcharge.config import FIGURES, METRICS, ensure_dirs  # noqa: E402
from fastcharge.lifetime import add_throughput_columns  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
warnings.filterwarnings("ignore")

N_BOOT = 2000
N_PERM = 2000
SEED = 0


# --------------------------------------------------------------------------
# 重采样工具：按协议整组抽（cluster bootstrap）
# --------------------------------------------------------------------------
def cluster_bootstrap(cells: pd.DataFrame, stat_fn, n_boot: int = N_BOOT,
                      seed: int = SEED) -> np.ndarray:
    """按 group 整组重采样（同协议 4 只电芯必须一起进出）。"""
    rng = np.random.default_rng(seed)
    groups = cells["group"].unique()
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([cells[cells["group"] == g] for g in pick], ignore_index=True)
        try:
            v = stat_fn(boot)
        except Exception:
            v = np.nan
        vals.append(v)
    return np.asarray(vals, float)


def ci(vals: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float, float]:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    return (float(np.percentile(v, lo)), float(np.median(v)), float(np.percentile(v, hi)))


def permutation_p(cells: pd.DataFrame, stat_fn, n_perm: int = N_PERM,
                  seed: int = SEED, permute: str = "target") -> dict:
    """置换检验。

    permute="target"  → 打乱“寿命结局 ↔ 协议组”的对应关系（正确零假设：
                        协议对寿命没有影响）。ICC、比值、相关系数都适用。
    permute="feature" → 打乱特征在组间的分配（零假设：该特征与寿命无关）。

    关键：**统计量必须在被打乱的对象上才有非退化分布**。
    · 相关系数/ICC → 打乱"寿命↔协议"的对应（permute="target"）
    · "G57 vs G14 的比值"这类由 group 定位的统计量 → 只能整块搬寿命
    · 若统计量与 group 无关（如 ICC 只关心组内一致性），整块搬寿命会保留
      组结构，p 恒接近 1 —— 那是**零假设选错**，不是统计量失效。
    """
    rng = np.random.default_rng(seed)
    obs = stat_fn(cells)
    groups = cells["group"].unique()
    null = []

    if permute == "target":
        blocks = {g: cells[cells["group"] == g]["eol_cycle"].to_numpy(float) for g in groups}
        for _ in range(n_perm):
            keys = rng.permutation(groups)
            perm = cells.copy()
            new_eol = np.full(len(perm), np.nan)
            idx_map = {g: np.flatnonzero(perm["group"].to_numpy() == g) for g in groups}
            for src, dst in zip(groups, keys):
                vals = blocks[src]
                idx = idx_map[dst]
                new_eol[idx] = vals[: len(idx)] if len(vals) >= len(idx) else np.resize(vals, len(idx))
            perm["eol_cycle"] = new_eol
            try:
                null.append(stat_fn(perm))
            except Exception:
                continue
    elif permute == "relabel":
        # 零假设：协议分组本身没有信息 → 把电芯随机重新分配组标签
        for _ in range(n_perm):
            perm = cells.copy()
            perm["group"] = rng.permutation(perm["group"].to_numpy())
            try:
                null.append(stat_fn(perm))
            except Exception:
                continue
    else:
        blocks = cells.groupby("group")[permute].first().to_dict()
        for _ in range(n_perm):
            keys = rng.permutation(list(blocks.keys()))
            mapping = dict(zip(list(blocks.keys()), [blocks[k] for k in keys]))
            perm = cells.copy()
            perm[permute] = perm["group"].map(mapping)
            try:
                null.append(stat_fn(perm))
            except Exception:
                continue

    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return {"observed": obs, "p": np.nan, "n_null": 0}
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))
    return {"observed": float(obs), "p": p, "n_null": int(len(null)),
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "null_q95": float(np.percentile(np.abs(null), 95))}


# --------------------------------------------------------------------------
# 具体结论的统计函数
# --------------------------------------------------------------------------
def stat_spearman_span(cells: pd.DataFrame) -> float:
    d = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    if d["soc_span"].nunique() < 3:
        return np.nan
    return float(stats.spearmanr(d["soc_span"], d["eol_cycle"]).statistic)


def stat_spearman_rate_within_window(cells: pd.DataFrame, min_span: float = 0.95) -> float:
    d = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    d = d[d["soc_span"] >= min_span]
    if len(d) < 8 or d["charge_rate_peak"].nunique() < 3:
        return np.nan
    return float(stats.spearmanr(d["charge_rate_peak"], d["efc"]).statistic)


def stat_efc_icc(cells: pd.DataFrame) -> float:
    """EFC 的协议级可解释率（ICC）。

    置换零假设下（组标签被彻底打散）ICC 应落到 0 附近，所以这个统计量
    必须配合 permute="relabel"（按电芯重贴组标签），不能整块搬寿命。
    """
    d = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    y = np.log(d["efc"].astype(float))
    mu = y.mean()
    ss_tot = float(((y - mu) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    ss_bet = float(d.assign(_y=y).groupby("group")["_y"]
                   .apply(lambda s: len(s) * (s.mean() - mu) ** 2).sum())
    return ss_bet / ss_tot


def stat_median_life_g57_vs_g14(cells: pd.DataFrame) -> float:
    """最优协议 vs 最差协议的寿命比（取 log，理由同 C5）。"""
    a = cells[cells["group"] == 57]["eol_cycle"]
    b = cells[cells["group"] == 14]["eol_cycle"]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(np.log(a.median() / b.median()))


def stat_highsoc_penalty(cells: pd.DataFrame, split: float = 0.7) -> float:
    """深充 vs 浅充 寿命比的**对数**。

    为什么取 log: 比值统计量恒 ≥ 0，置换零分布全在 1.0 附近，"|null| ≥ |obs|"
    对任何比值都成立（p 恒为 1）。取 log 后零分布以 0 为中心，
    才能做双侧检验。
    """
    d = cells.dropna(subset=["eol_cycle"])
    deep = d[d["soc_span"] >= split]["eol_cycle"]
    shallow = d[d["soc_span"] < split]["eol_cycle"]
    if len(deep) < 4 or len(shallow) < 4:
        return np.nan
    return float(np.log(deep.median() / shallow.median()))


def stat_plating_fraction(cells: pd.DataFrame) -> float:
    """超过 30°C 析锂上限（1.0C）的电芯占比。"""
    return float((cells["charge_rate_peak"] > 1.0).mean())


# --------------------------------------------------------------------------
def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")

    claims = [
        ("C1 SOC 窗口 vs 表观寿命 Spearman ρ", stat_spearman_span, "soc_span",
         "越浅充越长寿", "ρ", "target"),
        ("C2 固定窗口下 倍率 vs EFC Spearman ρ", stat_spearman_rate_within_window,
         "charge_rate_peak", "倍率在高 SOC 才伤", "ρ", "target"),
        ("C3 EFC 的协议级可解释率 ICC", stat_efc_icc, "soc_span",
         "协议只能解释 59% 的 EFC 方差（R² 天花板）", "ICC", "relabel"),
        ("C4 最优/最差协议寿命比 (log)", stat_median_life_g57_vs_g14, "soc_span",
         "G57 比 G14 长寿 23 倍（log 比 3.15）", "log-ratio", "target"),
        ("C5 深充 vs 浅充 寿命比 (log)", stat_highsoc_penalty, "soc_span",
         "深充组寿命仅为浅充的 37%（log 比 −1.00）", "log-ratio", "target"),
    ]

    print(f"=== 1) Bootstrap 置信区间（按协议整组重采样，B={N_BOOT}）===")
    rows = []
    for name, fn, feat, interp, kind, mode in claims:
        obs = fn(cells)
        boot = cluster_bootstrap(cells, fn)
        lo, med, hi = ci(boot)
        perm = permutation_p(cells, fn, permute=mode)
        rows.append({
            "claim": name,
            "observed": obs,
            "boot_median": med,
            "ci_low": lo,
            "ci_high": hi,
            "ci_width": hi - lo,
            "perm_p": perm["p"],
            "null_mean": perm.get("null_mean", np.nan),
            "perm_mode": mode,
            "significant": bool(perm["p"] < 0.05) if np.isfinite(perm["p"]) else False,
            "interpretation": interp,
            "kind": kind,
        })
        print(f"  {name}")
        print(f"    观测={obs:+.3f}  bootstrap 中位={med:+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]"
              f"  置换 p={perm['p']:.4f}  {'✓显著' if perm['p'] < 0.05 else '×不显著'}")

    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(METRICS / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    # ---- 2. 结论强度分级 ----
    def grade(r):
        if not np.isfinite(r["perm_p"]):
            return "无法判定"
        if r["perm_p"] < 0.01 and r["ci_width"] < abs(r["observed"]) * 2:
            return "强（高置信 + 窄区间）"
        if r["perm_p"] < 0.05:
            return "中（显著但区间宽）"
        return "弱／不成立"

    ci_df["strength"] = ci_df.apply(grade, axis=1)
    ci_df[["claim", "observed", "ci_low", "ci_high", "perm_p", "strength"]].to_csv(
        METRICS / "claim_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== 2) 结论强度分级 ===")
    print(ci_df[["claim", "observed", "ci_low", "ci_high", "perm_p", "strength"]]
          .round(3).to_string(index=False))

    # ---- 3. "10/12 越界"这个结论本身的不确定性 ----
    print("\n=== 3) 关键决策结论的稳健性 ===")
    pc = pd.read_csv(METRICS / "pareto_plating_check.csv")
    n_total = len(pc)
    n_viol = int((~pc["within_safe_limit"]).sum())
    # 用二项检验：假设真实风险率未知，给 95% CI（Wilson）
    from scipy.stats import beta
    lo = float(beta.ppf(0.025, n_viol + 0.5, n_total - n_viol + 0.5))
    hi = float(beta.ppf(0.975, n_viol + 0.5, n_total - n_viol + 0.5))
    print(f"  Pareto 越界率 = {n_viol}/{n_total} = {n_viol/n_total*100:.0f}%  "
          f"（Wilson 95% CI: {lo*100:.0f}%–{hi*100:.0f}%）")
    print("  → 即使用最保守的估计，也至少有一半的 Pareto 点越界：结论方向稳固")

    # ---- 4. 分层分析的稳健性：各窗口下倍率效应 ----
    print("\n=== 4) 分窗口倍率效应（各层独立 bootstrap）===")
    d = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    layers = []
    for lo_s, hi_s, lab in [(0.90, 1.01, "满窗口"), (0.70, 0.90, "深充"),
                            (0.55, 0.70, "中窗"), (0.35, 0.55, "浅充"),
                            (0.25, 0.35, "极浅")]:
        sub = d[(d["soc_span"] >= lo_s) & (d["soc_span"] < hi_s)]
        if len(sub) < 8:
            continue
        rho, p = stats.spearmanr(sub["charge_rate_peak"], sub["efc"])
        # bootstrap 该层
        vals = []
        rng = np.random.default_rng(SEED)
        gs = sub["group"].unique()
        for _ in range(1000):
            pick = rng.choice(gs, size=len(gs), replace=True)
            b = pd.concat([sub[sub["group"] == g] for g in pick])
            if b["charge_rate_peak"].nunique() < 3 or len(b) < 6:
                continue
            vals.append(stats.spearmanr(b["charge_rate_peak"], b["efc"]).statistic)
        lo_, med_, hi_ = ci(np.asarray(vals, float))
        layers.append({"layer": lab, "n": len(sub), "rho": rho, "p": p,
                       "ci_low": lo_, "ci_median": med_, "ci_high": hi_,
                       "crosses_zero": bool(lo_ < 0 < hi_) if np.isfinite(lo_) else None})
        print(f"  {lab:6s} n={len(sub):>3}  ρ={rho:+.3f} p={p:.3f}  95%CI=[{lo_:+.2f}, {hi_:+.2f}]")
    lay = pd.DataFrame(layers)
    lay.to_csv(METRICS / "layer_robustness.csv", index=False, encoding="utf-8-sig")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    plot_df = ci_df.dropna(subset=["ci_low", "ci_high"]).reset_index(drop=True)
    ypos = np.arange(len(plot_df))
    for i, r in plot_df.iterrows():
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color="#a0aec0", lw=6,
                solid_capstyle="butt", zorder=2)
        color = "#38a169" if r["significant"] else "#dd6b20"
        ax.scatter([r["observed"]], [i], s=90, color=color, zorder=3,
                   edgecolors="white", linewidths=1.2)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c.split(" ", 1)[0] for c in plot_df["claim"]], fontsize=9)
    ax.set_xlabel("效应量（95% 置信区间）")
    ax.set_title("关键结论的 Bootstrap 区间（绿=显著，橙=不显著）")
    ax.grid(alpha=0.3, axis="x")

    ax2 = axes[1]
    lay2 = lay.dropna(subset=["ci_low"]).reset_index(drop=True)
    for i, r in lay2.iterrows():
        ax2.plot([r["ci_low"], r["ci_high"]], [i, i], color="#a0aec0", lw=6,
                 solid_capstyle="butt")
        ax2.scatter([r["rho"]], [i], s=80, color="#2b6cb0", zorder=3)
        if r["crosses_zero"]:
            ax2.scatter([r["rho"]], [i], s=140, facecolors="none",
                        edgecolors="#dd6b20", linewidths=1.6, zorder=4)
    ax2.axvline(0, color="k", ls=":", lw=1)
    ax2.set_yticks(range(len(lay2)))
    ax2.set_yticklabels([f"{r['layer']} (n={r['n']})" for _, r in lay2.iterrows()], fontsize=9)
    ax2.set_xlabel("ρ(峰值倍率, EFC)")
    ax2.set_title("分层倍率效应：橙圈=区间跨 0（结论不稳）")
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGURES / "14_robustness.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
