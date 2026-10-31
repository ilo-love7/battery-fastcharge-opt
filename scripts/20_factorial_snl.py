"""分层对照实验：用 Sandia SNL 的因子化设计解决"倍率效应是否显著"的争议。

为什么这个数据集能解决问题
--------------------------
v1.1 §4 的结论是"倍率效应 perm p=0.17，不稳健"，根因是 ISU-ILCC 的倍率与 SOC 窗口
**严重混淆**——63 个协议里没有干净的"同窗口、只变倍率"对照。

Sandia SNL 是**因子化设计**：文件名编码 `化学_温度_窗口_充/放电倍率`，
同一个 (化学, 温度, 窗口) 下有 0.5/1/2/3C 多档放电倍率，且**充电倍率恒定 0.5C**。
→ 这是天然的双因素方差分析（two-way ANOVA）设计，倍率效应可干净分离。

设计矩阵（本数据集实际覆盖）
----------------------------
    化学   : LFP / NCA / NMC
    温度   : 15 / 25 / 35 °C
    窗口   : 0–100% / 20–80%
    放电倍率: 0.5 / 1 / 2 / 3 C
    充电倍率: 恒定 0.5C（关键！倍率效应不含充电侧混淆）

产出
----
    results_extra/snl/cells.parquet         Sandia 电芯表
    results_extra/snl/anova_rate.csv        倍率主效应与交互（双因素 ANOVA）
    results_extra/snl/rate_effect_clean.csv 干净对照下的倍率-寿命曲线
    results_extra/snl/cross_chemistry.csv   三化学体系对照
    figures/20_snl_factorial.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fastcharge.config import FIGURES, RESULTS, ensure_dirs  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

SNL_ZIP = Path(r"D:\项目\02_数据集\01_国际公开集\Sandia_SNL\SNL.zip")
OUT = RESULTS / "snl"
OUT.mkdir(parents=True, exist_ok=True)

#: 文件名 → 因子。例: SNL_18650_NMC_25C_0-100_0.5-1C_a
_RE = re.compile(
    r"SNL_18650_(?P<chem>LFP|NCA|NMC)_(?P<temp>\d+)C_"
    r"(?P<lo>\d+)-(?P<hi>\d+)_(?P<chg>[0-9.]+)-(?P<dis>[0-9.]+)C_(?P<rep>[a-z])\.pkl"
)

EOL_THRESHOLD = 0.80
SMOOTH = 11


def parse_name(name: str) -> dict:
    m = _RE.search(name)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "chem": d["chem"],
        "temp_C": float(d["temp"]),
        "soc_lo": float(d["lo"]) / 100,
        "soc_hi": float(d["hi"]) / 100,
        "charge_rate": float(d["chg"]),
        "discharge_rate": float(d["dis"]),
        "replicate": d["rep"],
    }


def eol_from_cycles(cycles: list, threshold: float = EOL_THRESHOLD) -> dict:
    """从 cycle_data 抽容量轨迹并判定 EOL，同时算**容量衰减率**。

    为什么要加衰减率
    ---------------
    EOL 循环数受"测试何时停止"影响（删失），而**容量衰减率 /1000 圈**
    只取决于容量轨迹本身，对删失不敏感。两者一起看才能判断
    "高倍率是让电池更早报废，还是只是让每次循环损耗更多"。
    """
    q = np.empty(len(cycles), dtype=float)
    for k, c in enumerate(cycles):
        qc = c.get("charge_capacity_in_Ah")
        if qc is None:
            q[k] = np.nan
        else:
            q[k] = float(np.nanmax(np.asarray(qc, dtype=float)))
    ok = np.isfinite(q)
    q = q[ok]
    if q.size < 10:
        return {"n_cycles": int(q.size), "q_ref": np.nan, "eol_cycle": np.nan,
                "censored": True, "soh_final": np.nan, "fade_per_1k": np.nan}
    sm = pd.Series(q).rolling(SMOOTH, min_periods=1, center=True).median().to_numpy()
    q_ref = float(np.nanmedian(q[:5]))
    if q_ref <= 0:
        return {"n_cycles": int(q.size), "q_ref": np.nan, "eol_cycle": np.nan,
                "censored": True, "soh_final": np.nan, "fade_per_1k": np.nan}
    below = np.flatnonzero(sm / q_ref < threshold)

    # 衰减率：SOH 对循环数做线性拟合，取斜率（每 1000 圈的百分点损失）
    k = np.arange(q.size, dtype=float)
    soh = sm / q_ref
    if q.size >= 20:
        slope = float(np.polyfit(k, soh, 1)[0])
        fade_per_1k = -slope * 1000
    else:
        fade_per_1k = np.nan

    return {
        "n_cycles": int(q.size),
        "q_ref": q_ref,
        "eol_cycle": float(below[0] + 1) if below.size else np.nan,
        "censored": bool(below.size == 0),
        "soh_final": float(np.nanmedian(sm[-5:]) / q_ref),
        "fade_per_1k": fade_per_1k,
    }


def build_cells() -> pd.DataFrame:
    import pickle
    import zipfile

    rows = []
    with zipfile.ZipFile(SNL_ZIP) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".pkl"):
                continue
            meta = parse_name(Path(name).name)
            if not meta:
                continue
            with zf.open(name) as fh:
                obj = pickle.loads(fh.read())
            eol = eol_from_cycles(obj["cycle_data"])
            rec = {**meta, "cell_id": Path(name).stem,
                   "nominal_Ah": obj.get("nominal_capacity_in_Ah")}
            rec.update(eol)
            rows.append(rec)
            del obj
            print(f"  {Path(name).name:46s} n={rec['n_cycles']:5d} "
                  f"EOL={rec['eol_cycle']}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 双因素 ANOVA（手工实现，不引入 statsmodels 依赖）
# --------------------------------------------------------------------------
def two_way_anova(df: pd.DataFrame, factor_a: str, factor_b: str,
                  value: str) -> dict:
    """**不平衡设计**的双因素 ANOVA（回归型，Type III 主效应 + 交互）。

    为什么不用平衡设计的方差分解
    ----------------------------
    Sandia 是**不平衡**的因子设计（NMC 无 3C、NCA 无 3C、35°C 无 0.5C/3C）。
    用平衡公式 `SS_AB = SS_cells - SS_A - SS_B` 会得到负 SS（实测交互 F=-9.39），
    因为不平衡时组均值投影不满足正交分解。

    正确做法：用**线性模型**（含主效应 + 交互哑变量）做嵌套 F 检验：
        完整模型: y ~ A + B + A:B
        无交互:   y ~ A + B
        F_AB = ((RSS_reduced - RSS_full) / df_AB) / (RSS_full / df_err)
    主效应同理（对每个主效应，从完整模型里删掉它）。
    """
    d = df.dropna(subset=[value, factor_a, factor_b]).copy()
    n = len(d)
    if n < 6:
        return {"note": "样本不足", "n": n}

    y = d[value].to_numpy(float)

    def design(fa: list[str]) -> np.ndarray:
        """构造设计矩阵：截距 + 各因子哑变量（丢基线）+ 交互。"""
        cols = [np.ones(n)]
        for f in fa:
            lv = pd.get_dummies(d[f].astype(str), drop_first=True, dtype=float)
            for c in lv.columns:
                cols.append(lv[c].to_numpy(float))
        if len(fa) == 2:
            a_lv = pd.get_dummies(d[fa[0]].astype(str), drop_first=True, dtype=float)
            b_lv = pd.get_dummies(d[fa[1]].astype(str), drop_first=True, dtype=float)
            for ca in a_lv.columns:
                for cb in b_lv.columns:
                    cols.append((a_lv[ca] * b_lv[cb]).to_numpy(float))
        return np.column_stack(cols)

    def rss(X: np.ndarray) -> tuple[float, int]:
        if X.shape[1] >= n:
            return np.nan, 0
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(resid @ resid), int(n - X.shape[1])

    X_full = design([factor_a, factor_b])
    rss_f, df_f = rss(X_full)
    if not np.isfinite(rss_f) or df_f <= 0:
        return {"note": "自由度不足（样本 < 参数数）", "n": n, "n_params": X_full.shape[1]}

    ms_err = rss_f / df_f
    out = {"n": n, "df_err": df_f, "rss_full": rss_f, "ms_err": ms_err,
           "n_params": X_full.shape[1]}

    # ---- 交互项 ----
    X_no_ab = design([factor_a, factor_b])[:, :1 + (d[factor_a].nunique() - 1) + (d[factor_b].nunique() - 1)]
    rss_r, _ = rss(X_no_ab)
    df_ab = X_full.shape[1] - X_no_ab.shape[1]
    if df_ab > 0 and np.isfinite(rss_r) and (rss_r - rss_f) >= -1e-9:
        ss_ab = max(rss_r - rss_f, 0.0)
        f_ab = (ss_ab / df_ab) / ms_err
        out.update({"ss_ab": ss_ab, "df_ab": df_ab, "f_ab": float(f_ab),
                    "p_ab": float(1 - stats.f.cdf(f_ab, df_ab, df_f))})
    else:
        out.update({"ss_ab": np.nan, "df_ab": 0, "f_ab": np.nan, "p_ab": np.nan})

    # ---- 主效应（从完整模型删掉该因子）----
    for tag, keep in (("a", [factor_b]), ("b", [factor_a])):
        X_red = design(keep)
        rss_r, _ = rss(X_red)
        df_eff = X_full.shape[1] - X_red.shape[1]
        if df_eff > 0 and np.isfinite(rss_r) and (rss_r - rss_f) >= -1e-9:
            ss = max(rss_r - rss_f, 0.0)
            f = (ss / df_eff) / ms_err
            out.update({f"ss_{tag}": ss, f"df_{tag}": df_eff, f"f_{tag}": float(f),
                        f"p_{tag}": float(1 - stats.f.cdf(f, df_eff, df_f))})
        else:
            out.update({f"ss_{tag}": np.nan, f"df_{tag}": 0, f"f_{tag}": np.nan,
                        f"p_{tag}": np.nan})

    # ---- 效应量 η²（用完整模型的总 SS）----
    ss_total = float(((y - y.mean()) ** 2).sum())
    for tag in ("a", "b", "ab"):
        v = out.get(f"ss_{tag}", np.nan)
        out[f"eta2_{tag}"] = float(v / ss_total) if (ss_total > 0 and np.isfinite(v)) else np.nan
    out["eta2_err"] = 1 - (out["eta2_a"] + out["eta2_b"] + out.get("eta2_ab", 0))
    return out


def survival_endpoint(cells: pd.DataFrame) -> pd.DataFrame:
    """**删失感知**的应变量：用"测试结束时达到的等效满循环数"。

    为什么不能用 eol_cycle
    ---------------------
    Sandia 61 只里有 25 只删失（LFP 12/18、NMC 14/21、NMC 1C 4/4 全删失）。
    直接对 eol_cycle 做 ANOVA 会：
        ① 丢掉 25 只（有效样本只剩 35）
        ② 系统性偏向"短寿命组"——因为长寿命的还在跑，没有 EOL

    正确做法：对每只电芯都用 **它实际跑到的循环数**（不论是否跌破 80%），
    这是"至少存活这么久"的保守下界（右删失数据的标准处理）。
    这样 61 只全部进入分析，方向不被删失结构扭曲。

    同时保留 eol_cycle 供对照，并在报告里标注两种口径的结论差异。
    """
    c = cells.copy()
    c["survival_cycles"] = c["n_cycles"].astype(float)
    c["log_survival"] = np.log(c["survival_cycles"].clip(lower=1))
    # 同时在跑时间上做归一化：n_cycles × 每圈吞吐（窗口宽度）
    c["throughput"] = c["survival_cycles"] * (c["soc_hi"] - c["soc_lo"])
    c["log_throughput"] = np.log(c["throughput"].clip(lower=1))
    return c


def clean_rate_effect(cells: pd.DataFrame) -> pd.DataFrame:
    """干净对照：固定 (化学, 温度, 窗口)，只变放电倍率。

    用**存活循环数**（删失感知）而非 eol_cycle，否则长寿命组会被系统性排除。
    每个格子只有 1–4 只电芯，如实标注组数——**不夸大样本量**。
    """
    d = cells.copy()
    d["window"] = d["soc_lo"].astype(str) + "-" + d["soc_hi"].astype(str)
    rows = []
    for (chem, temp, win), g in d.groupby(["chem", "temp_C", "window"], observed=True):
        rates = g["discharge_rate"].nunique()
        if rates < 2:
            continue
        for rate, gg in g.groupby("discharge_rate"):
            rows.append({
                "chem": chem, "temp_C": temp, "window": win,
                "discharge_rate": rate, "n_stages": rates,
                "n": len(gg),
                "surv_median": float(gg["survival_cycles"].median()),
                "surv_min": float(gg["survival_cycles"].min()),
                "surv_max": float(gg["survival_cycles"].max()),
                "eol_median": float(gg["eol_cycle"].median()) if gg["eol_cycle"].notna().any() else np.nan,
                "censored": int(gg["censored"].sum()),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["baseline_median"] = out.groupby(["chem", "temp_C", "window"])["surv_median"].transform("min")
    out["life_ratio"] = out["surv_median"] / out["baseline_median"]
    out["life_delta_pct"] = (out["life_ratio"] - 1) * 100
    return out


def main() -> int:
    ensure_dirs()
    cache = OUT / "cells.parquet"
    if cache.exists():
        cells = pd.read_parquet(cache)
        print(f"复用缓存 {len(cells)} 电芯")
    else:
        print("=== 1) 解析 Sandia 全量 ===")
        cells = build_cells()
        cells.to_parquet(cache, index=False)
        print(f"\n  → {len(cells)} 电芯")

    print(f"\n  有效寿命结局: {cells['eol_cycle'].notna().sum()} / {len(cells)}"
          f"（删失 {int(cells['censored'].sum())}）")
    print("\n  **删失结构（这是必须换应变量的原因）**:")
    print("  按化学:"); print(pd.crosstab(cells["chem"], cells["censored"]).to_string())
    print("\n  按倍率:"); print(pd.crosstab(cells["discharge_rate"], cells["censored"]).to_string())
    print("\n  设计矩阵:")
    print(pd.crosstab(cells["chem"], cells["discharge_rate"]).to_string())
    print("\n  温度 × 倍率:")
    print(pd.crosstab(cells["temp_C"], cells["discharge_rate"]).to_string())

    cells = survival_endpoint(cells)
    cells.to_parquet(cache, index=False)

    # ---- 2. 双因素 ANOVA ----
    print("\n=== 2) 双因素 ANOVA（不平衡设计，回归型）===")
    print("   ★ 主分析: 应变量 = **容量衰减率 /1000 圈**（对删失最不敏感）")
    print("   对照:   log(存活循环数) / log(EOL 循环数)")

    anova_rows = []
    for ycol, ylabel in [
        ("fade_per_1k", "容量衰减率/1000圈 ★主分析"),
        ("log_survival", "存活循环数（删失感知）"),
        ("log_eol", "EOL 循环数（仅未删失）"),
    ]:
        if ycol == "log_eol":
            d = cells.dropna(subset=["eol_cycle"]).copy()
            d["log_eol"] = np.log(d["eol_cycle"].clip(lower=1))
        else:
            d = cells.copy()
        n_avail = d[ycol].notna().sum() if ycol in d else 0
        print(f"\n  ── 应变量: {ylabel} (可用 n={n_avail}) ──")
        for a, b, label in [
            ("discharge_rate", "chem", "倍率 × 化学"),
            ("discharge_rate", "temp_C", "倍率 × 温度"),
        ]:
            r = two_way_anova(d, a, b, ycol)
            r.update({"factor_a": a, "factor_b": b, "label": label, "response": ylabel})
            anova_rows.append(r)
            if "f_a" in r and np.isfinite(r["f_a"]):
                print(f"  [{label}] n={r['n']} 参数={r['n_params']} df_err={r['df_err']}")
                print(f"    A({a:14s}) F={r['f_a']:6.2f}  p={r['p_a']:.4f}  η²={r['eta2_a']:.3f}")
                print(f"    B({b:14s}) F={r['f_b']:6.2f}  p={r['p_b']:.4f}  η²={r['eta2_b']:.3f}")
                ab = r.get("f_ab", np.nan)
                if np.isfinite(ab):
                    print(f"    A×B             F={ab:6.2f}  p={r['p_ab']:.4f}")
                else:
                    print("    A×B             无法估计")
            else:
                print(f"  [{label}] 无法估计: {r.get('note','')} (n={r.get('n')}, 参数={r.get('n_params')})")
    anova = pd.DataFrame(anova_rows)
    anova.to_csv(OUT / "anova_rate.csv", index=False, encoding="utf-8-sig")

    # ---- 3. 干净对照 ----
    print("\n=== 3) 干净对照：固定化学/温度/窗口，只变倍率 ===")
    eff = clean_rate_effect(cells)
    eff.to_csv(OUT / "rate_effect_clean.csv", index=False, encoding="utf-8-sig")
    if not eff.empty:
        print(eff.round(3).to_string(index=False))
        # 汇总：各倍率相对基线的寿命比
        agg = eff.groupby("discharge_rate").agg(
            n_sets=("n", "size"), life_ratio_median=("life_ratio", "median"),
            life_ratio_mean=("life_ratio", "mean"),
            delta_pct_median=("life_delta_pct", "median"),
        ).reset_index()
        print("\n  各倍率相对同组最低倍率的寿命比:")
        print(agg.round(3).to_string(index=False))
        agg.to_csv(OUT / "rate_effect_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 4. 跨化学 ----
    print("\n=== 4) 化学体系差异（固定倍率 1C、温度 25°C、满窗口）===")
    sub = cells[(cells["discharge_rate"] == 1.0) & (cells["temp_C"] == 25.0)
                & (cells["soc_hi"] == 1.0)]
    by_chem = sub.groupby("chem").agg(
        n=("cell_id", "size"),
        surv_median=("survival_cycles", "median"),
        surv_min=("survival_cycles", "min"),
        surv_max=("survival_cycles", "max"),
        eol_median=("eol_cycle", "median"),
        censored=("censored", "sum"),
    ).reset_index()
    print(by_chem.round(1).to_string(index=False))
    print("\n  ⚠️ NMC 该格 4 只全删失 → 只有'至少活到 N 圈'的下界，没有 EOL")
    by_chem.to_csv(OUT / "cross_chemistry.csv", index=False, encoding="utf-8-sig")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    ax = axes[0]
    if not eff.empty:
        for chem in eff["chem"].unique():
            s = eff[eff["chem"] == chem]
            ax.plot(s["discharge_rate"], s["life_ratio"], "o-", lw=1.6, label=chem, alpha=0.85)
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("放电倍率 (C)")
        ax.set_ylabel("寿命 / 同组最低倍率寿命")
        ax.set_title("干净对照：倍率效应（固定化学·温度·窗口）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax2 = axes[1]
    d2 = cells.copy()
    for chem, marker in [("LFP", "o"), ("NCA", "s"), ("NMC", "^")]:
        s = d2[d2["chem"] == chem]
        if s.empty:
            continue
        off = {"LFP": -0.05, "NCA": 0, "NMC": 0.05}[chem]
        ax2.scatter(s["discharge_rate"] + off, s["survival_cycles"], s=32, marker=marker,
                    alpha=0.7, label=f"{chem} (n={len(s)})")
    ax2.set_yscale("log")
    ax2.set_xlabel("放电倍率 (C)")
    ax2.set_ylabel("存活循环数（删失感知）")
    ax2.set_title(f"三化学体系全样本（n={len(d2)}，含删失）")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(FIGURES / "20_snl_factorial.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
