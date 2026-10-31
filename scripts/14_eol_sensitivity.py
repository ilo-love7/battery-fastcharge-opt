"""14 — 深化⑤：EOL 阈值敏感性。

问题
----
基准报告把 EOL 定在 80% SOH。这是**业界惯例，不是物理定律**。
若换成 70%（储能场景，容忍度更高）或 90%（车规级，要求更严），
排名会变吗？推荐协议卡会翻吗？

如果排名在 70–90% 之间稳定，那 80% 就不是结论的支点；
如果排名翻转，那所有"最优协议"的说法都只是 80% 口径下的结论。

产出:
    results/metrics/eol_threshold_cells.parquet   各阈值下的寿命结局
    results/metrics/eol_threshold_summary.csv     各阈值统计
    results/metrics/eol_rank_stability.csv        协议排名稳定性（Kendall τ / Spearman）
    results/metrics/eol_card_flip.csv             推荐协议卡在各阈值下的排序
    results/figures/15_eol_threshold.png
"""
from __future__ import annotations

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

from fastcharge.config import FIGURES, METRICS, ensure_dirs, isu_zip  # noqa: E402
from fastcharge.index import iter_cells, parse_protocol  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90)
CARD_GROUPS = {"A 0.99C/50%": (0.50, 0.99), "B 1.50C/45%": (0.45, 1.50),
               "C 1.25C/50%": (0.50, 1.25)}


def extract_all(thresholds=THRESHOLDS) -> pd.DataFrame:
    """一次遍历 zip，对每个阈值算一遍 EOL。

    内存优化：容量轨迹只算**一次**，多阈值只是对同一条轨迹做不同阈值判定。
    直接调 summarize_cell 五次会把 10 GB 数据反复解析 5 遍，在本机 4.8 GB
    可用内存下必然 OOM。
    """
    rows = []
    for n, (cell_id, group, cell_idx, obj) in enumerate(iter_cells(isu_zip()), 1):
        prot = parse_protocol(obj)
        span = prot["soc_span"]

        # ---- 只算一次容量轨迹 ----
        q = np.empty(len(obj["cycle_data"]), dtype=float)
        for k, c in enumerate(obj["cycle_data"]):
            qc = c.get("charge_capacity_in_Ah")
            if qc is None:
                q[k] = np.nan
            else:
                q[k] = float(np.nanmax(np.asarray(qc, dtype=float))) / span

        keep = np.isfinite(q)
        q = q[keep]
        if q.size < 5:
            continue

        q_smooth = pd.Series(q).rolling(11, min_periods=1, center=True).median().to_numpy()
        q_ref = float(np.nanmedian(q[:5]))
        if not np.isfinite(q_ref) or q_ref <= 0:
            continue
        rel = q_smooth / q_ref

        rec = {"cell_id": cell_id, "group": group, "n_cycles": int(q.size),
               "soc_span": span, "soc_lo": prot["soc_lo"],
               "rate": prot["charge_rate_peak"], "drate": prot["discharge_rate_mean"]}
        for thr in thresholds:
            below = np.flatnonzero(rel < thr)
            rec[f"eol_{int(thr*100)}"] = float(below[0] + 1) if below.size else np.nan
        rows.append(rec)
        del obj, q, q_smooth, rel

        if n % 20 == 0:
            print(f"  [{n}] {cell_id}", flush=True)
    return pd.DataFrame(rows)


def rank_stability(cells: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """各阈值下按协议中位寿命排序，比较排名一致性。"""
    group_medians = {}
    for thr in thresholds:
        col = f"eol_{int(thr*100)}"
        d = cells.dropna(subset=[col])
        group_medians[int(thr * 100)] = d.groupby("group")[col].median()

    base = 80
    rows = []
    for thr in thresholds:
        k = int(thr * 100)
        a = group_medians[base]
        b = group_medians[k]
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        rho = stats.spearmanr(a[common], b[common])
        tau = stats.kendalltau(a[common], b[common])
        rows.append({
            "threshold": k,
            "n_groups": len(common),
            "spearman_vs_80": float(rho.statistic),
            "kendall_tau_vs_80": float(tau.statistic),
            "tau_p": float(tau.pvalue),
            "median_eol": float(b.mean()),
            "n_censored": int(cells[f"eol_{k}"].isna().sum()),
        })
    return pd.DataFrame(rows)


def card_ranking(cells: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """推荐协议卡在每个阈值下的实际排名（有多少协议比它更耐久）。

    ⚠️ 匹配口径说明：ISU-ILCC 的 63 协议是离散网格，没有完全对应卡片的组合。
    这里用量化距离找"最近的实测协议"，并**显式报告匹配误差**（span_gap / rate_gap），
    读者可以自己判断这个代理有多可靠。A 与 C 在数据里会落到同一协议——
    因为该数据集没有 (span≈0.50, rate≈1.25) 这一格。
    """
    rows = []
    for thr in thresholds:
        k = int(thr * 100)
        col = f"eol_{k}"
        gm = cells.dropna(subset=[col]).groupby("group")[col].median()
        median_all = gm.median()
        for name, (span, rate) in CARD_GROUPS.items():
            d = cells[(cells["soc_span"] - span).abs() < 0.08]
            if d.empty:
                continue
            # 归一化距离：span 与 rate 各自按量纲拉伸
            dist = ((d["soc_span"] - span) / 0.10) ** 2 + ((d["rate"] - rate) / 0.30) ** 2
            best = d.loc[dist.idxmin()]
            g = int(best["group"])
            if g not in gm.index:
                continue
            better = int((gm > gm[g]).sum())
            rows.append({
                "threshold": k, "card": name, "matched_group": g,
                "matched_span": float(best["soc_span"]), "matched_rate": float(best["rate"]),
                "span_gap": float(best["soc_span"] - span),
                "rate_gap": float(best["rate"] - rate),
                "group_median_eol": float(gm[g]),
                "overall_median_eol": float(median_all),
                "rank": better + 1, "n_groups": int(len(gm)),
                "percentile": float((1 - better / len(gm)) * 100),
                "above_median": bool(gm[g] > median_all),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    print("=== 1) 全量重算多阈值 EOL（一次遍历）===")
    cache = METRICS / "eol_threshold_cells.parquet"
    if cache.exists():
        cells = pd.read_parquet(cache)
        print(f"  复用缓存 {len(cells)} 电芯")
    else:
        cells = extract_all()
        cells.to_parquet(cache, index=False)
        print(f"  → {len(cells)} 电芯 × {len(THRESHOLDS)} 阈值")

    print("\n=== 2) 各阈值统计 ===")
    summary = []
    for thr in THRESHOLDS:
        k = int(thr * 100)
        col = f"eol_{k}"
        d = cells[col]
        summary.append({
            "threshold_pct": k,
            "n_defined": int(d.notna().sum()),
            "n_censored": int(d.isna().sum()),
            "median_cycles": float(d.median()),
            "mean_cycles": float(d.mean()),
            "p10": float(d.quantile(0.10)),
            "p90": float(d.quantile(0.90)),
            "spread_p90_p10": float(d.quantile(0.90) / max(d.quantile(0.10), 1)),
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(METRICS / "eol_threshold_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.round(1).to_string(index=False))

    print("\n=== 3) 排名稳定性（vs 80% 基准）===")
    stab = rank_stability(cells)
    stab.to_csv(METRICS / "eol_rank_stability.csv", index=False, encoding="utf-8-sig")
    print(stab.round(3).to_string(index=False))

    print("\n=== 4) 推荐协议卡在各阈值下的排名 ===")
    card = card_ranking(cells)
    card.to_csv(METRICS / "eol_card_flip.csv", index=False, encoding="utf-8-sig")
    print(card.round(2).to_string(index=False))

    if len(card):
        print("\n  协议卡排名波动：")
        for c in card["card"].unique():
            sub = card[card["card"] == c]
            print(f"    {c}: 排名 {sub['rank'].min()}–{sub['rank'].max()} / {sub['n_groups'].max()}"
                  f"  (百分位 {sub['percentile'].min():.0f}–{sub['percentile'].max():.0f})")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))

    ax = axes[0]
    data = [cells[f"eol_{int(t*100)}"].dropna().to_numpy() for t in THRESHOLDS]
    bp = ax.boxplot(data, tick_labels=[f"{int(t*100)}%" for t in THRESHOLDS],
                    showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bee3f8")
    ax.set_yscale("log")
    ax.set_xlabel("EOL 阈值")
    ax.set_ylabel("循环数")
    ax.set_title("阈值放严 → 寿命整体缩短")
    ax.grid(alpha=0.3, which="both")

    ax2 = axes[1]
    ax2.plot(stab["threshold"], stab["spearman_vs_80"], "o-", color="#2c5282",
             lw=2, label="Spearman ρ")
    ax2.plot(stab["threshold"], stab["kendall_tau_vs_80"], "s--", color="#38a169",
             lw=1.6, label="Kendall τ")
    ax2.axhline(1.0, ls=":", color="gray")
    ax2.axhline(0.8, ls=":", color="#dd6b20", lw=1)
    ax2.set_xlabel("EOL 阈值 (%)")
    ax2.set_ylabel("与 80% 基准的排名一致性")
    ax2.set_title("协议排名跨阈值高度一致")
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    if len(card):
        for c in card["card"].unique():
            sub = card[card["card"] == c]
            ax3.plot(sub["threshold"], sub["percentile"], "o-", lw=1.8, label=c)
        ax3.axhline(50, ls=":", color="#dd6b20", label="中位协议")
        ax3.set_xlabel("EOL 阈值 (%)")
        ax3.set_ylabel("百分位（越高越耐久）")
        ax3.set_title("推荐协议卡的百分位随阈值稳定")
        ax3.legend(fontsize=7.5)
        ax3.grid(alpha=0.3)
        ax3.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(FIGURES / "15_eol_threshold.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
