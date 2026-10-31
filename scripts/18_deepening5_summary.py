"""18 — 第五轮深化汇总（EOL 阈值 / 模型敏感性 / 老化漂移 / 温度协议表）。

产出:
    results/metrics/deepening5_summary.csv
    docs/figures/main_deepening5.png
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

from fastcharge.config import FIGURES, METRICS  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
warnings.filterwarnings("ignore")

OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    rows = []

    # ⑤ EOL 阈值
    stab = pd.read_csv(METRICS / "eol_rank_stability.csv")
    summ = pd.read_csv(METRICS / "eol_threshold_summary.csv")
    r70 = stab[stab["threshold"] == 70]
    r90 = stab[stab["threshold"] == 90]
    rows.append({
        "深化": "⑤ EOL 阈值敏感性",
        "结论": f"70–85% 区间协议排名几乎不变（Spearman ≥ {stab[stab['threshold']<=85]['spearman_vs_80'].min():.3f}）；"
                f"90% 才开始松动（ρ={r90['spearman_vs_80'].iloc[0]:.3f}）",
        "关键数字": f"中位寿命 {summ['median_cycles'].min():.0f}–{summ['median_cycles'].max():.0f} 循环"
                    f"（70%→90% 差 {summ['median_cycles'].max()/summ['median_cycles'].min():.1f} 倍）",
        "强度": "强（排名结论与阈值无关）",
    })

    # ⑥ 模型敏感性
    lim = pd.read_csv(METRICS / "safe_limit_by_model.csv")
    piv = lim.pivot(index="temperature_C", columns="model", values="safe_rate_limit_C")
    d30 = piv.loc[30.0] if 30.0 in piv.index else None
    agr = pd.read_csv(METRICS / "model_agreement.csv")
    rows.append({
        "深化": "⑥ 析锂模型敏感性",
        "结论": f"30°C 安全上限 SPM={d30['SPM']:.2f}C vs DFN={d30['DFN']:.2f}C"
                f" → **SPM 反而更保守**（与预期相反）",
        "关键数字": f"三模型一致判定 {int(agr['unanimous'].sum())}/{len(agr)} 个倍率点",
        "强度": "中（结论方向一致，幅度有分歧）",
    })

    # ⑦ 老化漂移
    dr = pd.read_csv(METRICS / "charge_time_drift.csv")
    rows.append({
        "深化": "⑦ 充电时长老化漂移",
        "结论": f"归一化 τ（每 Ah 时间）中位 +{(dr['tau_ratio'].median()-1)*100:.0f}%"
                f"（内阻恶化），但绝对时长 −{(1-dr['t_ratio'].median())*100:.0f}%"
                f"（电荷量 −{(1-dr['q_ratio'].median())*100:.0f}% 更狠）",
        "关键数字": f"70→80% SOH 期内阻 +{dr['tau_ratio'].quantile(0.9)*100-100:.0f}%（P90）",
        "强度": "强（两种口径同时可算，逻辑自洽）",
    })

    # ⑧ 温度协议表
    tbl = pd.read_csv(METRICS / "protocol_table_by_temp.csv")
    pre = pd.read_csv(METRICS / "winter_summer_presets.csv")
    comp = pd.read_csv(METRICS / "compliance_check.csv")
    rows.append({
        "深化": "⑧ 温度自适应协议表",
        "结论": f"5 个温区 × 4 个窗口 = {len(tbl)} 格协议表；"
                f"冬季上限 {pre['max_rate_C'].iloc[0]:.2f}C、夏季 {pre['max_rate_C'].iloc[-1]:.2f}C",
        "关键数字": f"实测协议在常温下仅 {comp[comp['temp_bin'].str.contains('常温')]['compliant_frac'].mean()*100:.0f}% 合规",
        "强度": "强（工程交付物）",
    })

    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "deepening5_summary.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    # ---- 汇总图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    # (a) EOL 阈值
    ax = axes[0, 0]
    ax.plot(stab["threshold"], stab["spearman_vs_80"], "o-", color="#2c5282", lw=2,
            label="Spearman ρ")
    ax.plot(stab["threshold"], stab["kendall_tau_vs_80"], "s--", color="#38a169", lw=1.6,
            label="Kendall τ")
    ax.axhline(0.8, ls=":", color="#dd6b20", lw=1)
    ax.set_ylim(0.6, 1.05)
    ax.set_xlabel("EOL 阈值 (%)")
    ax.set_ylabel("与 80% 基准的排名一致性")
    ax.set_title("⑤ 排名对阈值不敏感")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 模型敏感性
    ax2 = axes[0, 1]
    t30 = pd.read_csv(METRICS / "eta_curves_30C.csv")
    colors = {"SPM": "#2b6cb0", "SPMe": "#38a169", "DFN": "#dd6b20"}
    for m in ("SPM", "SPMe", "DFN"):
        if m in t30.columns:
            ax2.plot(t30["c_rate"], t30[m], "o-", color=colors[m], lw=1.8, label=m)
    ax2.axhline(0, color="red", ls="--", lw=1)
    ax2.axhline(0.005, color="orange", ls=":", lw=1)
    ax2.set_xlabel("充电倍率 (C)")
    ax2.set_ylabel("min η_pl (V)")
    ax2.set_title("⑥ 三模型析锂判据（30°C）")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (c) 老化漂移
    ax3 = axes[1, 0]
    dr2 = dr.dropna(subset=["tau_ratio", "t_ratio"])
    ax3.scatter(dr2["tau_ratio"], dr2["t_ratio"], s=20, alpha=0.6, c="#2b6cb0")
    ax3.axhline(1.0, color="gray", ls=":", lw=1)
    ax3.axvline(1.0, color="gray", ls=":", lw=1)
    lim_ = [0.9, 1.25]
    ax3.plot(lim_, lim_, "k--", lw=0.8, label="等时长线")
    ax3.set_xlabel("τ_late/τ_early（每 Ah 时间比，内阻）")
    ax3.set_ylabel("t_late/t_early（绝对时长比）")
    ax3.set_title("⑦ 内阻恶化 vs 绝对时长：两者反向")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # (d) 温度协议表
    ax4 = axes[1, 1]
    pt = tbl.pivot(index="temp_bin", columns="soc_window", values="recommended_rate_C")
    for i, span in enumerate(pt.columns):
        ax4.plot(range(len(pt)), pt[span].to_numpy(float), "o-", lw=1.8,
                 label=f"窗口 {span:.0%}")
    ax4.set_xticks(range(len(pt)))
    ax4.set_xticklabels([b.split(" ")[0] for b in pt.index], fontsize=8)
    ax4.set_xlabel("温度分档")
    ax4.set_ylabel("推荐最大倍率 (C)")
    ax4.set_title("⑧ 温度自适应协议表")
    ax4.legend(fontsize=7.5)
    ax4.grid(alpha=0.3)

    fig.suptitle("第五轮深化：四个敏感性检验", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "main_deepening5.png", bbox_inches="tight")
    fig.savefig(FIGURES / "19_deepening5_summary.png")
    plt.close(fig)
    print(f"\n图已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
