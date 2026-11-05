"""日历老化与循环老化的分离（用 Sandia SNL 的逐圈时长重建时间轴）。

为什么这个 DataSet 能分离
-------------------------
Sandia 的每圈 `time_in_s` 是**真实测试时长**（含充、放电、静置），
把全部历圈时长累加即得**日历时间轴**（无需外部时间戳）。

于是有两条正交的自变量：
    循环数   N   —— 循环老化
    日历时间 t   —— 日历老化（含静置期的 SEI 生长）

对同一只电芯，容量损失可以拟合成：
    ΔQ/Q0 = a·N^p + b·√t          （循环项 + 日历项，SEI 通常 ~√t）
两个系数 per-cell 分别估计，然后看：
    · a（循环系数）随倍率/温度怎么变
    · b（日历系数）随温度怎么变（Arrhenius 的期望方向）
    · **两者的相对贡献**：在 80% SOH 时刻，容量损失里多少来自日历、多少来自循环

这一步直接回答 v1.3 留下的问题：
"Sandia 按日历期停止测试"到底是噪声，还是可以被建模的信号？

产出
----
    results/calendar/cells_time.parquet   逐电芯的时间轴与两系数
    results/calendar/decomposition.csv    日历/循环贡献分解
    results/calendar/arrhenius.csv        日历系数 vs 温度（Arrhenius 拟合）
    pictures/21_calendar_vs_cycle.png
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
from scipy.optimize import curve_fit  # noqa: E402

from fastcharge.config import FIGURES, RESULTS, ensure_dirs  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

SNL_ZIP = Path(r"D:\项目\02_数据集\01_国际公开集\Sandia_SNL\SNL.zip")
OUT = RESULTS / "calendar"
OUT.mkdir(parents=True, exist_ok=True)

_RE = re.compile(
    r"SNL_18650_(?P<chem>LFP|NCA|NMC)_(?P<temp>\d+)C_"
    r"(?P<lo>\d+)-(?P<hi>\d+)_(?P<chg>[0-9.]+)-(?P<dis>[0-9.]+)C_(?P<rep>[a-z])\.pkl"
)

SMOOTH = 11


def parse_name(name: str) -> dict:
    m = _RE.search(name)
    if not m:
        return {}
    d = m.groupdict()
    return {"chem": d["chem"], "temp_C": float(d["temp"]),
            "soc_lo": float(d["lo"]) / 100, "soc_hi": float(d["hi"]) / 100,
            "charge_rate": float(d["chg"]), "discharge_rate": float(d["dis"]),
            "replicate": d["rep"]}


def extract_one(cycles: list) -> dict:
    """抽容量 + 逐圈时长 → 日历时间轴 + 平均温度。"""
    n = len(cycles)
    q = np.full(n, np.nan)
    dur = np.full(n, np.nan)
    temp = np.full(n, np.nan)
    for k, c in enumerate(cycles):
        qc = c.get("charge_capacity_in_Ah")
        if qc is not None:
            q[k] = float(np.nanmax(np.asarray(qc, dtype=float)))
        t = c.get("time_in_s")
        if t is not None and len(t) > 1:
            tt = np.asarray(t, dtype=float)
            dur[k] = float(tt[-1] - tt[0])
        tm = c.get("temperature_in_C")
        if tm is not None and len(tm) > 0:
            temp[k] = float(np.nanmean(np.asarray(tm, dtype=float)))

    ok = np.isfinite(q) & np.isfinite(dur) & (dur > 0)
    q, dur, temp = q[ok], dur[ok], temp[ok]
    if q.size < 20:
        return {}

    sm = pd.Series(q).rolling(SMOOTH, min_periods=1, center=True).median().to_numpy()
    q_ref = float(np.nanmedian(q[:5]))
    if q_ref <= 0:
        return {}

    soh = sm / q_ref
    N = np.arange(q.size, dtype=float)
    t_cal = np.cumsum(dur) / 86400.0         # 日历天
    return {"soh": soh, "N": N, "t_days": t_cal, "n": int(q.size),
            "mean_temp": float(np.nanmean(temp)) if np.isfinite(temp).any() else np.nan,
            "t_total_days": float(t_cal[-1])}


def loss_model(t, a: float, p: float, b: float):
    """ΔSOH = a·N^p + b·√t（循环项 + 日历项）。"""
    N, t_days = t
    return a * np.power(N, p) + b * np.sqrt(np.maximum(t_days, 0))


def fit_two_term(rec: dict) -> dict:
    """拟合双项模型，返回两系数与各自的贡献。"""
    soh, N, t_days = rec["soh"], rec["N"], rec["t_days"]
    loss = 1.0 - soh
    # 只用 SOH 还在 0.6 以上的区间（更深处 SEI 机制可能切换）
    keep = soh >= 0.60
    if keep.sum() < 20:
        keep = np.ones_like(soh, dtype=bool)
    Nk, tk, lk = N[keep], t_days[keep], loss[keep]
    if lk.max() <= 1e-4:
        return {}

    try:
        popt, _ = curve_fit(
            loss_model, (Nk, tk), lk,
            p0=[1e-5, 0.5, 1e-3],
            bounds=([0, 0.05, 0], [1e-1, 2.0, 1.0]),
            maxfev=20000,
        )
    except Exception:
        return {}
    a, p, b = (float(v) for v in popt)

    # 在 SOH=0.80 处分解（若没到 0.80 就用末态）
    target = 0.80
    idx = np.flatnonzero(soh <= target)
    i80 = int(idx[0]) if idx.size else int(len(soh) - 1)
    cyc_part = a * (i80 ** p)
    cal_part = b * np.sqrt(max(t_days[i80], 0))
    total = cyc_part + cal_part
    return {
        "a_cycle": a, "p_exp": p, "b_calendar": b,
        "cycle_part_at_eol": cyc_part,
        "calendar_part_at_eol": cal_part,
        "cycle_share": cyc_part / total if total > 0 else np.nan,
        "calendar_share": cal_part / total if total > 0 else np.nan,
        "soh_at_eval": float(soh[i80]),
        "n_at_eval": float(i80),
        "t_at_eval_days": float(t_days[i80]),
        "fit_total_loss": total,
        "observed_loss": float(1 - soh[i80]),
        "eq_cycles_per_day": float(np.mean(1 / (np.diff(t_days) + 1e-9)))
        if len(t_days) > 1 else np.nan,
        "cycles_per_day": float(len(t_days) / max(t_days[-1], 1e-9)),
    }


def build_all() -> pd.DataFrame:
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
            rec = extract_one(obj["cycle_data"])
            del obj
            if not rec:
                print(f"  {Path(name).name:46s} 跳过（数据不足）", flush=True)
                continue
            f = fit_two_term(rec)
            if not f:
                print(f"  {Path(name).name:46s} 跳过（拟合失败）", flush=True)
                continue
            rows.append({**meta, "cell_id": Path(name).stem,
                         "n_cycles": rec["n"], "t_total_days": rec["t_total_days"],
                         "mean_temp_measured": rec["mean_temp"], **f})
            print(f"  {Path(name).name:46s} n={rec['n']:5d} t={rec['t_total_days']:5.1f}d "
                  f"循环占 {f['cycle_share']*100:4.0f}% 日历占 {f['calendar_share']*100:4.0f}%",
                  flush=True)
    return pd.DataFrame(rows)


def arrhenius(df: pd.DataFrame) -> pd.DataFrame:
    """日历系数 b 对温度的 Arrhenius 拟合（按化学分组）。"""
    rows = []
    for chem, g in df.groupby("chem"):
        g = g.dropna(subset=["b_calendar", "temp_C"])
        g = g[(g["b_calendar"] > 0) & (g["temp_C"] > 0)]
        if len(g) < 2:
            continue
        T = 1.0 / (g["temp_C"].to_numpy(float) + 273.15)
        y = np.log(g["b_calendar"].to_numpy(float))
        if len(np.unique(T)) < 2:
            rows.append({"chem": chem, "n": len(g), "ea_kJ_mol": np.nan,
                         "note": "单一温度，无法拟合"})
            continue
        slope, intercept, r, p, se = __import__("scipy.stats", fromlist=["linregress"]).linregress(T, y)
        ea = -slope * 8.314 / 1000      # kJ/mol
        rows.append({"chem": chem, "n": len(g), "ea_kJ_mol": float(ea),
                     "r2": float(r ** 2), "p": float(p),
                     "b_at_25C": float(np.exp(intercept + slope / 298.15))})
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    cache = OUT / "cells_time.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"复用缓存 {len(df)} 电芯")
    else:
        print("=== 1) 重建时间轴并拟合双项模型 ===")
        df = build_all()
        df.to_parquet(cache, index=False)
        print(f"\n  → {len(df)} 电芯成功")

    if df.empty:
        print("无可用数据")
        return 1

    print("\n=== 2) 日历 vs 循环 的贡献分解 ===")
    print(f"  循环占比 中位 {df['cycle_share'].median()*100:.1f}%"
          f"（{df['cycle_share'].quantile(0.25)*100:.0f}–{df['cycle_share'].quantile(0.75)*100:.0f}%）")
    print(f"  日历占比 中位 {df['calendar_share'].median()*100:.1f}%"
          f"（{df['calendar_share'].quantile(0.25)*100:.0f}–{df['calendar_share'].quantile(0.75)*100:.0f}%）")

    by_rate = df.groupby("discharge_rate").agg(
        n=("cell_id", "size"),
        cycle_share_median=("cycle_share", "median"),
        calendar_share_median=("calendar_share", "median"),
        b_calendar_median=("b_calendar", "median"),
        cycles_per_day=("cycles_per_day", "median"),
    ).reset_index()
    print("\n  按放电倍率:")
    print(by_rate.round(3).to_string(index=False))
    by_rate.to_csv(OUT / "decomposition.csv", index=False, encoding="utf-8-sig")

    by_temp = df.groupby("temp_C").agg(
        n=("cell_id", "size"),
        cycle_share_median=("cycle_share", "median"),
        calendar_share_median=("calendar_share", "median"),
        b_calendar_median=("b_calendar", "median"),
        cycles_per_day=("cycles_per_day", "median"),
    ).reset_index()
    print("\n  按温度:")
    print(by_temp.round(3).to_string(index=False))

    # ---- 3. Arrhenius ----
    print("\n=== 3) 日历系数 vs 温度（Arrhenius）===")
    arr = arrhenius(df)
    if not arr.empty:
        print(arr.round(3).to_string(index=False))
        arr.to_csv(OUT / "arrhenius.csv", index=False, encoding="utf-8-sig")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    ax = axes[0]
    for rate, marker in zip(sorted(df["discharge_rate"].unique()),
                            ["o", "s", "^", "D", "v"]):
        s = df[df["discharge_rate"] == rate]
        ax.scatter(s["cycle_share"] * 100, s["calendar_share"] * 100,
                   s=46, marker=marker, alpha=0.8, label=f"{rate:g}C")
    ax.plot([0, 100], [100, 0], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("循环老化贡献占比 (%)")
    ax.set_ylabel("日历老化贡献占比 (%)")
    ax.set_title("80% SOH 处的容量损失分解")
    ax.legend(fontsize=8, title="放电倍率")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    agg = df.groupby(["temp_C", "chem"], observed=True).agg(
        cal=("calendar_share", "mean"), n=("cell_id", "size")).reset_index()
    markers = {"LFP": "o", "NCA": "s", "NMC": "^"}
    for chem in agg["chem"].unique():
        s = agg[agg["chem"] == chem]
        ax2.plot(s["temp_C"], s["cal"] * 100, marker=markers.get(chem, "o"),
                 lw=1.8, label=chem)
    ax2.set_xlabel("温度 (°C)")
    ax2.set_ylabel("日历老化占比 (%)")
    ax2.set_title("温度越高，日历老化越主导")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "21_calendar_vs_cycle.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
