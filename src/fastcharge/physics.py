"""物理约束：用 PyBaMM 判定"无析锂"的倍率上限，把优化限制在安全可行域（阶段 4）。

析锂判据（这是本项目采用的那一个，不是模糊的"电位看图"）
----------------------------------------------------------
PyBaMM 的析锂反应过电位:
    η_pl = φ_s - φ_l - U_pl(0 V)
**η_pl < 0 表示锂在负极表面沉积（析锂）而不是嵌入石墨。**
所以判据是: 全程 min(η_pl) ≥ 0。这里用 X-averaged 版本（比逐点 x 分布快、稳定性好），
并对关键点用 DFN 复核。

为什么不能只看 "Negative electrode potential [V]"
----------------------------------------------
该变量是固相电位 φ_s，其参考点由模型内部设定（实测恒在 0 V 附近），
不是相对 Li/Li+ 的电势差，直接拿它比 0 V 会得到"任何倍率都不析锂"的错误结论。
用反应过电位才是唯一自洽的判据——这一点已写进报告的方法说明。

化学体系对齐
------------
ISU-ILCC = NMC/石墨软包 502030（0.25 Ah）；OKane2022 = NMC811/石墨-SiO2 商用电芯。
同族不同芯：本模块只用其**边界形状与相对趋势**（低温/高倍率 → 析锂），
不把绝对数值当作 ISU-ILCC 的真值。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: 析锂判据阈值（V）。η_pl ≥ 0 即安全；留 5 mV 工程裕度。
PLATING_THRESHOLD_V = 0.0
SAFETY_MARGIN_V = 0.005


@dataclass
class SimulationSetup:
    model: str = "SPM"
    plating: bool = True
    c_rate: float = 1.0
    temperature_C: float = 30.0
    soc_start: float = 0.0
    soc_end: float = 1.0
    v_max: float = 4.2
    cutoff_mA: float = 50.0


def _build_model(setup: SimulationSetup):
    import pybamm

    opts = {}
    if setup.plating:
        opts["lithium plating"] = "irreversible"
    if setup.model == "DFN":
        return pybamm.lithium_ion.DFN(options=opts)
    if setup.model == "SPMe":
        opts["particle"] = "quadratic profile"
        return pybamm.lithium_ion.SPMe(options=opts)
    return pybamm.lithium_ion.SPM(options=opts)


def _parameter_values(setup: SimulationSetup):
    import pybamm

    pv = pybamm.ParameterValues("OKane2022")
    pv["Ambient temperature [K]"] = setup.temperature_C + 273.15
    pv["Initial temperature [K]"] = setup.temperature_C + 273.15
    pv.set_initial_state(setup.soc_start)
    return pv


def min_plating_overpotential(setup: SimulationSetup) -> dict:
    """给定充电条件，返回全程析锂过电位最小值与判据结果。"""
    import pybamm

    model = _build_model(setup)
    pv = _parameter_values(setup)
    exp = pybamm.Experiment([
        f"Charge at {setup.c_rate:.3f}C until {setup.v_max} V",
        f"Hold at {setup.v_max} V until {setup.cutoff_mA} mA",
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp).solve()

    eta = np.asarray(sol["X-averaged negative electrode lithium plating reaction overpotential [V]"].entries)
    loss = np.asarray(sol["Loss of lithium to negative lithium plating [mol]"].entries)
    t = np.asarray(sol["Time [s]"].entries)
    v = np.asarray(sol["Voltage [V]"].entries)
    i_min = float(np.nanmin(eta))
    return {
        "model": setup.model,
        "c_rate": setup.c_rate,
        "temperature_C": setup.temperature_C,
        "soc_start": setup.soc_start,
        "min_plating_overpotential_V": i_min,
        "t_min_overpotential_s": float(t[int(np.nanargmin(eta))]) if eta.size else np.nan,
        "plating_lithium_mol": float(np.nanmax(loss)) if loss.size else 0.0,
        "plated": bool(i_min < PLATING_THRESHOLD_V),
        "plated_with_margin": bool(i_min < SAFETY_MARGIN_V),
        "charge_time_s": float(t[-1]) if t.size else np.nan,
        "v_reached": float(np.nanmax(v)) if v.size else np.nan,
    }


def scan_plating_envelope(rates=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
                          temps=(0.0, 15.0, 30.0, 45.0),
                          model: str = "SPM", soc_start: float = 0.0) -> pd.DataFrame:
    """扫描 (倍率 × 温度) → 析锂过电位最小值，得到安全边界。"""
    rows = []
    for T in temps:
        for c in rates:
            setup = SimulationSetup(model=model, c_rate=c, temperature_C=T, soc_start=soc_start)
            try:
                r = min_plating_overpotential(setup)
            except Exception as exc:
                r = {"model": model, "c_rate": c, "temperature_C": T,
                     "min_plating_overpotential_V": np.nan, "plated": None,
                     "error": str(exc)[:120]}
            rows.append(r)
            print(f"  {model} {c:.2f}C @ {T:.0f}°C → η_min = "
                  f"{r['min_plating_overpotential_V']:+.4f} V  "
                  f"{'析锂' if r.get('plated') else '安全'}", flush=True)
    return pd.DataFrame(rows)


def safe_rate_limit(envelope: pd.DataFrame, temperature_C: float = 30.0,
                    margin: float = SAFETY_MARGIN_V) -> float:
    """某温度下的最大安全倍率：**从最小倍率起连续安全段**的上界。

    注意: PyBaMM/SPM 下 η_pl 对倍率并非单调（极片表面在高倍率下提前达到
    cutoff，反而回到安全区）。因此不能取"任意安全点的最大倍率"，
    那样会得出"4C 也安全"的荒谬结论。工程上只承诺连续安全段。
    """
    sub = envelope[(envelope["temperature_C"] == temperature_C)
                   & envelope["min_plating_overpotential_V"].notna()].sort_values("c_rate")
    if sub.empty:
        return np.nan
    sub = sub.drop_duplicates("c_rate")
    safe_flag = (sub["min_plating_overpotential_V"] >= margin).to_numpy()
    if not safe_flag.any():
        return 0.0
    n_ok = 0
    for f in safe_flag:
        if f:
            n_ok += 1
        else:
            break
    return float(sub["c_rate"].to_numpy()[n_ok - 1])


def temperature_rate_map(rates=(0.5, 1.0, 2.0, 3.0, 4.0, 6.0),
                         temps=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0)) -> pd.DataFrame:
    """构建 温度 × 倍率 的安全 map（布尔 + 过电位值）。"""
    rows = []
    for T in temps:
        for c in rates:
            setup = SimulationSetup(c_rate=c, temperature_C=T)
            try:
                r = min_plating_overpotential(setup)
            except Exception:
                r = {"min_plating_overpotential_V": np.nan, "plated": None}
            rows.append({"temperature_C": T, "c_rate": c,
                         "eta_min_V": r["min_plating_overpotential_V"],
                         "safe": (r["min_plating_overpotential_V"] is not None
                                  and r["min_plating_overpotential_V"] >= SAFETY_MARGIN_V)})
            print(f"  {c:.1f}C @ {T:.0f}°C → {r['min_plating_overpotential_V']:+.4f} V", flush=True)
    return pd.DataFrame(rows)


def check_protocols(protocols: pd.DataFrame, model: str = "SPM",
                    temperature_C: float = 30.0) -> pd.DataFrame:
    """对一批候选协议逐个做析锂检查（统计建议 vs 机理的一致性验证）。"""
    rows = []
    for _, p in protocols.iterrows():
        rate = float(p["peak_rate_C"])
        span = float(p.get("soc_window", 1.0))
        setup = SimulationSetup(model=model, c_rate=rate, temperature_C=temperature_C,
                                soc_start=max(0.0, 1.0 - span))
        try:
            r = min_plating_overpotential(setup)
        except Exception as exc:
            r = {"model": model, "c_rate": rate, "temperature_C": temperature_C,
                 "min_plating_overpotential_V": np.nan, "plated": None,
                 "error": str(exc)[:100]}
        r.update({"soc_window": span,
                  "pred_cycles": float(p.get("pred_cycles", np.nan)),
                  "time_per_fec_h": float(p.get("time_per_fec_h", np.nan))})
        rows.append(r)
    return pd.DataFrame(rows)


def cascade_experiment(steps: list[tuple[float, float]], v_hold: float = 4.2,
                       cutoff_mA: float = 50.0) -> "pybamm.Experiment":
    """构造电压触发的阶梯充电实验。

    steps = [(rate_C, v_until_V), ...]。PyBaMM 的 Experiment 支持
    "Charge at X C until Y V" 序列，因此阶梯协议可以逐段直接模拟，
    不必用倍率上限表去近似——这是最如实的做法。
    """
    import pybamm

    ops = [f"Charge at {r:.3f}C until {v:.3f} V" for r, v in steps]
    ops.append(f"Hold at {v_hold:.3f} V until {cutoff_mA} mA")
    return pybamm.Experiment(ops)


def simulate_cascade(steps: list[tuple[float, float]], *, model: str = "SPM",
                     temperature_C: float = 30.0, soc_start: float = 0.0,
                     v_hold: float = 4.2) -> dict:
    """模拟一条阶梯协议，返回析锂判据、总时长、SOC 与电压轨迹。"""
    import pybamm

    setup = SimulationSetup(model=model, c_rate=steps[0][0], temperature_C=temperature_C,
                            soc_start=soc_start, v_max=v_hold)
    mdl = _build_model(setup)
    pv = _parameter_values(setup)
    exp = cascade_experiment(steps, v_hold=v_hold)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = pybamm.Simulation(mdl, parameter_values=pv, experiment=exp).solve()

    eta = np.asarray(sol["X-averaged negative electrode lithium plating reaction overpotential [V]"].entries)
    t = np.asarray(sol["Time [s]"].entries)
    v = np.asarray(sol["Voltage [V]"].entries)
    soc = np.asarray(sol["Discharge capacity [A.h]"].entries)  # 充电时递增
    q_ref = np.asarray(sol["Discharge capacity [A.h]"].entries)[-1] if t.size else np.nan

    eta_min = float(np.nanmin(eta)) if eta.size else np.nan
    stages = [{"rate_C": r, "v_until_V": vv} for r, vv in steps]
    return {
        "steps": steps,
        "stages": stages,
        "model": model,
        "temperature_C": temperature_C,
        "soc_start": soc_start,
        "min_plating_overpotential_V": eta_min,
        "plated": bool(eta_min < PLATING_THRESHOLD_V),
        "safe_with_margin": bool(eta_min >= SAFETY_MARGIN_V),
        "charge_time_s": float(t[-1]) if t.size else np.nan,
        "charge_time_h": float(t[-1]) / 3600.0 if t.size else np.nan,
        "capacity_charged_Ah": float(np.nanmax(soc) - np.nanmin(soc)) if soc.size else np.nan,
        "v_max_V": float(np.nanmax(v)) if v.size else np.nan,
        "n_points": int(len(t)),
        "soc_traj": soc,
        "time_s": t,
    }


def sweep_cascades(step_sets: list[list[tuple[float, float]]], **kwargs) -> pd.DataFrame:
    """批量模拟多条阶梯协议（用于枚举筛选）。"""
    rows = []
    for steps in step_sets:
        try:
            r = simulate_cascade(steps, **kwargs)
        except Exception as exc:
            rows.append({"steps": steps, "min_plating_overpotential_V": np.nan,
                         "plated": None, "charge_time_h": np.nan, "error": str(exc)[:100]})
            continue
        r.pop("soc_traj", None)
        r.pop("time_s", None)
        r["steps_str"] = " → ".join(f"{rr:g}C@{vv:g}V" for rr, vv in steps)
        rows.append(r)
        print(f"  {r['steps_str']:38s} η_min={r['min_plating_overpotential_V']:+.4f} V  "
              f"t={r['charge_time_h']:.3f} h  {'析锂' if r['plated'] else '安全'}", flush=True)
    df = pd.DataFrame(rows)
    if "stages" in df.columns:
        df = df.drop(columns=["stages"])
    return df
