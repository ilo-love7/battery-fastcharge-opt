"""集中配置：数据源、输出目录、口径常量。

设计原则:
- 数据源只读，绝不改动 D:\\项目\\02_数据集 下的原始压缩包
- 所有产物落 results/ ，可重建
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
# 允许用环境变量覆盖，便于 CI / 其他机器复现
ENV_DATA = "FASTCHARGE_ISU_ZIP"

DEFAULT_ISU_ZIP = r"D:\项目\02_数据集\01_国际公开集\ISU_ILCC\ISU_ILCC.zip"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
CACHE = RESULTS / "cache"
MODELS = RESULTS / "models"
REPORTS = REPO_ROOT / "reports"
CONFIGS = REPO_ROOT / "configs"


def isu_zip() -> Path:
    """ISU-ILCC 统一格式压缩包路径（首次调用时校验存在性）。"""
    p = Path(os.environ.get(ENV_DATA, DEFAULT_ISU_ZIP))
    if not p.exists():
        raise FileNotFoundError(
            f"ISU-ILCC 数据包不存在: {p}\n"
            f"设置环境变量 {ENV_DATA} 指向 ISU_ILCC.zip，或从 "
            f"https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234 下载。"
        )
    return p


def ensure_dirs() -> None:
    for d in (METRICS, FIGURES, CACHE, MODELS, REPORTS, CONFIGS):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 口径常量
# --------------------------------------------------------------------------
#: 电芯额定容量 (Ah)。ISU-ILCC 全部为同一款 502030 软包 NMC，标称 0.25 Ah。
NOMINAL_CAPACITY_AH = 0.25

#: EOL 判据：容量保持率阈值。官方 ISU-ILCC 与多数文献用 80% of nominal。
EOL_SOH_THRESHOLD = 0.80

#: 充电段与放电段的电流符号约定（BatteryLife 统一格式：正=充电）
CHARGE_SIGN = 1

#: 放电倍率固定为 0.5C 的"经典"分组（G1–G24），用于分层分析
CLASSIC_DISCHARGE_C = 0.5


@dataclass(frozen=True)
class LifetimeThresholds:
    """寿命结局判定的备选口径，供敏感性分析。"""

    primary: float = EOL_SOH_THRESHOLD
    alternatives: tuple[float, ...] = (0.90, 0.85, 0.80, 0.70)
