"""01 — ISU-ILCC 全量索引：zip 直读 → cells / cycles 标量表。

用法:
    python scripts/01_index.py            # 全量（240 电芯，约 3–5 分钟）
    python scripts/01_index.py --limit 12 # 冒烟测试
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from fastcharge.config import METRICS, ensure_dirs, isu_zip  # noqa: E402
from fastcharge.index import build_tables  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    ensure_dirs()
    zp = isu_zip()
    print(f"数据源: {zp} ({zp.stat().st_size / 1e9:.2f} GB)")
    t0 = time.time()
    cells, cycles = build_tables(zp, soh_threshold=args.threshold, limit=args.limit)
    dt = time.time() - t0

    print(f"\n完成: {len(cells)} 电芯 / {len(cycles):,} 循环 / {dt:.1f} s")
    print(f"cells  → {METRICS / 'cells.parquet'}")
    print(f"cycles → {METRICS / 'cycles.parquet'}")

    # 健康度体检
    n_cens = int(cells["censored"].sum())
    print(f"\n删失电芯（未跌破 {args.threshold:.0%} SOH）: {n_cens} / {len(cells)}")
    print("\nEOL 循环数分布:")
    print(cells["eol_cycle"].describe().round(0).to_string())
    print("\n分组协议一览（前 10）:")
    cols = ["group", "charge_rate_peak", "discharge_rate_mean", "soc_lo", "soc_span",
            "n_cycles", "eol_cycle", "censored"]
    print(cells.sort_values("group")[cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
