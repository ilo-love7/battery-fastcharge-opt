#!/bin/sh
# 一键复现全流程（基础版约 45 min + 深化版约 30 min）
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}

echo "=== 基础版 ==="
$PY scripts/01_index.py
$PY scripts/02_eda.py
$PY scripts/03_surrogate.py
$PY scripts/04_optimize.py
$PY scripts/05_physics.py
$PY scripts/06_constrained_opt.py
$PY scripts/07_recommend.py
$PY scripts/08_readme_figures.py

echo "=== 深化版 v1.1 ==="
$PY scripts/09_multistage.py
$PY scripts/10_rl.py
$PY scripts/11_early.py
$PY scripts/12_robustness.py
$PY scripts/13_deepening_summary.py

echo "=== 测试 ==="
$PY -m pytest tests -q

echo "done. see results/ and reports/"
