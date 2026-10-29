
"""
main.py - 程序启动入口

仅保留启动入口和配置变量，所有业务逻辑已迁移至 baseline_runner.py。

配置说明:
    1. 特征选择方法:
       - 方式A（推荐）: 直接设置 FEATURE_SELECTION_METHOD
         可选值: 'boruta' | 'none' | 自定义注册的方法名
       - 方式B（向后兼容）: 在 config.py 中设置 USE_BORUTA = True/False
         True  -> 使用 boruta
         False -> 使用 none（全量特征直通）
       方式B会被方式A覆盖（如果方式A已设置的话）

    2. 可解释性分析开关:
       - USE_SHAP:  是否执行 SHAP 分析（默认 True）
       - USE_LIME:  是否执行 LIME 分析（默认 True）

使用示例:
    # 示例1: 使用 Boruta 特征选择 + SHAP + LIME（默认）
    python main-bak.py

    # 示例2: 不使用特征选择 + 仅 SHAP
    FEATURE_SELECTION_METHOD='none'
    USE_SHAP=True
    USE_LIME=False
    python main-bak.py

    # 示例3: 使用自定义特征选择方法
    FEATURE_SELECTION_METHOD='mutual_info'
    python main-bak.py
"""

import os
import sys

# ============================================================
# 配置区域（修改此处即可控制实验行为）
# ============================================================

# --- 特征选择方法 ---
# 优先使用 FEATURE_SELECTION_METHOD 直接指定
# 若未设置，则向后兼容读取 config.USE_BORUTA
FEATURE_SELECTION_METHOD = os.environ.get(
    'FEATURE_SELECTION_METHOD', None
)

if FEATURE_SELECTION_METHOD is None:
    # 尝试从 config 模块读取 USE_BORUTA
    try:
        import config
        use_boruta = getattr(config, 'USE_BORUTA', True)
        FEATURE_SELECTION_METHOD = 'boruta' if use_boruta else 'none'
    except ImportError:
        FEATURE_SELECTION_METHOD = 'boruta'

# --- 可解释性分析开关 ---
USE_SHAP = os.environ.get('USE_SHAP', None)
if USE_SHAP is not None:
    USE_SHAP = USE_SHAP.lower() in ('true', '1', 'yes')
else:
    try:
        import config
        USE_SHAP = getattr(config, 'USE_SHAP', True)
    except ImportError:
        USE_SHAP = True

USE_LIME = os.environ.get('USE_LIME', None)
if USE_LIME is not None:
    USE_LIME = USE_LIME.lower() in ('true', '1', 'yes')
else:
    try:
        import config
        USE_LIME = getattr(config, 'USE_LIME', True)
    except ImportError:
        USE_LIME = True

# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    from models.baseline_runner import run_experiment_pipeline

    run_experiment_pipeline(
        feature_selection_method=FEATURE_SELECTION_METHOD,
        enable_shap=USE_SHAP,
        enable_lime=USE_LIME,
    )