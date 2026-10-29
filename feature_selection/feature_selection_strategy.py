"""
==============================================================
Feature Selection Strategy - 特征选择策略工厂

提供统一的特征选择器获取接口，支持多种特征选择方法的注册与切换。

支持的内置方法:
    - 'boruta':  Boruta特征选择（基于随机森林的嵌入式选择）
    - 'none':    不做特征选择，全量特征直通

扩展自定义方法:
    1. 定义选择器类（继承 BaseSelector）
    2. 调用 register_feature_selector('method_name', SelectorClass) 注册
    3. 在 main-bak.py 中设置 FEATURE_SELECTION_METHOD = 'method_name' 即可切换

==============================================================
"""

from feature_selection.base_selector import BaseSelector
from feature_selection.boruta_selector import BorutaSelector
import os
import numpy as np


# ============================================================
# 内置选择器注册表
# ============================================================
_FEATURE_SELECTORS = {}


# ============================================================
# 内置选择器实现
# ============================================================

class NoFeatureSelector(BaseSelector):
    """不做特征选择，全量特征直通。

    当 USE_BORUTA=False 或其他 "no-op" 方法被选中时，
    所有特征原样通过，不进行修改。
    """

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.selected_features_ = list(X.columns)
            self.support_mask_ = np.ones(len(X.columns), dtype=bool)
        else:
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            self.selected_features_ = [f'feature_{i}' for i in range(n_features)]
            self.support_mask_ = np.ones(n_features, dtype=bool)
        return self

    def transform(self, X):
        return X

    def get_selected_features(self):
        return self.selected_features_


# ============================================================
# 注册函数
# ============================================================

def register_feature_selector(name, selector_cls):
    """注册自定义特征选择器到全局工厂。

    Args:
        name:       方法名称标识（字符串），用于在 get_feature_selector 中引用
        selector_cls: 选择器类（必须继承 BaseSelector）

    Example:
        >>> from feature_selection_strategy import register_feature_selector
        >>> register_feature_selector('mutual_info', MutualInfoSelector)
    """
    if not issubclass(selector_cls, BaseSelector):
        raise TypeError("Selector class must inherit from BaseSelector")
    _FEATURE_SELECTORS[name] = selector_cls


def get_feature_selector(method='boruta', **kwargs):
    """获取特征选择器实例（工厂函数）。

    Args:
        method: 选择器方法名。
                内置: 'boruta'（Boruta选择）, 'none'（直通）
                自定义: 通过 register_feature_selector 注册的名称
        **kwargs: 传递给选择器构造函数的关键字参数

    Returns:
        特征选择器实例（已实现 BaseSelector 接口）

    Raises:
        ValueError: 当方法名未注册时
        TypeError: 当选择的类未继承 BaseSelector 时

    Example:
        >>> selector = get_feature_selector('boruta', random_state=42, max_iter=100)
        >>> selector.fit(X_train, y_train)
        >>> X_selected = selector.transform(X_test)
    """
    _register_builtins()

    if method not in _FEATURE_SELECTORS:
        available = ', '.join(sorted(_FEATURE_SELECTORS.keys()))
        raise ValueError(
            f"Unknown feature selection method: '{method}'.\n"
            f"Available methods: {available}"
        )

    selector_cls = _FEATURE_SELECTORS[method]
    return selector_cls(**kwargs)


def list_available_selectors():
    """列出所有可用的特征选择器方法名。

    Returns:
        list[str]: 可用方法名列表
    """
    _register_builtins()
    return list(_FEATURE_SELECTORS.keys())


# ============================================================
# 内置注册（模块加载时自动执行）
# ============================================================

def _register_builtins():
    """注册内置选择器（延迟注册，避免循环导入）"""
    if 'boruta' not in _FEATURE_SELECTORS:
        _FEATURE_SELECTORS['boruta'] = BorutaSelector
    if 'none' not in _FEATURE_SELECTORS:
        _FEATURE_SELECTORS['none'] = NoFeatureSelector


# 模块导入时自动注册内置选择器
_register_builtins()


# ============================================================
# 便捷函数：根据方法名获取输出目录路径
# ============================================================

def get_feature_output_dir(method, group, name, model_name):
    """根据特征选择方法名生成对应的输出目录路径。

    目录规则:
        - 'boruta'  -> results/feature_selection/boruta/{group}/{name}/{model_name}/
        - 其他方法   -> results/feature_selection/{method}/{group}/{name}/{model_name}/
        - 'none'     -> results/no_feature_selection/{group}/{name}/{model_name}/

    Args:
        method:        特征选择方法名（'boruta', 'none' 或自定义方法名）
        group:         数据集分组名（如 NASA）
        name:          数据集名（如 CM1）
        model_name:    模型名（如 LogisticRegression）

    Returns:
        str: 输出目录的绝对路径
    """
    if method == 'none':
        return os.path.join("results", "no_feature_selection", group, name, model_name)
    else:
        return os.path.join("results", "feature_selection", method, group, name, model_name)


# 导出公共 API
__all__ = [
    'BaseSelector',
    'BorutaSelector',
    'NoFeatureSelector',
    'register_feature_selector',
    'get_feature_selector',
    'list_available_selectors',
    'get_feature_output_dir',
]
