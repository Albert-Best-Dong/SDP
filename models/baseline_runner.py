
"""
baseline_runner.py - 基线实验运行器

主逻辑功能分离，各模块职责明确。

特征选择策略:
    通过 feature_selection_strategy 统一获取特征选择器实例。
    支持的方法（通过 main-bak.py 中的 FEATURE_SELECTION_METHOD 控制）:
        - 'boruta':  使用 Boruta 进行特征选择（默认）
        - 'none':    不做特征选择，全量特征直接通过
    扩展新方法:
        1. 在 feature_selection_strategy.py 中定义新的选择器类（继承 BaseSelector）
        2. 在 _FEATURE_SELECTORS 字典中注册，或调用 register_feature_selector()
        3. 在 main-bak.py 中设置 FEATURE_SELECTION_METHOD='new_method'

可解释性分析开关（通过 config 或 main-bak.py 控制）:
    - USE_SHAP:  是否执行 SHAP 分析（默认 True）
    - USE_LIME:  是否执行 LIME 分析（默认 True）

目录输出规则:
    - Boruta      -> results/feature_selection/boruta/{group}/{name}/{model}/
    - 其他方法      -> results/feature_selection/{method}/{group}/{name}/{model}/
    - 无特征选择    -> results/no_feature_selection/{group}/{name}/{model}/
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config
from feature_selection.feature_summary import FeatureSummary
from feature_selection.feature_selection_strategy import (
    get_feature_selector,
    get_feature_output_dir,
)
from models.classifiers import get_models

from myutils.metrics import calculate_metrics

from myutils.visualization import (
    save_ROC_comparison,
    save_PR_comparison,
    save_CM,
)

import matplotlib

matplotlib.use("Agg")


# ============================================================
# 1. 环境初始化
# ============================================================

def create_result_directories():
    """创建结果输出目录"""
    dirs = [
        "results/metrics",
        "results/roc_curve",
        "results/pr_curve",
        "results/confusion_matrix",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ============================================================
# 2. 数据加载与预处理
# ============================================================

def load_and_preprocess_data(group, name):
    """加载数据集并进行预处理

    Args:
        group: 数据集分组名（如 NASA）
        name: 数据集名称（如 CM1）

    Returns:
        X: 特征矩阵（numpy array）
        y: 标签向量（numpy array）
        feature_names: 特征名列表
    """
    path = f"datasets/{group}/{name}.csv"
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return None

    df = pd.read_csv(path)

    # 分离标签
    label = config.get_label(df)
    # 先统一标签：0 = 无缺陷，1 = 有缺陷
    y = config.encode_labels(df[label], group)
    # 再删除标签列，保留特征
    X = df.drop(columns=[label])
    feature_names = X.columns.tolist()

    # 缺失值处理
    X = X.replace("?", np.nan)
    X = X.fillna(X.median())
    X = X.values
    y = np.array(y)

    return X, y, feature_names


# ============================================================
# 3. 特征选择（统一策略接口）
# ============================================================

def run_feature_selection(X_train, y_train, method='boruta', **kwargs):
    """执行特征选择（统一策略接口）

    Args:
        X_train: 训练特征（DataFrame）
        y_train: 训练标签
        method: 特征选择方法名，可选 'boruta', 'none' 或自定义注册的方法
        **kwargs: 传递给选择器的额外参数

    Returns:
        selector: 拟合后的特征选择器实例
        selected_features: 选中的特征名列表
    """
    selector = get_feature_selector(method, **kwargs)
    selector.fit(X_train, y_train)
    selected_features = selector.get_selected_features()
    return selector, selected_features


def save_selected_features(selected_features, feature_dir, fold_id):
    """保存单次特征选择结果到CSV

    Args:
        selected_features: 选中的特征名列表
        feature_dir: 输出目录
        fold_id: 折编号
    """
    pd.DataFrame({"Feature": selected_features}).to_csv(
        os.path.join(feature_dir, f"fold_{fold_id}.csv"),
        index=False,
    )


# ============================================================
# 4. 可解释性分析
# ============================================================

def run_shap_explanation(
    model, X_train, X_test, y_test,
    feature_names, dataset_name, project_name, model_name,
):
    """执行SHAP解释分析

    Args:
        model:          训练好的模型
        X_train:        训练特征
        X_test:         测试特征
        y_test:         测试标签
        feature_names:  特征名列表
        dataset_name:   数据集分组名
        project_name:   项目名
        model_name:     模型名
    """
    from explain.shap_module import SHAPAnalyzer

    shap_analysis = SHAPAnalyzer(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        dataset_name=dataset_name,
        project_name=project_name,
        model_name=model_name,
    )
    shap_analysis.run_all(
        html=True, interaction=True, waterfall_samples=(0, 5, 10)
    )


def run_lime_explanation(
    model, X_train, X_test, y_test,
    feature_names, dataset_name, project_name, model_name,
):
    """执行LIME解释分析

    Args:
        model:          训练好的模型
        X_train:        训练特征
        X_test:         测试特征
        y_test:         测试标签
        feature_names:  特征名列表
        dataset_name:   数据集分组名
        project_name:   项目名
        model_name:     模型名
    """
    from explain.lime_module import LIMEAnalyzer

    lime_analysis = LIMEAnalyzer(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        dataset_name=dataset_name,
        project_name=project_name,
        model_name=model_name,
    )
    lime_analysis.run_all()


# ============================================================
# 5. 特征选择统计
# ============================================================

def compute_feature_selection_statistics(feature_dir, n_folds):
    """计算特征选择统计（频率、排名、选择率）并保存

    Args:
        feature_dir: 特征选择结果目录
        n_folds:     交叉验证折数
    """
    counter = Counter()
    for i in range(1, n_folds + 1):
        csv_file = os.path.join(feature_dir, f"fold_{i}.csv")
        df_feature = pd.read_csv(csv_file)
        counter.update(df_feature["Feature"])

    frequency_df = pd.DataFrame({
        "Feature": list(counter.keys()),
        "Frequency": list(counter.values()),
    })
    frequency_df = frequency_df.sort_values(by="Frequency", ascending=False)
    frequency_df.to_csv(
        os.path.join(feature_dir, "feature_frequency.csv"), index=False
    )

    statistics_df = frequency_df.copy()
    statistics_df["Selected_Rate"] = (
        statistics_df["Frequency"] / n_folds * 100
    ).round(2)
    statistics_df["Rank"] = range(1, len(statistics_df) + 1)
    statistics_df.to_csv(
        os.path.join(feature_dir, "feature_statistics.csv"), index=False
    )


# ============================================================
# 6. 单模型交叉验证训练
# ============================================================

def train_single_model(
    X, y, feature_names, model_name, model, cv,
    group, name,
    feature_selection_method='boruta',
    enable_shap=True,
    enable_lime=True,
):
    """对一个模型进行K折交叉验证训练

    包含：特征选择、Pipeline训练、SHAP/LIME分析、指标计算

    Args:
        X:                    特征矩阵（numpy array）
        y:                    标签向量（numpy array）
        feature_names:        特征名列表（用于构建DataFrame）
        model_name:           模型名称
        model:                模型实例
        cv:                   StratifiedKFold对象
        group:                数据集分组名（用于路径）
        name:                 数据集名（用于路径）
        feature_selection_method: 特征选择方法名
            可选: 'boruta'（Boruta特征选择）, 'none'（不使用特征选择）
            后续可扩展其他方法（需在 feature_selection_strategy 中注册）
        enable_shap:          是否执行SHAP分析（默认 True）
        enable_lime:          是否执行LIME分析（默认 True）

    Returns:
        fold_metrics:     每折指标列表
        y_true_all:       全部真实标签
        y_pred_all:       全部预测标签
        y_prob_all:       全部预测概率
        selected_features: 最后一次选中的特征（用于后续统计）
        feature_dir:      特征选择输出目录
    """
    fold_metrics = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    selected_features = None
    feature_dir = None

    for train, test in cv.split(X, y):
        # 构建训练/测试DataFrame
        X_train = pd.DataFrame(X[train], columns=feature_names)
        X_test = pd.DataFrame(X[test], columns=feature_names)
        y_train = y[train]
        y_test = y[test]

        # --- Feature Selection（统一策略接口）---
        selector, selected_features = run_feature_selection(
            X_train, y_train, method=feature_selection_method
        )
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        # 根据特征选择方法设置不同的输出目录
        feature_dir = get_feature_output_dir(
            feature_selection_method, group, name, model_name
        )
        os.makedirs(feature_dir, exist_ok=True)

        fold_id = len(fold_metrics) + 1
        save_selected_features(selected_features, feature_dir, fold_id)

        if feature_selection_method != 'none':
            print(f"Selected {len(selected_features)} Features")
            print(selected_features)

        # --- Pipeline ---
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)

        # --- SHAP（可配置开关）---
        if enable_shap:
            run_shap_explanation(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=X_train.columns.tolist(),
                dataset_name=group,
                project_name=name,
                model_name=model_name,
            )

        # --- LIME（可配置开关）---
        if enable_lime:
            run_lime_explanation(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=X_train.columns.tolist(),
                dataset_name=group,
                project_name=name,
                model_name=model_name,
            )

        # --- 预测与指标 ---
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(metrics)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    # --- 特征频率统计 ---
    compute_feature_selection_statistics(feature_dir, len(fold_metrics))

    return (
        fold_metrics, y_true_all, y_pred_all, y_prob_all,
        selected_features, feature_dir,
    )


# ============================================================
# 7. 可视化保存
# ============================================================

def save_model_visualizations(
    y_true_all, y_pred_all,
    model_probs, y_reference,
    group, name, model_name,
):
    """保存模型可视化结果（混淆矩阵、ROC/PR对比图）

    Args:
        y_true_all:  全部真实标签
        y_pred_all:  全部预测标签
        model_probs: {model_name: probs} 字典
        y_reference: 参考真实标签
        group:       数据集分组名
        name:        数据集名
        model_name:  模型名
    """
    filename = name + "_" + model_name
    save_CM(y_true_all, y_pred_all, filename)

    if y_reference is not None and len(model_probs) > 0:
        save_ROC_comparison(y_reference, model_probs, group, name)
        save_PR_comparison(y_reference, model_probs, group, name)
        print(f"Saved comparison curves for {name}")


# ============================================================
# 8. 结果汇总
# ============================================================

def save_final_results(all_results):
    """保存最终结果到CSV

    Args:
        all_results: 结果列表
    """
    pd.DataFrame(all_results).to_csv(
        "results/metrics/final_results.csv", index=False
    )


def run_feature_overall_summary(method_root_dirs):
    """运行特征总体统计（支持多个方法目录）

    Args:
        method_root_dirs: 方法根目录列表，如
            ['results/feature_selection/boruta', 'results/no_feature_selection']
            或更多自定义方法目录
    """
    for root_dir in method_root_dirs:
        if os.path.isdir(root_dir):
            summary = FeatureSummary(root_dir)
            summary.overall_summary()
        else:
            print(f"[Warning] Directory not found: {root_dir}")

    print("\nFinished!")


# ============================================================
# 9. 主实验流程编排
# ============================================================

def run_experiment_pipeline(
    feature_selection_method='boruta',
    enable_shap=True,
    enable_lime=True,
):
    """主实验流程编排入口

    协调数据加载、模型训练、可解释性分析、可视化、结果保存等各个模块。

    Args:
        feature_selection_method: 特征选择方法
            可选: 'boruta'（Boruta特征选择）, 'none'（不使用特征选择）
            后续可扩展其他方法，只需在 feature_selection_strategy.py 中注册即可
        enable_shap:  是否执行SHAP分析（默认 True）
        enable_lime:  是否执行LIME分析（默认 True）
    """
    # 0. 打印当前配置
    print("=" * 60)
    print(f"Feature selection method: {feature_selection_method}")
    print(f"SHAP enabled:             {enable_shap}")
    print(f"LIME enabled:             {enable_lime}")
    print("=" * 60)

    # 1. 初始化
    create_result_directories()

    models = get_models()
    all_results = []

    # 收集所有训练过的方法目录，用于后续特征总体统计
    trained_method_dirs = []

    # 2. 遍历数据集
    for group, names in config.datasets.items():
        for name in names:
            print(f"\nDataset: {name}")

            # 加载数据
            data = load_and_preprocess_data(group, name)
            if data is None:
                continue
            X, y, feature_names = data

            # 交叉验证策略
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            # 保存当前数据集所有模型的预测概率
            model_probs = {}
            y_reference = None

            # 3. 遍历模型
            for model_name, model in models.items():
                print(f"Model: {model_name}")

                # 训练单个模型
                (fold_metrics, y_true_all, y_pred_all, y_prob_all,
                 selected_features, feature_dir) = train_single_model(
                    X, y, feature_names, model_name, model, cv,
                    group, name,
                    feature_selection_method=feature_selection_method,
                    enable_shap=enable_shap,
                    enable_lime=enable_lime,
                )

                # 记录已训练的方法目录
                if feature_selection_method == 'none':
                    method_root = os.path.join("results", "no_feature_selection")
                else:
                    method_root = os.path.join(
                        "results", "feature_selection", feature_selection_method
                    )
                if method_root not in trained_method_dirs:
                    trained_method_dirs.append(method_root)

                # 保存模型预测概率
                model_probs[model_name] = np.asarray(y_prob_all)
                if y_reference is None:
                    y_reference = np.asarray(y_true_all)

                # 均值/标准差汇总
                result = pd.DataFrame(fold_metrics)
                row = {"Dataset": name, "Model": model_name}
                for col in result.columns:
                    row[col] = (
                        f"{result[col].mean():.4f}+"
                        f"{result[col].std():.4f}"
                    )
                all_results.append(row)

                # 保存可视化
                save_model_visualizations(
                    y_true_all, y_pred_all,
                    model_probs, y_reference,
                    group, name, model_name,
                )

    # 4. 保存最终结果
    save_final_results(all_results)

    # 5. 特征总体统计
    run_feature_overall_summary(trained_method_dirs)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    run_experiment_pipeline()