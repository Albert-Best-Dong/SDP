
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)


# 创建输出目录
def create_result_dirs():
    for folder in [
        "results/roc_curve",
        "results/pr_curve",
        "results/confusion_matrix",
        "results/metrics"
    ]:
        os.makedirs(folder, exist_ok=True)


# 同一数据集：四个模型的 ROC 对比图
def save_ROC_comparison(y_true, model_probs, dataset_name, project_name):
    create_result_dirs()

    plt.figure(figsize=(7, 6))

    for model_name, y_prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {auc_value:.3f})"
        )

    # 随机分类器参考线
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {project_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        f"results/roc_curve/{dataset_name}_{project_name}_ROC.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# 同一数据集：四个模型的 PR 对比图
def save_PR_comparison(y_true, model_probs, dataset_name,project_name):
    create_result_dirs()

    plt.figure(figsize=(7, 6))

    for model_name, y_prob in model_probs.items():
        precision, recall, _ = precision_recall_curve(
            y_true, y_prob
        )

        ap_value = average_precision_score(y_true, y_prob)

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (AP = {ap_value:.3f})"
        )

    # 随机排序的基准线：正类比例
    positive_rate = (y_true == 1).mean()
    plt.axhline(
        y=positive_rate,
        linestyle="--",
        linewidth=1,
        label=f"Positive-class rate = {positive_rate:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {project_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        f"results/pr_curve/{dataset_name}_{project_name}_PR.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# 保存混淆矩阵 CSV
def save_CM(y_true, y_pred, name):
    create_result_dirs()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pd.DataFrame(
        cm,
        index=["Actual_Non-defective", "Actual_Defective"],
        columns=["Predicted_Non-defective", "Predicted_Defective"]
    ).to_csv(
        f"results/confusion_matrix/{name}.csv",
        index=True
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Predicted_Non-defective", "Predicted_Defective"], yticklabels=["Actual_Non-defective", "Actual_Defective"],)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')