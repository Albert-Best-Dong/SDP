from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
def collect_all_data(results_dir: str) -> pd.DataFrame:
    """
    扫描目录，收集所有 shap_importance.csv 并拼接为长表。
    结果包含四列: repository, project, feature, importance
    """
    records = []
    results_path = Path(results_dir)

    for csv_file in results_path.rglob("shap_importance.csv"):
        try:
            # 1. 尝试读取文件
            df = pd.read_csv(csv_file)

            # 2. 如果没有预期的列名，强制重命名（假设前两列是特征和重要性）
            if "Feature" not in df.columns and "feature" not in df.columns:
                if df.shape[1] >= 2:
                    df.columns = ["Feature", "Importance"] + [f"Extra_{i}" for i in range(df.shape[1] - 2)]
                else:
                    print(f"[Warning] 列数不足，跳过文件: {csv_file}")
                    continue

            # 统一列名为小写
            df.columns = [str(c).strip().lower() for c in df.columns]

            # 【核心修复】强制清洗 Importance 列
            # 如果原始数据是字符串（如 "0.05" 或 "5%"），先转字符串再去掉非数字字符，最后转浮点数
            df["importance"] = (
                df["importance"]
                .astype(str)
                .str.replace('%', '', regex=False)  # 去掉百分号
                .str.strip()  # 去掉首尾空格
            )
            df["importance"] = pd.to_numeric(df["importance"], errors="coerce")

            # 丢弃重要性为 NaN 的行
            df = df.dropna(subset=["importance"])

            # 如果清洗后没有有效数据，跳过该文件
            if df.empty:
                print(f"[Warning] 清洗后无有效数值数据，跳过文件: {csv_file}")
                continue

            # 3. 从路径解析 Repository 和 Project
            project = csv_file.parents[2].name
            repository = csv_file.parents[3].name

            df["repository"] = repository
            df["project"] = project

            # 仅保留需要的四列
            records.append(df[["repository", "project", "feature", "importance"]])

        except Exception as e:
            print(f"[Warning] 读取文件失败 {csv_file}: {e}")
            continue

    if not records:
        raise ValueError(f"未在 {results_dir} 中找到任何有效的 shap_importance.csv 文件")

    final_df = pd.concat(records, ignore_index=True)
    print(f"[Info] 成功收集数据，总行数: {len(final_df)}, 唯一特征数: {final_df['feature'].nunique()}")
    return final_df
"""
According to the no_FS results, gathering all the shap_importance.csv files together into Repository_SHAP_Importance.csv
"""
if __name__ == "__main__":
    collect_all_data("../../results").to_csv("Repository_SHAP_Importance.csv", index=False)
    # SHAPGlobalAnalysis("results", "RQ3_Output").run()