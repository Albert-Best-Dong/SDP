import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from cliffs_delta import cliffs_delta
from scipy.stats import mannwhitneyu


import numpy as np

def sign_based_cliffs_delta(x, y):
    """
    Sign-based Cliff's Delta for paired samples.

    Parameters
    ----------
    x, y : array-like
        Paired observations (same length)

    Returns
    -------
    delta : float
        Sign-based Cliff's Delta
    magnitude : str
        negligible / small / medium / large
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    diff = x - y

    wins = np.sum(diff > 0)
    losses = np.sum(diff < 0)
    ties = np.sum(diff == 0)

    delta = (wins - losses) / len(diff)

    abs_delta = abs(delta)

    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return delta, magnitude
# ==========================
# Load results
# ==========================

df = pd.read_csv("final_results.csv")

metric = "MCC"        # 可以改成 ROC-AUC

# Pivot
pivot = df.pivot(index="Dataset",
                 columns="Model",
                 values=metric)
pivot = pivot.map(
    lambda x: float(str(x).replace("±", "+").split("+")[0])
)
print(pivot)
delta, size = sign_based_cliffs_delta(
    pivot["RandomForest"],
    pivot["XGBoost"]
)
print("===========delta=======")
print(delta)
print(size)


# Average Rank: higher metric value = better rank
ranks = pivot.rank(
    axis=1,
    ascending=False,
    method="average"
)

avg_ranks = ranks.mean(axis=0).sort_values()

print("\nAverage Rank (lower is better):")
print(avg_ranks.round(3))


# ==========================
# Friedman Test
# ==========================
stat, p = friedmanchisquare(
    pivot["RandomForest"],
    pivot["XGBoost"],
    pivot["LightGBM"],
    pivot["CatBoost"]
)

print("="*50)
print("Friedman Test")
print(f"Statistic = {stat:.4f}")
print(f"P-value   = {p:.6f}")
print("="*50)

# ==========================
# Pairwise Wilcoxon
# ==========================
models = list(pivot.columns)

results = []

p_values = []

for i in range(len(models)):
    for j in range(i+1, len(models)):

        stat, p = wilcoxon(
            pivot[models[i]],
            pivot[models[j]]
        )

        results.append([
            models[i],
            models[j],
            stat,
            p
        ])

        p_values.append(p)

# Holm correction
reject, corrected_p, _, _ = multipletests(
    p_values,
    method='holm'
)

for i in range(len(results)):
    results[i].append(corrected_p[i])
    results[i].append(reject[i])

result_df = pd.DataFrame(
    results,
    columns=[
        "Model A",
        "Model B",
        "Wilcoxon Statistic",
        "Raw p",
        "Holm p",
        "Significant"
    ]
)

print("===========result_df=======")
print(result_df)

result_df.to_csv(
    "statistical_significance.csv",
    index=False
)

print("Saved -> statistical_significance.csv")