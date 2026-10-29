# cross_repo_shap_scheme2.py
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置参数 ==========
TOP_K = 10
INPUT_CSV = "Repository_SHAP_Importance.csv"

# 【和方案1完全一致的特征→度量族映射，保证两套方案对齐】
feature_family_map = {
    # ---------------- AEEEM features ----------------
    "ck_oo_wmc": "OO Metrics",
    "ck_oo_cbo": "OO Metrics",
    "ck_oo_rfc": "OO Metrics",
    "ck_oo_dit": "OO Metrics",
    "ck_oo_noc": "OO Metrics",
    "ck_oo_lcom": "OO Metrics",
    "ck_oo_numberOfAttributes": "OO Metrics",
    "ck_oo_numberOfPublicAttributes": "OO Metrics",
    "ck_oo_numberOfPrivateAttributes": "OO Metrics",
    "ck_oo_numberOfMethods": "OO Metrics",
    "ck_oo_numberOfPublicMethods": "OO Metrics",
    "ck_oo_numberOfPrivateMethods": "OO Metrics",
    "ck_oo_numberOfAttributesInherited": "OO Metrics",
    "ck_oo_numberOfMethodsInherited": "OO Metrics",
    "ck_oo_fanIn": "OO Metrics",
    "ck_oo_fanOut": "OO Metrics",
    "ck_oo_numberOfLinesOfCode": "Size",

    "LDHH_wmc": "OO Metrics",
    "LDHH_cbo": "OO Metrics",
    "LDHH_rfc": "OO Metrics",
    "LDHH_dit": "OO Metrics",
    "LDHH_noc": "OO Metrics",
    "LDHH_lcom": "OO Metrics",
    "LDHH_numberOfAttributes": "OO Metrics",
    "LDHH_numberOfPublicAttributes": "OO Metrics",
    "LDHH_numberOfPrivateAttributes": "OO Metrics",
    "LDHH_numberOfMethods": "OO Metrics",
    "LDHH_numberOfPublicMethods": "OO Metrics",
    "LDHH_numberOfPrivateMethods": "OO Metrics",
    "LDHH_numberOfAttributesInherited": "OO Metrics",
    "LDHH_numberOfMethodsInherited": "OO Metrics",
    "LDHH_fanIn": "OO Metrics",
    "LDHH_fanOut": "OO Metrics",
    "LDHH_numberOfLinesOfCode": "Size",

    "WCHU_wmc": "OO Metrics",
    "WCHU_cbo": "OO Metrics",
    "WCHU_rfc": "OO Metrics",
    "WCHU_dit": "OO Metrics",
    "WCHU_noc": "OO Metrics",
    "WCHU_lcom": "OO Metrics",
    "WCHU_numberOfAttributes": "OO Metrics",
    "WCHU_numberOfPublicAttributes": "OO Metrics",
    "WCHU_numberOfPrivateAttributes": "OO Metrics",
    "WCHU_numberOfMethods": "OO Metrics",
    "WCHU_numberOfPublicMethods": "OO Metrics",
    "WCHU_numberOfPrivateMethods": "OO Metrics",
    "WCHU_numberOfAttributesInherited": "OO Metrics",
    "WCHU_numberOfMethodsInherited": "OO Metrics",
    "WCHU_fanIn": "OO Metrics",
    "WCHU_fanOut": "OO Metrics",
    "WCHU_numberOfLinesOfCode": "Size",

    "CvsEntropy": "Evolution Metrics",
    "CvsWEntropy": "Evolution Metrics",
    "CvsLogEntropy": "Evolution Metrics",
    "CvsExpEntropy": "Evolution Metrics",
    "CvsLinEntropy": "Evolution Metrics",

    "numberOfBugsFoundUntil:": "Bug History",
    "numberOfNonTrivialBugsFoundUntil:": "Bug History",
    "numberOfMajorBugsFoundUntil:": "Bug History",
    "numberOfCriticalBugsFoundUntil:": "Bug History",
    "numberOfHighPriorityBugsFoundUntil:": "Bug History",

    # ---------------- NASA / PROMISE features (Halstead, cyclomatic, size) ----------------
    "LOC_COMMENTS": "Comment & Blank",
    "LOC_BLANK": "Comment & Blank",
    "LOC_CODE_AND_COMMENT": "Size",
    "LOC_EXECUTABLE": "Size",
    "LOC_TOTAL": "Size",
    "NUMBER_OF_LINES": "Size",
    "PERCENT_COMMENTS": "Comment & Blank",

    "CYCLOMATIC_COMPLEXITY": "Cyclomatic & Structural Complexity",
    "CYCLOMATIC_DENSITY": "Cyclomatic & Structural Complexity",
    "ESSENTIAL_COMPLEXITY": "Cyclomatic & Structural Complexity",
    "ESSENTIAL_DENSITY": "Cyclomatic & Structural Complexity",
    "DESIGN_COMPLEXITY": "Cyclomatic & Structural Complexity",
    "DESIGN_DENSITY": "Cyclomatic & Structural Complexity",
    "DECISION_COUNT": "Cyclomatic & Structural Complexity",
    "DECISION_DENSITY": "Cyclomatic & Structural Complexity",
    "CONDITION_COUNT": "Cyclomatic & Structural Complexity",
    "MODIFIED_CONDITION_COUNT": "Cyclomatic & Structural Complexity",
    "MULTIPLE_CONDITION_COUNT": "Cyclomatic & Structural Complexity",
    "BRANCH_COUNT": "Cyclomatic & Structural Complexity",
    "EDGE_COUNT": "Cyclomatic & Structural Complexity",
    "NODE_COUNT": "Cyclomatic & Structural Complexity",
    "CALL_PAIRS": "Cyclomatic & Structural Complexity",

    "HALSTEAD_LENGTH": "Halstead & Lexical",
    "HALSTEAD_VOLUME": "Halstead & Lexical",
    "HALSTEAD_DIFFICULTY": "Halstead & Lexical",
    "HALSTEAD_CONTENT": "Halstead & Lexical",
    "HALSTEAD_EFFORT": "Halstead & Lexical",
    "HALSTEAD_PROG_TIME": "Halstead & Lexical",
    "HALSTEAD_ERROR_EST": "Halstead & Lexical",
    "NUM_UNIQUE_OPERATORS": "Halstead & Lexical",
    "NUM_UNIQUE_OPERANDS": "Halstead & Lexical",
    "NUM_OPERATORS": "Halstead & Lexical",
    "NUM_OPERANDS": "Halstead & Lexical",
    "MAINTENANCE_SEVERITY": "Halstead & Lexical",
    "NORMALIZED_CYLOMATIC_COMPLEXITY": "Cyclomatic & Structural Complexity",

    # PROMISE short‑name metrics
    "lOComment": "Comment & Blank",
    "lOBlank": "Comment & Blank",
    "loc": "Size",
    "lOCode": "Size",
    "locCodeAndComment": "Size",
    "v(g)": "Cyclomatic & Structural Complexity",
    "ev(g)": "Cyclomatic & Structural Complexity",
    "iv(g)": "Cyclomatic & Structural Complexity",
    "branchCount": "Cyclomatic & Structural Complexity",
    "uniq_Op": "Halstead & Lexical",
    "uniq_Opnd": "Halstead & Lexical",
    "total_Op": "Halstead & Lexical",
    "total_Opnd": "Halstead & Lexical",
    "n": "Halstead & Lexical",
    "v": "Halstead & Lexical",
    "e": "Halstead & Lexical",
    "d": "Halstead & Lexical",
    "i": "Halstead & Lexical",
    "b": "Halstead & Lexical",
    "t": "Halstead & Lexical",
    "l": "Halstead & Lexical",

    # ---------------- ReLink features ----------------
    "CountLine": "Size",
    "CountLineCode": "Size",
    "CountLineCodeExe": "Size",
    "CountLineCodeDecl": "Size",
    "CountLineBlank": "Comment & Blank",
    "CountLineComment": "Comment & Blank",
    "RatioCommentToCode": "Comment & Blank",
    "CountStmt": "Size",
    "CountStmtDecl": "Size",
    "CountStmtExe": "Size",
    "CountSemicolon": "Size",
    "AvgLine": "Size",
    "AvgLineCode": "Size",
    "AvgLineBlank": "Comment & Blank",
    "AvgLineComment": "Comment & Blank",
    "MaxCyclomatic": "Cyclomatic & Structural Complexity",
    "MaxCyclomaticModified": "Cyclomatic & Structural Complexity",
    "MaxCyclomaticStrict": "Cyclomatic & Structural Complexity",
    "SumCyclomatic": "Cyclomatic & Structural Complexity",
    "SumCyclomaticModified": "Cyclomatic & Structural Complexity",
    "SumCyclomaticStrict": "Cyclomatic & Structural Complexity",
    "AvgCyclomatic": "Cyclomatic & Structural Complexity",
    "AvgCyclomaticModified": "Cyclomatic & Structural Complexity",
    "AvgCyclomaticStrict": "Cyclomatic & Structural Complexity",
    "SumEssential": "Cyclomatic & Structural Complexity",
    "AvgEssential": "Cyclomatic & Structural Complexity"
}

FAMILIES = [
    "Size",
    "Comment & Blank",
    "Cyclomatic & Structural Complexity",
    "Halstead & Lexical",
    "OO Metrics",
    "Evolution Metrics",
    "Bug History"
]

# ========== 读取数据 ==========
df_raw = pd.read_csv(INPUT_CSV)
df_raw["metric_family"] = df_raw["feature"].map(feature_family_map).fillna("Unknown")

# 同一项目‑特征，多个模型取SHAP均值
df_proj = df_raw.groupby(["repository", "project", "feature", "metric_family"])["importance"].mean().reset_index()

# 每个project内部按importance降序，取Top‑K
topk_list = []
for (repo, proj), group in df_proj.groupby(["repository", "project"]):
    grp_sorted = group.sort_values("importance", ascending=False).head(TOP_K).copy()
    topk_list.append(grp_sorted)

df_topk = pd.concat(topk_list, ignore_index=True)

# 统计每个仓库下，各个度量族出现频次
count_df = df_topk.groupby(["repository", "metric_family"]).size().unstack(fill_value=0)
count_df = count_df.reindex(columns=FAMILIES, fill_value=0)
count_df.to_csv("repo_topk_family_count.csv")

print(f"===== Scheme2: Top‑{TOP_K} metric‑family occurrence count per repository =====")
print(count_df)

# ========== 绘图：堆叠柱状图（正文主图） ==========
fig, ax = plt.subplots(figsize=(10, 6))
count_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
ax.set_ylabel(f"Occurrence Count (Top‑{TOP_K} features per project)")
ax.set_xlabel("Software Repository")
ax.set_title(f"Occurrence of Metric Families among Top‑{TOP_K} SHAP Important Features")
ax.legend(title="Metric Family", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig_topk_stacked_bar.png", dpi=300, bbox_inches="tight")

print("\n[DONE] Output: repo_topk_family_count.csv , fig_topk_stacked_bar.png")
