RESULT_PATH="results"

DATA_PATH="datasets"

RANDOM_SEED=42

N_SPLITS=10

TEST_SIZE=0.2

USE_STANDARD=True

USE_SMOTE=False

USE_GRID_SEARCH=False

SAVE_SHAP=True

SAVE_LIME=True

SAVE_MODEL=False

USE_FEATURE_SELECTION=True
USE_BORUTA = False

USE_LIME = False
# ===============================
# 数据集配置
# ===============================


datasets={


"NASA":[
"CM1",
"JM1",
"KC1",
"KC2",
"MW1",
"PC1",
"PC3",
"PC4"
],
"PROMISE":[
    "cm1",
    "jm1",
    "kc1",
    "kc2",
    "pc1"
],
"AEEEM":[
"EQ",
"JDT"
"LC",
"ML",
"PDE"
],


"ReLink":["Apache",
"Safe",
"ZXing"
]


}

# ===============================
# 查找标签列
# ===============================

def get_label(df):


    labels=[
        "bug",
        "Bug",
        "defect",
        "Defective",
        "isDefective",
        "label",
        "class", "defects","problems"
    ]


    for col in labels:

        if col in df.columns:

            return col


    raise Exception(
        "No label column found"
    )


def encode_labels(series, dataset_name):
    """
    统一缺陷标签：
    0 = 无缺陷
    1 = 有缺陷
    """
    # 去除字符串首尾空格，并统一大小写
    labels = series.astype(str).str.strip().str.lower()

    # NASA: Defective，N = 无缺陷，Y = 有缺陷
    if dataset_name == "NASA":
        mapping = {
            "n": 0,
            "y": 1
        }
    elif dataset_name == "PROMISE":
        mapping = {
            "false": 0,
            "true": 1,
            "yes":1,
            "no":0
        }
    # AEEEM: class，clean = 无缺陷，buggy = 有缺陷
    elif dataset_name == "AEEEM":
        mapping = {
            "clean": 0,
            "buggy": 1
        }

    # ReLink: isDefective，clean = 无缺陷，buggy = 有缺陷
    elif dataset_name == "ReLink":
        mapping = {
            "clean": 0,
            "buggy": 1
        }

    else:
        raise ValueError(f"未知的数据集类型：{dataset_name}")

    y = labels.map(mapping)

    # 检查是否存在未识别的标签，防止静默转换出错
    if y.isna().any():
        unknown = series[y.isna()].unique().tolist()
        raise ValueError(
            f"{dataset_name} 数据集存在无法识别的标签：{unknown}"
        )

    y = y.astype(int)

    # 输出转换结果，便于检查
    print(f"{dataset_name} 标签转换结果：")
    print(y.value_counts().sort_index())

    return y

# ===============================
# 查找标签列
# ===============================

def get_label(df):


    labels=[
        "bug",
        "Bug",
        "defect",
        "Defective",
        "isDefective",
        "label",
        "class", "defects","problems"
    ]


    for col in labels:

        if col in df.columns:

            return col


    raise Exception(
        "No label column found"
    )


def encode_labels(series, dataset_name):
    """
    统一缺陷标签：
    0 = 无缺陷
    1 = 有缺陷
    """
    # 去除字符串首尾空格，并统一大小写
    labels = series.astype(str).str.strip().str.lower()

    # NASA: Defective，N = 无缺陷，Y = 有缺陷
    if dataset_name == "NASA":
        mapping = {
            "n": 0,
            "y": 1
        }
    elif dataset_name == "PROMISE":
        mapping = {
            "false": 0,
            "true": 1,
            "yes":1,
            "no":0
        }
    # AEEEM: class，clean = 无缺陷，buggy = 有缺陷
    elif dataset_name == "AEEEM":
        mapping = {
            "clean": 0,
            "buggy": 1
        }

    # ReLink: isDefective，clean = 无缺陷，buggy = 有缺陷
    elif dataset_name == "ReLink":
        mapping = {
            "clean": 0,
            "buggy": 1
        }

    else:
        raise ValueError(f"未知的数据集类型：{dataset_name}")

    y = labels.map(mapping)

    # 检查是否存在未识别的标签，防止静默转换出错
    if y.isna().any():
        unknown = series[y.isna()].unique().tolist()
        raise ValueError(
            f"{dataset_name} 数据集存在无法识别的标签：{unknown}"
        )

    y = y.astype(int)

    # 输出转换结果，便于检查
    print(f"{dataset_name} 标签转换结果：")
    print(y.value_counts().sort_index())

    return y