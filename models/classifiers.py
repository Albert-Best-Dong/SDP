from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def get_models():


    models={

        "RandomForest":
        RandomForestClassifier(
            n_estimators=200,
            # class_weight='balanced',
            random_state=42
        ),


        "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            min_child_weight=3,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
            # scale_pos_weight = 1
        ),


        "LightGBM":
        LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            # class_weight='balanced',
            random_state=42
        ),


        "CatBoost":
        CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            verbose=0,
            # auto_class_weights='Balanced',
            random_state=42
        )

    }


    return models