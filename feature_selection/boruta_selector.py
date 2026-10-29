
"""
==============================================================
Boruta Feature Selector

Author : Albert
==============================================================
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy

from feature_selection.base_selector import BaseSelector


class BorutaSelector(BaseSelector):

    def __init__(
        self,
        random_state=42,
        max_iter=100,
        verbose=0,
    ):
        super().__init__()

        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose

        self.rf = RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        self.selector = BorutaPy(
            estimator=self.rf,
            n_estimators="auto",
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
        )

    ###############################################################
    # Fit
    ###############################################################

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()

        # 统一转为 numpy array
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = np.asarray(y)

        self.selector.fit(X_np, y_np)

        mask = self.selector.support_

        # Fallback 1: Boruta 未选中任何特征 -> 使用 weak features
        if mask.sum() == 0:
            print("Boruta selected 0 features.")
            print("Fallback to weak features...")
            mask = self.selector.support_ | self.selector.support_weak_

        # Fallback 2: 仍然为 0 -> 使用全部特征
        if mask.sum() == 0:
            print("Still 0 features.")
            print("Fallback to all features.")
            mask = np.ones(len(X.columns), dtype=bool)

        self.support_mask_ = mask
        self.selected_features_ = list(X.columns[self.support_mask_])

        return self

    ###############################################################
    # Transform
    ###############################################################

    def transform(self, X):
        return X.loc[:, self.selected_features_]

    ###############################################################
    # Ranking
    ###############################################################

    def get_feature_ranking(self):
        ranking = pd.DataFrame({
            "Feature": self.feature_names,
            "Ranking": self.selector.ranking_,
        })
        ranking = ranking.sort_values("Ranking")
        return ranking