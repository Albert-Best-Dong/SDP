"""
==============================================================
Base Feature Selector

Author : Albert
Paper  : A Robust and Explainable Framework for Cross-Project
         Software Defect Prediction
==============================================================
"""

from abc import ABC, abstractmethod


class BaseSelector(ABC):
    """
    Base class of all feature selectors.
    """

    def __init__(self):

        self.selected_features_ = None

        self.support_mask_ = None

    @abstractmethod
    def fit(self, X, y):
        """
        Fit selector using training data.
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Transform dataset.
        """
        pass

    def fit_transform(self, X, y):

        self.fit(X, y)

        return self.transform(X)

    def get_support(self):
        """
        Return boolean mask.
        """

        return self.support_mask_

    def get_selected_features(self):
        """
        Return selected feature names.
        """

        return self.selected_features_