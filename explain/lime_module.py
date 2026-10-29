import os
import numpy as np
import pandas as pd

from pathlib import Path

from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt



class LIMEAnalyzer:


    def __init__(

        self,

        model,

        X_train,

        X_test,

        y_test,

        feature_names,

        dataset_name,

        project_name,

        model_name,

        result_root="results"

    ):


        self.model = model

        self.X_train = X_train

        self.X_test = X_test

        self.y_test = y_test


        self.feature_names = list(
            feature_names
        )

        self.y_pred = self.model.predict(
            self.X_test.values
        )

        self.y_prob = self.model.predict_proba(
            self.X_test.values
        )[:, 1]
        # 保存路径

        self.output_dir = (

            Path(result_root)

            / dataset_name

            / project_name

            / model_name

            / "LIME"

        )


        self.output_dir.mkdir(

            parents=True,

            exist_ok=True

        )


        # 初始化LIME

        self.explainer = LimeTabularExplainer(

            training_data=np.asarray(
                X_train
            ),

            feature_names=self.feature_names,

            class_names=[
                "Non-defective",
                "Defective"
            ],

            mode="classification",

            discretize_continuous=True

        )



    # ====================================
    # 自动选择解释样本
    # ====================================

    # def select_samples(self):
    #
    #     prob = self.model.predict_proba(
    #
    #         self.X_test
    #
    #     )[:,1]
    #
    #
    #     high_risk = np.argmax(
    #         prob
    #     )
    #
    #
    #     low_risk = np.argmin(
    #         prob
    #     )
    #
    #
    #     boundary = np.argmin(
    #
    #         np.abs(
    #             prob-0.5
    #         )
    #
    #     )
    #
    #
    #     return {
    #
    #         "high_risk": high_risk,
    #
    #         "low_risk": low_risk,
    #
    #         "boundary": boundary
    #
    #     }
    def select_samples(self):

        samples = {}

        # =====================================================
        # High Risk
        # Highest probability defective sample
        # =====================================================
        high_risk_indices = np.where(
            self.y_pred == 1
        )[0]

        if len(high_risk_indices) > 0:
            samples["High_Risk"] = high_risk_indices[
                np.argmax(
                    self.y_prob[high_risk_indices]
                )
            ]

        # =====================================================
        # Low Risk
        # Lowest probability clean sample
        # =====================================================
        low_risk_indices = np.where(
            self.y_pred == 0
        )[0]

        if len(low_risk_indices) > 0:
            samples["Low_Risk"] = low_risk_indices[
                np.argmin(
                    self.y_prob[low_risk_indices]
                )
            ]

        # =====================================================
        # Boundary
        # Probability close to 0.5
        # =====================================================
        boundary_idx = np.argmin(
            np.abs(
                self.y_prob - 0.5
            )
        )

        samples["Boundary"] = boundary_idx

        # =====================================================
        # True Positive
        # Correctly predicted defective modules
        # =====================================================
        tp_indices = np.where(
            (self.y_test == 1) &
            (self.y_pred == 1)
        )[0]

        if len(tp_indices) > 0:
            samples["TP"] = tp_indices[
                np.argmax(
                    self.y_prob[tp_indices]
                )
            ]

        # =====================================================
        # True Negative
        # Correctly predicted clean modules
        # =====================================================
        tn_indices = np.where(
            (self.y_test == 0) &
            (self.y_pred == 0)
        )[0]

        if len(tn_indices) > 0:
            samples["TN"] = tn_indices[
                np.argmin(
                    self.y_prob[tn_indices]
                )
            ]

        # =====================================================
        # False Positive
        # Predicted defective but actually clean
        # =====================================================
        fp_indices = np.where(
            (self.y_test == 0) &
            (self.y_pred == 1)
        )[0]

        if len(fp_indices) > 0:
            samples["FP"] = fp_indices[
                np.argmax(
                    self.y_prob[fp_indices]
                )
            ]

        # =====================================================
        # False Negative
        # Predicted clean but actually defective
        # =====================================================
        fn_indices = np.where(
            (self.y_test == 1) &
            (self.y_pred == 0)
        )[0]

        if len(fn_indices) > 0:
            samples["FN"] = fn_indices[
                np.argmin(
                    self.y_prob[fn_indices]
                )
            ]

        return samples
    # =====================================================
    # Select True Positive samples
    # Correctly predicted defective modules
    # =====================================================
    def select_TP(self):

        indices = np.where(
            (self.y_test == 1) &
            (self.y_pred == 1)
        )[0]

        if len(indices) == 0:
            print("No TP samples found.")
            return None

        # choose highest confidence TP
        selected_idx = indices[
            np.argmax(
                self.y_prob[indices]
            )
        ]

        return selected_idx

    # =====================================================
    # Select True Negative samples
    # Correctly predicted clean modules
    # =====================================================
    def select_TN(self):

        indices = np.where(
            (self.y_test == 0) &
            (self.y_pred == 0)
        )[0]

        if len(indices) == 0:
            print("No TN samples found.")
            return None

        # choose lowest defect probability TN
        selected_idx = indices[
            np.argmin(
                self.y_prob[indices]
            )
        ]

        return selected_idx

    # =====================================================
    # Select False Positive samples
    # Predicted defective but actually clean
    # =====================================================
    def select_FP(self):

        indices = np.where(
            (self.y_test == 0) &
            (self.y_pred == 1)
        )[0]

        if len(indices) == 0:
            print("No FP samples found.")
            return None

        # choose strongest false alarm
        selected_idx = indices[
            np.argmax(
                self.y_prob[indices]
            )
        ]

        return selected_idx

    # =====================================================
    # Select False Negative samples
    # Predicted clean but actually defective
    # =====================================================
    def select_FN(self):

        indices = np.where(
            (self.y_test == 1) &
            (self.y_pred == 0)
        )[0]

        if len(indices) == 0:
            print("No FN samples found.")
            return None

        # choose most dangerous missed defect
        selected_idx = indices[
            np.argmin(
                self.y_prob[indices]
            )
        ]

        return selected_idx
    # ====================================
    # 预测函数
    # ====================================

    # def predict_fn(self, data):
    #
    #
    #     data_df = pd.DataFrame(
    #
    #         data,
    #
    #         columns=self.feature_names
    #
    #     )
    #
    #
    #     return self.model.predict_proba(
    #         data_df
    #     )

    def predict_fn(self, data):
        data_df = pd.DataFrame(
            data,
            columns=self.feature_names
        )

        return self.model.predict_proba(
            data_df
        )

    # ====================================
    # 单样本LIME
    # ====================================

    def explain_sample(

            self,

            sample_id,

            name

    ):


        instance = np.asarray(

            self.X_test.iloc[sample_id]

        )


        explanation = self.explainer.explain_instance(

            data_row=instance,

            predict_fn=self.predict_fn,

            num_features=15
            # num_features=len(
            #     self.feature_names
            # )

        )


        # ==========================
        # HTML
        # ==========================


        html_path = (

            self.output_dir

            / f"lime_{name}.html"

        )


        explanation.save_to_file(

            str(html_path)

        )



        # ==========================
        # 权重保存
        # ==========================


        weights = explanation.as_list(

            label=1

        )


        df = pd.DataFrame(

            weights,

            columns=[

                "Feature",

                "Weight"

            ]

        )


        df.to_csv(

            self.output_dir

            /

            f"lime_{name}_weights.csv",

            index=False,

            encoding="utf-8-sig"

        )



        # ==========================
        # 图片
        # ==========================


        fig = explanation.as_pyplot_figure(

            label=1

        )


        plt.title(

            f"LIME Explanation - {name}"

        )


        plt.tight_layout()


        plt.savefig(

            self.output_dir

            /

            f"lime_{name}.png",

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



        print(

            f"LIME saved: {name}"

        )



    # ====================================
    # 全流程
    # ====================================

    def run_all(self):


        samples = self.select_samples()


        for name, idx in samples.items():


            self.explain_sample(

                sample_id=idx,

                name=name

            )


        print(

            "LIME analysis completed."

        )