from pathlib import Path

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from myutils import result_manager


class SHAPAnalyzer:

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

        self.feature_names = feature_names
        self.project_name = project_name
        self.output_dir =  Path(result_root) / dataset_name / project_name / model_name / "SHAP"

        # os.makedirs(
        #     self.output_dir,
        #     exist_ok=True
        # )
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.explainer = None
        self.shap_values = None

    def get_values(self):

        if hasattr(
                self.shap_values,
                "values"
        ):

            return self.shap_values.values

        else:

            return self.shap_values
    # ==========================
    # TreeSHAP
    # ==========================

    # def build_explainer(self):
    #
    #     print("Building TreeSHAP model...")
    #
    #     self.explainer = shap.TreeExplainer(
    #         self.model
    #     )
    #
    #
    #     self.shap_values = self.explainer(
    #         self.X_test
    #     )
    #     print(
    #         self.shap_values.shape
    #     )
    #
    #     print(
    #         "SHAP calculation finished"
    #     )
    #
    #     return self.shap_values

    # def build_explainer(self):
    #
    #     print("Building TreeSHAP...")
    #
    #     self.explainer = shap.TreeExplainer(
    #         self.model
    #     )
    #
    #     shap_values = self.explainer(
    #         self.X_test
    #     )
    #
    #     # 二分类模型兼容
    #     if len(shap_values.shape) == 3:
    #         print(
    #             "Binary classification detected"
    #         )
    #
    #         # 取 defective=1 类
    #         shap_values = shap_values[:, :, 1]
    #
    #     self.shap_values = shap_values
    #
    #     print(
    #         "SHAP shape:",
    #         self.shap_values.shape
    #     )
    #
    #     return self.shap_values
    def build_explainer(self):

        print("Building TreeSHAP...")

        self.explainer = shap.TreeExplainer(
            self.model
        )

        # 新版SHAP对象
        shap_exp = self.explainer(
            self.X_test
        )

        # self.shap_explanation = shap_exp
        # ==========================
        # 二分类处理
        # ==========================

        if len(shap_exp.values.shape) == 3:

            # 只解释 Defective=1

            self.shap_explanation = shap.Explanation(

                values=shap_exp.values[:, :, 1],

                base_values=shap_exp.base_values[:, 1],

                data=shap_exp.data,

                feature_names=self.feature_names

            )

            self.shap_values = (
                shap_exp.values[:, :, 1]
            )

            self.expected_value = (
                shap_exp.base_values[0, 1]
                if len(shap_exp.base_values.shape) > 1
                else shap_exp.base_values[1]
            )


        else:

            self.shap_explanation = shap_exp

            self.shap_values = shap_exp.values

            self.expected_value = (
                shap_exp.base_values[0]
                if hasattr(shap_exp, "base_values")
                else self.explainer.expected_value
            )
        # 保存numpy版本
        values = shap_exp.values

        print(
            "Original SHAP:",
            values.shape
        )

        # 二分类
        if len(values.shape) == 3:

            # Defective=1

            self.shap_values = values[:, :, 1]

            self.expected_value = (
                self.explainer.expected_value[1]
            )


        else:

            self.shap_values = values

            self.expected_value = (
                self.explainer.expected_value
            )

        print(
            "Numpy SHAP:",
            self.shap_values.shape
        )

        print(
            "Explanation shape:",
            self.shap_explanation.shape
        )

        print(
            "Values shape:",
            self.shap_values.shape
        )
        return self.shap_values

    # ==========================
    # 1 Global Importance
    # ==========================

    # def global_importance(self):
    #
    #     plt.figure()
    #
    #     shap.plots.bar(
    #         self.shap_values,
    #         show=False
    #     )
    #
    #
    #     plt.title(
    #         "Global Feature Importance"
    #     )
    #
    #
    #     plt.savefig(
    #         f"{self.output_dir}/bar_importance.png",
    #         dpi=300,
    #         bbox_inches="tight"
    #     )
    #
    #     plt.close()

    def global_importance(self):

        importance = np.abs(
            self.get_values()
            if hasattr(self.shap_values, "values")
            else self.shap_values
        ).mean(axis=0)

        feature_importance = pd.DataFrame({
            "Feature":
                self.feature_names,

            "Importance":
                importance
        })

        feature_importance.sort_values(
            "Importance",
            ascending=False,
            inplace=True
        )
        # 只留15个，给JDT做测试
        # feature_importance = feature_importance.head(15)

        plt.figure(figsize=(8, 6))

        plt.barh(
            feature_importance["Feature"],
            feature_importance["Importance"]
        )

        plt.gca().invert_yaxis()

        plt.title(
            "SHAP Feature Importance"
        )

        plt.savefig(
            self.output_dir/"bar_importance.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        feature_importance = feature_importance.set_index("Feature")

        feature_importance.columns = ["Importance"]

        feature_importance.to_csv(
            self.output_dir / "shap_importance.csv"
        )
    # ==========================
    # 2 Beeswarm
    # ==========================

    def beeswarm(self):

        plt.figure()

        shap.plots.beeswarm(

            self.shap_explanation,

            max_display=20,

            show=False

        )

        plt.title(
            "SHAP Beeswarm Plot"
        )

        plt.savefig(

            self.output_dir/"beeswarm.png",
            # result_manager.ResultManager.file("SHAP","beeswarm.png")
            dpi=300,

            bbox_inches="tight"

        )

        plt.close()



    # ==========================
    # 3 Waterfall
    # ==========================

    def waterfall(
            self,
            sample_id=0,
            name = "test"
    ):

        plt.figure()

        shap.plots.waterfall(

            self.shap_explanation[sample_id],

            show=False

        )


        plt.title(
            f"Waterfall Sample {sample_id}"
        )


        plt.savefig(
            self.output_dir/f"waterfall_{name}.png",
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()



    # ==========================
    # 4 Decision Plot
    # ==========================

    def decision_plot(self, samples=100):

        shap_values = np.asarray(
            self.shap_values[:samples]
        )

        X_test = (
            self.X_test
            .iloc[:samples]
            .values
        )

        shap.decision_plot(

            self.expected_value,

            shap_values,

            X_test,

            feature_names=list(
                self.feature_names
            ),

            show=False

        )

        plt.title(
            "SHAP Decision Plot"
        )

        plt.savefig(

            self.output_dir/"decision_plot.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()



    # ==========================
    # 5 Dependence Plot
    # ==========================

    def dependence_plot(self, feature):

        shap.dependence_plot(

            feature,

            self.shap_values,

            self.X_test.values,

            feature_names=list(
                self.feature_names
            ),

            show=False

        )

        plt.savefig(
            self.output_dir/f"dependence_{feature}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()



    # ==========================
    # 6 Interaction
    # ==========================

    def interaction_plot(
            self,
            feature1,
            feature2
    ):


        shap.dependence_plot(

            feature1,

            self.get_values(),

            self.X_test,

            interaction_index=feature2,

            show=False

        )


        plt.title(
            f"Interaction {feature1}-{feature2}"
        )


        plt.savefig(

            f"{self.output_dir}/interaction_{feature1}_{feature2}.png",

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



    # ==========================
    # 7 HTML Report
    # ==========================

    def html_report(self, sample_id=0):

        html_path = self.output_dir/"shap_report.html"


        force_plot = shap.force_plot(

            self.expected_value,

            self.shap_values[sample_id],

            self.X_test.iloc[sample_id],

            feature_names=self.feature_names,

            matplotlib=False

        )

        shap.save_html(

            str(html_path),

            force_plot

        )

        print(
            "HTML report saved:",
            html_path
        )

    # def select_representative_samples(self):
    #
    #     proba = self.model.predict_proba(self.X_test)[:, 1]
    #
    #     high = np.argmax(proba)
    #
    #     low = np.argmin(proba)
    #
    #     boundary = np.argmin(np.abs(proba - 0.5))
    #
    #     return [low, boundary, high]
    def select_representative_samples(self):

        proba = self.model.predict_proba(
            self.X_test
        )[:, 1]

        samples = {}

        # Highest Risk
        samples["Highest_Risk"] = np.argmax(proba)

        # Lowest Risk
        samples["Lowest_Risk"] = np.argmin(proba)

        # Boundary
        samples["Boundary"] = np.argmin(
            np.abs(proba - 0.5)
        )

        # Misclassified
        y_pred = self.model.predict(
            self.X_test
        )

        wrong = np.where(
            y_pred != self.y_test
        )[0]

        if len(wrong) > 0:
            samples["Misclassified"] = wrong[0]

        return samples

    # ==========================
    # Run all
    # ==========================

    def run_all(self, html=True, interaction=True, waterfall_samples=(0,5,10)):

        self.build_explainer()

        self.global_importance()

        self.beeswarm()

        # self.waterfall(
        #     sample_id=0
        # )
        # for sample_id in waterfall_samples:
        #
        #     if 0 <= sample_id < len(self.shap_explanation):
        #
        #         self.waterfall(sample_id)
        #
        #     else:
        #
        #         print(f"Sample {sample_id} does not exist, skipped.")
        # samples = self.select_representative_samples()
        #
        # for s in samples:
        #     self.waterfall(s)

        samples = self.select_representative_samples()

        for name, sample_id in samples.items():
            self.waterfall(

                sample_id,

                name

            )

        self.decision_plot()

        # 自动获取Top feature

        importance = np.abs(
            self.shap_values
        ).mean(axis=0)

        top_indices = np.argsort(
            importance
        )[-2:]

        top_features = [
            self.feature_names[i]
            for i in top_indices
        ]

        self.dependence_plot(
            top_features[0]
        )

        # self.interaction_plot(
        #     top_features[0],
        #     top_features[1]
        # )

        self.html_report()