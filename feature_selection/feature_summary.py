"""
======================================================
Feature Selection Summary

Generate:

1. feature_frequency.csv
2. feature_statistics.csv
3. repository_summary.csv
4. overall_summary.csv


FOR RQ 4
======================================================
"""

from pathlib import Path

from collections import Counter

import pandas as pd


class FeatureSummary:

    def __init__(self, root):

        self.root = Path(root)

    ##############################################################
    # One Model
    ##############################################################

    def summarize_model(self, model_dir):

        model_dir = Path(model_dir)

        csv_files = sorted(

            model_dir.glob("fold_*.csv")

        )

        counter = Counter()

        fold_num = len(csv_files)

        for csv in csv_files:

            df = pd.read_csv(csv)

            counter.update(

                df["Feature"]

            )

        ##########################################################

        frequency = pd.DataFrame(

            {

                "Feature": list(counter.keys()),

                "Frequency": list(counter.values())

            }

        )

        frequency = frequency.sort_values(

            "Frequency",

            ascending=False

        )

        frequency.to_csv(

            model_dir /

            "feature_frequency.csv",

            index=False

        )

        ##########################################################

        statistics = frequency.copy()

        statistics["Selected_Rate"] = (

            statistics["Frequency"]

            / fold_num

            * 100

        ).round(2)

        statistics["Rank"] = range(

            1,

            len(statistics) + 1

        )

        statistics.to_csv(

            model_dir /

            "feature_statistics.csv",

            index=False

        )

        return statistics

    ##############################################################
    # Repository
    ##############################################################

    def summarize_repository(self):

        repository_result = []

        for repo in sorted(

                self.root.iterdir()

        ):

            if not repo.is_dir():

                continue

            for dataset in sorted(

                    repo.iterdir()

            ):

                if not dataset.is_dir():

                    continue

                for model in sorted(

                        dataset.iterdir()

                ):

                    if not model.is_dir():

                        continue

                    stat = self.summarize_model(

                        model

                    )

                    stat["Repository"] = repo.name

                    stat["Dataset"] = dataset.name

                    stat["Model"] = model.name

                    repository_result.append(

                        stat

                    )

        repository_result = pd.concat(

            repository_result,

            ignore_index=True

        )

        repository_result.to_csv(

            self.root /

            "repository_summary.csv",

            index=False

        )

        return repository_result

    ##############################################################
    # Overall
    ##############################################################

    def overall_summary(self):

        df = self.summarize_repository()

        overall = (

            df.groupby(

                "Feature"

            )

            ["Frequency"]

            .sum()

            .reset_index()

        )

        overall = overall.sort_values(

            "Frequency",

            ascending=False

        )

        overall["Rank"] = range(

            1,

            len(overall) + 1

        )

        overall.to_csv(

            self.root /

            "overall_summary.csv",

            index=False

        )

        print()

        print("=" * 70)

        print("Feature Summary Finished")

        print("=" * 70)
