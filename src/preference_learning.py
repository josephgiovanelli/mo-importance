import random
import warnings
import os

import csrank as cs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from itertools import combinations
from IPython.display import clear_output
from csrank.dataset_reader import ChoiceDatasetGenerator, ObjectRankingDatasetGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from ranker.my_expected_rank_regression import MyExpectedRankRegression
from ranker.my_rank_svc import MyRankSVM


from utils.argparse import parse_args
from utils.input import ConfDict, create_configuration
from utils.output import (
    load_encoded,
    check_preferences,
    load_preferences,
    save_preferences,
)

if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(args.conf_file)

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    if check_preferences():
        flatten_encoded = {
            key: np.array(value).flatten() for key, value in load_encoded().items()
        }
        preferences = load_preferences()
        X = np.array(
            [
                np.array([flatten_encoded[str(pair[0])], flatten_encoded[str(pair[1])]])
                for pair in preferences[["pair_0", "pair_1"]].to_numpy()
            ]
        )
        y = np.array(
            [
                # np.array([0, 1] if preference == 0 else [1, 0])
                preference
                for preference in preferences["preference"].to_numpy()
            ]
        )

        (
            preferences_train,
            preferences_test,
            X_train,
            X_test,
            Y_train,
            Y_test,
        ) = train_test_split(
            preferences[["pair_0", "pair_1"]].to_numpy(),
            X,
            y,
            test_size=0.33,
            random_state=42,
        )

        # fate = MyExpectedRankRegression()
        # fate = cs.RankNetObjectRanker()
        # print(X_train)
        # print(Y_train)
        # print(X_test)
        # print(Y_test)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for n_features in range(1, 20):
                print()
                print(f"{n_features} features")

                fate = MyRankSVM(n_features=n_features)

                scores = cross_validate(
                    fate,
                    X,
                    y,
                    scoring=["accuracy"],
                    cv=10,
                    return_estimator=False,
                    return_train_score=False,
                    verbose=0,
                )

                print(f"""cross_validation: {np.mean(scores["test_accuracy"])}""")

                fate = MyRankSVM(n_features=n_features)
                fate.fit(X_train, Y_train)

                result = pd.DataFrame(
                    {
                        "pair_0": [pref[0] for pref in preferences_test],
                        "pair_0_score": [
                            elem
                            for list in fate.predict_scores(
                                np.array([[elem[0]] for elem in X_test])
                            )
                            for elem in list
                        ],
                        "pair_1": [pref[1] for pref in preferences_test],
                        "pair_1_score": [
                            elem
                            for list in fate.predict_scores(
                                np.array([[elem[1]] for elem in X_test])
                            )
                            for elem in list
                        ],
                        "y_true": Y_test,
                        "y_pred": fate.predict(X_test),
                    }
                )

                result.to_csv(f"predictions_pca_{n_features}.csv", index=False)

                print(
                    f"""train_test split: {accuracy_score(result["y_true"], result["y_pred"])}"""
                )

# %%
