from __future__ import annotations

import warnings

import numpy as np

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio,
    equalized_odds_ratio,
)

from codecarbon import EmissionsTracker

from utils.input import ConfDict


class MLP:
    @property
    def configspace(self) -> ConfigurationSpace:
        return ConfigurationSpace(
            {
                "n_layer": Integer("n_layer", (1, 5), default=1),
                "n_neurons": Integer("n_neurons", (8, 256), log=True, default=10),
                "activation": Categorical(
                    "activation", ["logistic", "tanh", "relu"], default="tanh"
                ),
                "solver": Categorical("solver", ["sgd", "adam"], default="adam"),
                "learning_rate_init": Float(
                    "learning_rate_init", (0.0001, 1.0), default=0.001, log=True
                ),
                "alpha": Float("alpha", (0.000001, 10.0), default=0.0001, log=True),
            }
        )

    def train(
        self,
        config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ) -> dict[str, float]:
        def __adapt_to_mode(value, mode):
            return value if mode == "min" else 1 - value

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # try:
            classifier = MLPClassifier(
                hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                solver=config["solver"],
                activation=config["activation"],
                learning_rate_init=config["learning_rate_init"],
                alpha=config["alpha"],
                max_iter=int(np.ceil(budget)),
                random_state=ConfDict()["seed"],
            )

            X_train, X_test, y_train, y_test = train_test_split(
                ConfDict()["X"], ConfDict()["y"], test_size=0.33, random_state=seed
            )

            if ConfDict()["use_case"] == "green automl":
                tracker = EmissionsTracker()
                tracker.start()

            classifier = classifier.fit(X_train, y_train)

            if ConfDict()["use_case"] == "green automl":
                use_case_dict = {
                    f"""{ConfDict()["use_case_objective"]["metric"]}""": __adapt_to_mode(
                        tracker.stop(), ConfDict()["performance_objective"]["mode"]
                    )
                }

            y_pred = classifier.predict(X_test)

            performance_dict = {
                f"""{ConfDict()["performance_objective"]["metric"]}""": globals()[
                    f"""{ConfDict()["performance_objective"]["metric"]}"""
                ](y_test, y_pred)
            }

            if ConfDict()["use_case"] == "fairness":
                use_case_dict = {
                    f"""{ConfDict()["use_case_objective"]["metric"]}""": globals()[
                        f"""{ConfDict()["use_case_objective"]["metric"]}"""
                    ](
                        y_test,
                        y_pred,
                        sensitive_features=X_test[
                            :, ConfDict()["use_case_objective"]["sensitive_feature_idx"]
                        ],
                    )
                }

            return {**performance_dict, **use_case_dict}

            # except:
            #     print("Something went wrong!")
