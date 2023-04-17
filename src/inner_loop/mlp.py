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

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

# from codecarbon import EmissionsTracker

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            try:
                # tracker = EmissionsTracker()
                # tracker.start()

                classifier = MLPClassifier(
                    hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                    solver=config["solver"],
                    activation=config["activation"],
                    learning_rate_init=config["learning_rate_init"],
                    alpha=config["alpha"],
                    max_iter=int(np.ceil(budget)),
                    random_state=ConfDict()["seed"],
                )

                # Returns the 5-fold cross validation accuracy
                cv = StratifiedKFold(
                    n_splits=5, random_state=seed, shuffle=True
                )  # to make CV splits consistent

                scores = cross_validate(
                    classifier,
                    ConfDict()["X"].copy(),
                    ConfDict()["y"].copy(),
                    scoring=ConfDict()["obj_metrics"],
                    cv=cv,
                    return_estimator=False,
                    return_train_score=False,
                    verbose=0,
                    error_score="raise",
                )

                # tracker.stop()

                return {
                    f"{metric}": np.mean(scores["test_" + metric])
                    * (-1 if ConfDict()["obj_modes"][idx] == "max" else 1)
                    for idx, metric in enumerate(ConfDict()["obj_metrics"])
                }
            except:
                print("Something went wrong!")
