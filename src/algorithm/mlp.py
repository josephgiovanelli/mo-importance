from __future__ import annotations

import time
import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier


class MLP:
    def __init__(self, X, y, metrics, modes):
        self.X = X
        self.y = y
        self.metrics = metrics
        self.modes = modes

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical(
            "activation", ["logistic", "tanh", "relu"], default="tanh"
        )
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate = Categorical(
            "learning_rate", ["constant", "invscaling", "adaptive"], default="constant"
        )
        learning_rate_init = Float(
            "learning_rate_init", (0.0001, 1.0), default=0.001, log=True
        )

        cs.add_hyperparameters(
            [
                n_layer,
                n_neurons,
                activation,
                solver,
                batch_size,
                learning_rate,
                learning_rate_init,
            ]
        )

        use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        use_lr_init = InCondition(
            child=learning_rate_init, parent=solver, values=["sgd", "adam"]
        )
        use_batch_size = InCondition(
            child=batch_size, parent=solver, values=["sgd", "adam"]
        )

        # We can also add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_lr, use_batch_size, use_lr_init])

        return cs

    def train(
        self,
        config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ) -> dict[str, float]:
        lr = config["learning_rate"] if config["learning_rate"] else "constant"
        lr_init = (
            config["learning_rate_init"] if config["learning_rate_init"] else 0.001
        )
        batch_size = config["batch_size"] if config["batch_size"] else 200

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = MLPClassifier(
                hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                solver=config["solver"],
                batch_size=batch_size,
                activation=config["activation"],
                learning_rate=lr,
                learning_rate_init=lr_init,
                max_iter=int(np.ceil(budget)),
                random_state=seed,
            )

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(
                n_splits=5, random_state=seed, shuffle=True
            )  # to make CV splits consistent

            scores = cross_validate(
                classifier,
                self.X.copy(),
                self.y.copy(),
                scoring=self.metrics,
                cv=cv,
                return_estimator=False,
                return_train_score=False,
                verbose=0,
                error_score="raise",
            )

        return {
            f"{metric}": np.mean(scores["test_" + metric])
            * (-1 if self.modes[idx] == "max" else 1)
            for idx, metric in enumerate(self.metrics)
        }
