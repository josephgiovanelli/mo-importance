from __future__ import annotations

import warnings

import numpy as np
import ConfigSpace
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from utils.sample import grid_search


class MLP:
    def __init__(self, X, y, metrics, modes, application, setting, grid_samples):
        self.X = X
        self.y = y
        self.metrics = metrics
        self.modes = modes
        self.application = application
        self.setting = setting
        self.p_star = None
        self.grid_samples = grid_samples

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical(
            "activation", ["logistic", "tanh", "relu"], default="tanh"
        )
        solver = Categorical("solver", ["sgd", "adam"], default="adam")
        learning_rate_init = Float(
            "learning_rate_init", (0.0001, 1.0), default=0.001, log=True
        )
        alpha = Float("alpha", (0.000001, 10.0), default=0.0001, log=True)

        hps = [n_layer, n_neurons, activation, solver, learning_rate_init, alpha]

        if self.setting != "mo":
            self.p_star = alpha if self.application == "fairness" else n_layer
            hps = [hp for hp in hps if hp != self.p_star]

        cs.add_hyperparameters(hps)

        return cs

    def __train(
        self,
        config: dict,
        seed: int = 0,
        budget: int = 10,
    ) -> dict[str, float]:
        classifier = MLPClassifier(
            hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
            solver=config["solver"],
            activation=config["activation"],
            learning_rate_init=config["learning_rate_init"],
            alpha=config["alpha"],
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

    def objective(
        self,
        config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            my_config = config.get_dictionary()
            if self.setting != "mo":
                scores = []
                cs = ConfigurationSpace()
                cs.add_hyperparameters([self.p_star])
                for p in grid_search(cs, self.grid_samples):
                    my_config[self.p_star.name] = p[self.p_star.name]
                    scores += [self.__train(my_config, seed, budget)]
            else:
                scores = self.__train(config, seed, budget)

        return scores
