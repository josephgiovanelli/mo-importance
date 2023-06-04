from __future__ import annotations
import os

import warnings

import numpy as np

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

from ConfigSpace import Configuration

from smac.facade.abstract_facade import AbstractFacade

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from inner_loop.pareto_mlp import ParetoMLP
from my_model.my_grid_model import MyGridModel
from ranker.my_rank_svc import MyRankSVM
from utils.input import ConfDict
from utils.output import adapt_paretos, check_preferences, load_json_file, update_config
from utils.pareto import encode_pareto
from utils.preference_learning import create_preference_dataset

from utils.sample import grid_search

from inner_loop.mlp import MLP


class PreferenceParetoMLP(ParetoMLP):
    def __init__(self, implementation="sklearn"):
        super().__init__(implementation)

        preference_path = os.path.join(
            os.path.dirname(ConfDict()["output_folder"]), "preliminar_sampling"
        )
        if check_preferences(os.path.join(preference_path, "preferences.csv")):
            X, y, preferences = create_preference_dataset(
                preference_path=preference_path
            )
            config_dict = load_json_file(
                os.path.join(preference_path, "preference_incumbent.json")
            )
            self.preference_model_ = MyRankSVM(
                **config_dict, random_state=ConfDict()["seed"]
            )
            self.preference_model_.fit(X.copy(), y.copy())
        else:
            raise Exception("No preferences found")

    def train(
        self,
        random_config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            pareto = [super().train(random_config, seed, budget)]

            adapt_paretos(pareto)
            flatten_encoded = np.array(encode_pareto(pareto)).flatten()

            score = self.preference_model_.predict_scores(
                np.array([[flatten_encoded]])
            )[0]

            ConfDict({"paretos": ConfDict()["paretos"] + pareto})
            ConfDict({"scores": ConfDict()["scores"] + list(score)})

            return score[0] * -1
