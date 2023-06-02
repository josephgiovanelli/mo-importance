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

from ConfigSpace import Configuration

from smac.facade.abstract_facade import AbstractFacade

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from inner_loop.pareto_mlp import ParetoMLP
from my_model.my_grid_model import MyGridModel
from utils.input import ConfDict

from utils.sample import grid_search

from inner_loop.mlp import MLP


class PreferenceParetoMLP(ParetoMLP):
    def __init__(self, implementation="sklearn"):
        super().__init__(implementation)

    def train(
        self,
        random_config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            pareto = [super().train(random_config, seed, budget)]

            ConfDict({"paretos": ConfDict()["paretos"] + pareto})
