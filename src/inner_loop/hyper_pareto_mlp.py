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

from pymoo.indicators.hv import HV

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from inner_loop.pareto_mlp import ParetoMLP
from my_model.my_grid_model import MyGridModel
from utils.input import ConfDict
from utils.output import adapt_paretos
from utils.pareto import encode_pareto

from utils.sample import grid_search
from utils.output import adapt_to_mode

from inner_loop.mlp import MLP


class HyperParetoMLP(ParetoMLP):
    def __init__(self, implementation="sklearn"):
        super().__init__(implementation)
        ref_point = [0, 0]
        for obj_idx in range(len(ConfDict()["objectives"])):
            ref_point[obj_idx] = adapt_to_mode(
                ConfDict()["objectives"][obj_idx]["upper_bound"]
                if ConfDict()["obj_modes"][obj_idx] == "min"
                else ConfDict()["objectives"][obj_idx]["lower_bound"],
                ConfDict()["obj_modes"][obj_idx],
            )
        self.ind_ = HV(ref_point=ref_point)

    def train(
        self,
        random_config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            pareto = [super().train(random_config, seed, budget)]

            encoded_pareto = encode_pareto(pareto)
            score = self.ind_(np.array(encoded_pareto[0]))

            adapt_paretos(pareto)

            ConfDict({"paretos": ConfDict()["paretos"] + pareto})
            ConfDict({"scores": ConfDict()["scores"] + [score]})

            return score * -1
