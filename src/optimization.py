# %%
from __future__ import annotations
import os
import time
import random

import numpy as np

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from utils.argparse import parse_args
from utils.common import make_dir
from utils.dataset import load_dataset_from_openml
from utils.optimization import multi_objective, single_objective
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
    plot_pareto_from_smac,
    get_pareto_indicators,
)
from utils.sample import grid_search, random_search
from utils.input import ConfDict, create_configuration
from utils.output import (
    adapt_paretos,
    check_pictures,
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(
        file_name=args.conf_file,
        origin="optimization",
    )

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])
    for main_indicator in get_pareto_indicators().keys():
        for mode in ["indicators", "preferences"]:
            single_objective(main_indicator=main_indicator, mode=mode)
    for mode in ["fair", "unfair"]:
        multi_objective(mode=mode)


# %%
