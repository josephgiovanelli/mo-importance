# %%
from __future__ import annotations
from itertools import combinations
import logging
import os
import time
import random

import numpy as np
import pandas as pd

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from utils.argparse import parse_args
from utils.common import get_tuning_datasets, make_dir
from utils.dataset import load_dataset_from_openml
from utils.optimization import (
    multi_objective,
    single_objective,
    restore_results,
    process_results,
)
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
    plot_pareto_from_smac,
    get_pareto_indicators,
)
from utils.sample import grid_search, random_search
from utils.input import ConfDict, create_configuration
from utils.preference_learning import get_preference_budgets
from utils.output import (
    adapt_paretos,
    check_pictures,
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)


logger = logging.getLogger()
logger.disabled = True


if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "input"))
    confs = [p for p in os.listdir(input_path) if ".json" in p]
    datasets = [elem for elem in confs if elem not in get_tuning_datasets()]
    create_configuration(file_name=datasets, origin="optimization")

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    preference_budgets = get_preference_budgets()
    indicators = ["hv", "sp", "ms", "r2"]

    results = pd.concat(
        [
            pd.read_csv(
                os.path.join(ConfDict()[conf]["output_folder"], "results.csv")
            ).assign(dataset=conf.split(".")[0].split("_")[1])
            for conf in datasets
        ],
        axis=0,
    )

    summary_path = make_dir(os.path.join("/", "home", "output", "summary"))
    results.to_csv(os.path.join(summary_path, "results.csv"), index=False)

    def get_element_from_results(preference_budget, column, row, mode):
        return round(
            results.loc[
                (results["second_indicator"] == column)
                & (results["preference_budget"] == preference_budget)
                & (
                    results["main_indicator"]
                    == (column if mode == "preferences" else row)
                )
                & (results["mode"] == mode),
                "preferences",
            ].mean(),
            2,
        )

    per_budget_results = {
        preference_budget: pd.concat(
            [
                pd.DataFrame({"indicators\preferences": indicators}),
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                column: [
                                    f"""{get_element_from_results(preference_budget, column, row, "indicators")}\{get_element_from_results(preference_budget, column, row, "preferences")}"""
                                    for row in indicators
                                ]
                            }
                        )
                        for column in indicators
                    ],
                    axis=1,
                ),
            ],
            axis=1,
        )
        for preference_budget in preference_budgets
    }
    for k, v in per_budget_results.items():
        v.to_csv(
            os.path.join(
                summary_path,
                f"budget_{k}.csv",
            ),
            index=False,
        )
# %%
