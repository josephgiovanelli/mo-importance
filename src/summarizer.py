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
from utils.comparison import get_cell_value, get_element_from_results
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
    load_json_file,
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)


logger = logging.getLogger()
logger.disabled = True


if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "interactive-mo-ml", "input"))
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

    summary_path = make_dir(
        os.path.join("/", "home", "interactive-mo-ml", "output", "summary")
    )
    results.to_csv(os.path.join(summary_path, "results_raw.csv"), index=False)

    results = results[
        [
            "main_indicator",
            "second_indicator",
            "preference_budget",
            "mode",
            "preferences mean",
            "indicators mean",
        ]
    ]
    results = (
        results.groupby(
            ["main_indicator", "second_indicator", "preference_budget", "mode"]
        )
        .agg(["mean", "std"])
        .reset_index()
    )
    results.columns = [" ".join(col).strip() for col in results.columns.values]
    results.columns = [
        col if " " not in col else f"""{col.split(" ")[0]} {col.split(" ")[2]}"""
        for col in results.columns.values
    ]
    results.to_csv(os.path.join(summary_path, "results.csv"), index=False)

    per_budget_results = {
        preference_budget: pd.concat(
            [
                pd.DataFrame({"indicators\preferences": indicators}),
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                column: [
                                    f"{get_cell_value(results, preference_budget, column, row)}"
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

    mo_results = {}
    for mo_mode in ["fair", "unfair"]:
        mo_results[mo_mode] = pd.concat(
            [
                pd.DataFrame(
                    load_json_file(
                        os.path.join(
                            ConfDict()[conf]["output_folder"],
                            f"multi_objective_{mo_mode}",
                            "140",
                            "scores.json",
                        )
                    )["0"]
                ).rename(
                    mapper={"indicators": conf.split(".")[0].split("_")[1]},
                    axis="columns",
                )[
                    conf.split(".")[0].split("_")[1]
                ]
                for conf in datasets
            ],
            axis=1,
        )
        mo_results[mo_mode].to_csv(
            os.path.join(summary_path, f"results_mo_{mo_mode}_raw.csv")
        )
        mo_results[mo_mode][f"{mo_mode}_mean"] = mo_results[mo_mode].mean(axis=1)
        mo_results[mo_mode][f"{mo_mode}_std"] = mo_results[mo_mode].std(axis=1)
        mo_results[mo_mode] = mo_results[mo_mode][[f"{mo_mode}_mean", f"{mo_mode}_std"]]
        # mo_results[mo_mode] = mo_results[mo_mode].groupby().agg(["mean", "std"])
    pd.concat(mo_results.values(), axis=1).to_csv(
        os.path.join(summary_path, f"results_mo_agg.csv")
    )

# %%
