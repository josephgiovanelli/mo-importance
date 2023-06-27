import json
import random
import warnings
import logging

import os

import numpy as np
import pandas as pd

from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.model_selection import KFold

from utils.argparse import parse_args
from utils.common import get_tuning_datasets, make_dir
from utils.input import ConfDict, create_configuration
from utils.output import adapt_encoded, check_preferences, load_encoded, load_json_file
from utils.pareto import get_pareto_indicators

from utils.preference_learning import (
    configspace,
    create_preference_dataset,
    objective,
    evaluate_model,
)

from ranker.my_rank_svc import MyRankSVM

if __name__ == "__main__":
    datasets = get_tuning_datasets()
    create_configuration(datasets)
    ConfDict({"datasets": datasets, "indicators": {}})

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    incumbents = {}
    indicators = ["hv", "sp", "ms", "r2"]
    ConfDict()["summary"] = pd.DataFrame()
    for indicator in indicators:
        for dataset in datasets:
            if check_preferences(
                os.path.join(ConfDict()[dataset]["output_folder"], "preferences.csv")
            ):
                X, y, preferences = create_preference_dataset(
                    preference_path=os.path.join(ConfDict()[dataset]["output_folder"]),
                    indicator=indicator,
                )
                ConfDict()[dataset]["X"] = X
                ConfDict()[dataset]["Y"] = y
                ConfDict()[dataset]["flatten_encoded"] = {
                    key: np.array(value).flatten()
                    for key, value in load_encoded(
                        os.path.join(ConfDict()[dataset]["output_folder"])
                    ).items()
                }
                ConfDict()[dataset]["scores"] = load_json_file(
                    os.path.join(ConfDict()[dataset]["output_folder"], "scores.json")
                )
                ConfDict()[dataset]["test_folds"] = np.array(
                    [
                        fold
                        for _, fold in KFold(
                            n_splits=5, random_state=ConfDict()["seed"]
                        ).split(
                            range(
                                int(
                                    len(ConfDict()[dataset]["scores"].keys())
                                    / len(indicators)
                                )
                            )
                            # np.unique([int(elem.split("_")[1]) for elem in ConfDict()[dataset]["scores"].keys()])
                        )
                    ]
                )
                ConfDict()[dataset]["config_dicts"] = load_json_file(
                    os.path.join("/", "home", "output", "preference", "incumbent.json")
                )
                ConfDict()[dataset]["preferences"] = preferences[
                    ["pair_0", "pair_1"]
                ].to_numpy()
                ConfDict()["current_indicator"] = indicator

            else:
                raise Exception(f"No preference file found for {dataset}")

        config_dict = ConfDict()[dataset]["config_dicts"][
            ConfDict()["current_indicator"]
        ]
        result_dict = {
            "cross_validation_1": [],
            "cross_validation_2": [],
            "cross_validation_3": [],
            "cross_validation_4": [],
        }
        for seed in [0, 1, 42]:
            for dataset in ConfDict()["datasets"]:
                for mode in result_dict.keys():
                    evaluate_model(config_dict, result_dict, dataset, mode, seed)

        ConfDict()["summary"] = pd.concat(
            [
                ConfDict()["summary"],
                pd.DataFrame(
                    {
                        **{"indicator": [indicator]},
                        **{key: [value] for key, value in result_dict.items()},
                        **{
                            f"{key}_mean": [np.mean(value)]
                            for key, value in result_dict.items()
                        },
                        **{
                            f"{key}_std": [np.std(value)]
                            for key, value in result_dict.items()
                        },
                    }
                ),
            ],
            ignore_index=True,
        )

    ConfDict()["summary"].to_csv(
        os.path.join(
            make_dir(
                os.path.join("/", "home", "output", "preference"),
            ),
            f"model_evaluation.csv",
        ),
        index=False,
    )
# %%
