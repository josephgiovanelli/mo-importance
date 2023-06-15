from __future__ import annotations
import os
import time
import random

import numpy as np

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel
from inner_loop.mlp import MLP

from inner_loop.utility_pareto_mlp import UtilityParetoMLP

from utils.argparse import parse_args
from utils.common import make_dir
from utils.dataset import load_dataset_from_openml
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
    plot_pareto_from_smac,
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


def multi_objective():
    mlp = MLP("lcbench")

    # Define our environment variables
    scenario = Scenario(
        mlp.configspace,
        objectives=ConfDict()["obj_metrics"],
        n_trials=ConfDict()["optimization_samples"] * ConfDict()["grid_samples"],
        seed=ConfDict()["seed"],
        n_workers=1,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    # Let's optimize
    incumbents = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(mlp.configspace.get_default_configuration())
    print(f"Validated costs from default config: \n--- {default_cost}\n")

    print("Validated costs from the Pareto front (incumbents):")
    for incumbent in incumbents:
        print("---", incumbent)
        cost = smac.validate(incumbent)
        print("---", cost)

    plot_pareto_from_smac(
        smac,
        incumbents,
        os.path.join(
            make_dir(os.path.join(ConfDict()["output_folder"], "multi_objective")),
            "best",
        ),
        title=f"""Sample Multi-objective  w/ {ConfDict()["optimization_samples"]* ConfDict()["grid_samples"]} samples""",
    )


def single_objective(main_indicator="hv", mode="preferences"):
    new_output_path = make_dir(
        os.path.join(ConfDict()["output_folder"], mode, main_indicator)
    )
    if check_dump(file_name=os.path.join(new_output_path, "dump.json")):
        ConfDict(
            {"paretos": load_dump(file_name=os.path.join(new_output_path, "dump.json"))}
        )
    else:
        mlp = UtilityParetoMLP(
            implementation="lcbench", main_indicator=main_indicator, mode=mode
        )

        ConfDict({"paretos": []})
        ConfDict({"scores": []})

        # Define our environment variables
        scenario = Scenario(
            mlp.configspace,
            n_trials=ConfDict()["optimization_samples"],
            seed=ConfDict()["seed"],
            n_workers=1,
        )

        # We want to run five random configurations before starting the optimization.
        initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
        intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

        # Create our SMAC object and pass the scenario and the train method
        smac = HPOFacade(
            scenario,
            mlp.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

        # Let's optimize
        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(mlp.configspace.get_default_configuration())
        print(f"Validated costs from default config: \n--- {default_cost}\n")

        print("Validated costs from the Pareto front (incumbents):")
        cost = smac.validate(incumbent)
        print("---", cost)

        save_paretos(
            ConfDict()["paretos"],
            new_output_path,
            "dump",
        )
        save_paretos(
            np.array(ConfDict()["scores"]).flatten(),
            new_output_path,
            "scores",
        )

    # print(f"Optimization time: {time.time() - start_time}")

    update_config(ConfDict()["paretos"])

    if not check_pictures():
        for idx, history in enumerate(ConfDict()["paretos"]):
            plot_pareto_from_history(
                history,
                os.path.join(
                    new_output_path,
                    str(idx),
                ),
                title=f"""{mode.capitalize()} w/ {main_indicator.capitalize()} ({ConfDict()["optimization_samples"]} samples)""",
            )

        save_paretos(
            encode_pareto(ConfDict()["paretos"]),
            new_output_path,
            "encoded",
        )
