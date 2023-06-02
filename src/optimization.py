# %%
from __future__ import annotations
import os
import time
import random

import numpy as np

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from inner_loop.pareto_mlp import ParetoMLP
from inner_loop.preference_pareto_mlp import PreferenceParetoMLP

from utils.argparse import parse_args
from utils.dataset import load_dataset_from_openml
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
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
        file_name=args.conf_file, origin=os.path.basename(__file__).split(".")[0]
    )

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    # start_time = time.time()

    if check_dump():
        paretos = load_dump()
    else:
        mlp = PreferenceParetoMLP("lcbench")
        # grid_samples = grid_search(configspace=mlp.configspace, num_steps=2)
        random_samples = random_search(
            configspace=mlp.configspace, num_samples=ConfDict()["random_samples"]
        )

        ConfDict({"paretos": []})
        for idx, sample in enumerate(random_samples):
            # print(f"{idx}th conf of random sampling")
            mlp.train(sample)

        adapt_paretos(ConfDict()["paretos"])
        save_paretos(ConfDict()["paretos"], "dump")

    # print(f"Optimization time: {time.time() - start_time}")

    update_config(ConfDict()["paretos"])

    if check_pictures():
        save_paretos(encode_pareto(ConfDict()["paretos"]), "encoded")
    else:
        for idx, history in enumerate(ConfDict()["paretos"]):
            plot_pareto_from_history(
                history,
                os.path.join(ConfDict()["output_folder"], str(idx)),
            )

    # # Define our environment variables
    # scenario = Scenario(
    #     mlp.configspace,
    #     output_directory=args.output_path,
    #     objectives=args.metrics,
    #     walltime_limit=args.time_budget,
    #     n_trials=args.iterations,
    #     seed=args.seed,
    #     n_workers=1,
    # )

    # # We want to run five random configurations before starting the optimization.
    # initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    # intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

    # # Create our SMAC object and pass the scenario and the train method
    # smac = HPOFacade(
    #     scenario,
    #     mlp.train,
    #     initial_design=initial_design,
    #     model=RandomModel(mlp.configspace),
    #     intensifier=intensifier,
    #     overwrite=True,
    # )

    # # Let's optimize
    # incumbents = smac.optimize()

    # # Get cost of default configuration
    # default_cost = smac.validate(mlp.configspace.get_default_configuration())
    # print(f"Validated costs from default config: \n--- {default_cost}\n")

    # print("Validated costs from the Pareto front (incumbents):")
    # for incumbent in incumbents:
    #     cost = smac.validate(incumbent)
    #     print("---", cost)

    # # Let's plot a pareto front
    # plot_pareto(smac, incumbents)

# %%
