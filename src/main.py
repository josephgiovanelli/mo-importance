# %%
from __future__ import annotations

import numpy as np

from ConfigSpace import Configuration

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from algorithm.mlp import MLP

from utils.argparse import parse_args
from utils.dataset import load_dataset_from_openml
from utils.plot import plot_pareto


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


if __name__ == "__main__":
    args, _ = parse_args()
    np.random.seed(args.seed)

    X, y, _ = load_dataset_from_openml(args.dataset)
    mlp = MLP(
        X=X,
        y=y,
        metrics=args.metrics,
        modes=args.modes,
        application="fairness",
        setting="pareto",
    )
    # grid_samples = mlp.grid_search(num_steps=2)
    random_samples = mlp.random_search(num_samples=25)

    paretos = []
    for sample in random_samples:
        paretos += [mlp.objective(sample, args.seed)]

    print(paretos)

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
