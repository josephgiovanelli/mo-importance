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
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)


logger = logging.getLogger()
logger.disabled = True

import matplotlib.pyplot as plt


def plot_mean_std(df, output_path):
    indicators = df.index
    x_ticks = df.columns
    x_labels = "No. samples"

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

    for idx, indicator in enumerate(indicators):
        values = df.loc[indicator].values
        means = [value[0] for value in values]
        stds = [value[1] for value in values]

        ax = axes[int(idx / ncols), idx % ncols]

        ax.errorbar(x_ticks, means, yerr=stds, fmt="o", capsize=3)
        ax.set_title(indicator.upper())
        ax.set_ylim(0.5, 1)
        ax.set_yticks([i * 0.1 for i in range(5, 11)])
        ax.grid(axis="y")
        if idx > 1:
            ax.set_xticks(x_ticks)
            ax.set_xlabel(x_labels)
        if idx % 2 == 0:
            ax.set_ylabel("Tau")

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "preference_evaluation.png"))
    fig.savefig(os.path.join(output_path, "preference_evaluation.pdf"))


if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "output", "preference"))
    output_path = make_dir(os.path.join("/", "home", "plots"))
    indicators = ["hv", "sp", "ms", "r2"]

    def custom_agg(rows):
        converted_rows = np.array([np.fromstring(row[1:-1], sep=",") for row in rows])
        means = np.mean(converted_rows, axis=0)
        return [round(np.mean(means), 2), round(np.std(means), 2)]

    cross_validation_columns = [f"cross_validation_{i}" for i in range(1, 5)]
    results = (
        pd.concat(
            [
                pd.read_csv(os.path.join(input_path, f"{indicator}.csv"))
                .iloc[-3:, :]
                .assign(indicator=indicator)
                for indicator in indicators
            ],
            axis=0,
        )[cross_validation_columns + ["indicator"]]
        .groupby("indicator")
        .agg(custom_agg)
        .rename(
            columns={
                column: 28 * int(column.split("_")[-1])
                for column in cross_validation_columns
            }
        )
    )

    plot_mean_std(results, output_path)
