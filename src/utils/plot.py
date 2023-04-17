from __future__ import annotations
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ConfigSpace import Configuration

from smac.facade.abstract_facade import AbstractFacade

from utils.input import ConfDict


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def get_pareto_from_history(history: list[tuple[Configuration, dict]]):
    def _get_pareto_indeces(costs):
        is_efficient = np.arange(costs.shape[0])
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[
                nondominated_point_mask
            ]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        return is_efficient

    costs = np.array(
        [
            [costs[ConfDict()["obj_metrics"][0]], costs[ConfDict()["obj_metrics"][1]]]
            for _, costs in history
        ]
    )
    # average_costs = np.array([np.average(cost) for _, cost in costs])
    # configs = [config for config, _ in history]

    pareto_costs = [costs[i] for i in _get_pareto_indeces(costs)]
    # average_pareto_costs = [average_costs[i] for i in get_pareto_indeces(costs)]
    # pareto_configs = [configs[i] for i in get_pareto_indeces(costs)]

    return {"costs": costs, "pareto_costs": pareto_costs}


def get_pareto_from_smac(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    costs = []
    pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            pareto_costs += [average_cost]
        else:
            costs += [average_cost]

    return {"costs": costs, "pareto_costs": pareto_costs}


def plot_pareto(summary, output_path):
    # Let's work with a numpy array
    costs = np.vstack(summary["costs"])
    pareto_costs = np.vstack(summary["pareto_costs"])
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(costs_x, costs_y, marker="x", label="Configuration")
    ax.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    ax.step(
        [pareto_costs_x[0]]
        + pareto_costs_x.tolist()
        + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)]
        + pareto_costs_y.tolist()
        + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    ax.set_xlim(
        [
            ConfDict()["objectives"][0]["lower_bound"],
            ConfDict()["objectives"][0]["upper_bound"],
        ]
    )
    ax.set_ylim(
        [
            ConfDict()["objectives"][1]["lower_bound"],
            ConfDict()["objectives"][1]["upper_bound"],
        ]
    )
    ax.set_title("Pareto-Front")
    ax.set_xlabel(ConfDict()["obj_metrics"][0])
    ax.set_ylabel(ConfDict()["obj_metrics"][1])
    ax.legend()
    fig.savefig(output_path)


def plot_pareto_from_history(history: list[tuple[Configuration, dict]], output_path):
    plot_pareto(
        get_pareto_from_history(history),
        output_path,
    )
