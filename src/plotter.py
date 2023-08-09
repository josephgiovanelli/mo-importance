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


def plot_mean_std(df, indicators, output_path):
    x_ticks = df.columns
    x_labels = "No. comparisons"

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


def plot_preferencce_evaluation(output_path):
    input_path = make_dir(os.path.join("/", "home", "output", "preference"))
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
    print(results)
    plot_mean_std(results, indicators, output_path)


def export_end_to_end_evaluation(output_path):
    input_path = make_dir(os.path.join("/", "home", "output", "summary"))
    output_path = make_dir(os.path.join("/", "home", "plots"))

    def to_latex(cell):
        struct = {
            f"""{"indicator" if idx == 0 else "preference"}""": {
                "mean": content.split("  ")[0],
                "std": content.split("  ")[1],
            }
            for idx, content in enumerate(cell.split("/", 2))
        }
        return (
            "\\begin{tabular}{ccc} "
            + (
                "\\textbf{"
                if float(struct["preference"]["mean"])
                > float(struct["indicator"]["mean"])
                else ""
            )
            + struct["preference"]["mean"]
            + (
                "}"
                if float(struct["preference"]["mean"])
                > float(struct["indicator"]["mean"])
                else ""
            )
            + "  & \multirow{2}{*}{\Large{$\\backslash$}} & "
            + (
                "\\textbf{"
                if float(struct["indicator"]["mean"])
                > float(struct["preference"]["mean"])
                else ""
            )
            + struct["indicator"]["mean"]
            + (
                "}"
                if float(struct["indicator"]["mean"])
                > float(struct["preference"]["mean"])
                else ""
            )
            + " \\\\ "
            + "\\footnotesize{"
            + (
                "\\textbf{"
                if float(struct["preference"]["mean"])
                > float(struct["indicator"]["mean"])
                else ""
            )
            + struct["preference"]["std"]
            + (
                "}"
                if float(struct["preference"]["mean"])
                > float(struct["indicator"]["mean"])
                else ""
            )
            + "}"
            + " & & "
            + "\\footnotesize{"
            + (
                "\\textbf{"
                if float(struct["indicator"]["mean"])
                > float(struct["preference"]["mean"])
                else ""
            )
            + struct["indicator"]["std"]
            + (
                "}"
                if float(struct["indicator"]["mean"])
                > float(struct["preference"]["mean"])
                else ""
            )
            + "}"
            + "\end{tabular}"
        )

    for budget in range(28, 140 + 28, 28):
        results = pd.read_csv(
            os.path.join(input_path, f"budget_{budget}.csv"),
            index_col="indicators\preferences",
        ).applymap(to_latex)

        out = ""
        for row in results.index:
            for col in results.columns:
                prefix, outfix = "", ""
                if row == "hv" and col == "hv":
                    prefix += "\\begin{table*}[!ht]\n\centering\n\t\\begin{tabular}{l|c|c|c|c}\n\t\\toprule\n\t$\mathcal{P} \\backslash \mathcal{I}$ & HV & SP & MS & R2 \\\\ \midrule"
                if col == "hv":
                    prefix += "\n\t" + row.upper() + " & "
                outfix += " & " if col != "r2" else " \\\\ \midrule"
                outfix += (
                    ""
                    if row != "r2" or col != "r2"
                    else (
                        "\n\t\end{tabular}\n\caption{Comparison between optimizing quality indicators (i.e., $\mathcal{I}$, columns) and optimizing the preference learning utilities (i.e., $\mathcal{P}$, rows). The preference learning model is trained using "
                        + str(budget)
                        + " pairwise comparisons.}\label{tbl:end_to_end_evaluation_"
                        + str(budget)
                        + "}\end{table*}"
                    )
                )
                out += prefix + results[row][col] + outfix

        out = out.replace("+-", "$\pm$")
        with open(os.path.join(output_path, f"budget_{budget}.text"), "w") as text_file:
            text_file.write(out)


if __name__ == "__main__":
    output_path = make_dir(os.path.join("/", "home", "plots"))

    plot_preferencce_evaluation(output_path)
    export_end_to_end_evaluation(output_path)
