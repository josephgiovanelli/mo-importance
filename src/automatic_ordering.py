# %%
from collections import ChainMap
import random
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from itertools import combinations
from IPython.display import clear_output


from sklearn.model_selection import KFold


from utils.argparse import parse_args
from utils.input import ConfDict, create_configuration
from utils.output import (
    adapt_paretos,
    load_encoded,
    check_preferences,
    load_preferences,
    save_preferences,
    adapt_to_mode,
    adapt_encoded,
)
from utils.pareto import get_pareto_indicators


def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))


if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(args.conf_file)

    random.seed(ConfDict()["seed"])

    encoded = load_encoded(ConfDict()["output_folder"])
    combinations = np.array(
        [
            rSubset(fold, 2)
            for _, fold in KFold(n_splits=5, random_state=ConfDict()["seed"]).split(
                list(encoded.keys())
            )
        ]
    )
    combinations = combinations.reshape((-1, 2))
    tot = len(combinations)
    # random.shuffle(combinations)

    paretos = adapt_encoded(load_encoded(ConfDict()["output_folder"]))
    indicators = get_pareto_indicators()

    preferences = pd.DataFrame()

    for pair in combinations:
        scores = [
            {
                f"pair_{idx}": pair[idx],
                f"score_{idx}_{acronym}": indicator["indicator"](pareto),
            }
            for idx, pareto in enumerate(
                [np.array(paretos[str(elem)]) for elem in pair]
            )
            for acronym, indicator in indicators.items()
        ]
        scores = dict(ChainMap(*scores))

        preferences = pd.concat(
            [
                preferences,
                pd.DataFrame(
                    dict(
                        ChainMap(
                            *[
                                {key: [value] for key, value in scores.items()},
                                {
                                    f"preference_{acronym}": [
                                        indicator["mode"](
                                            pd.Series(
                                                [
                                                    scores[f"score_{idx}_{acronym}"]
                                                    for idx, _ in enumerate(pair)
                                                ]
                                            )
                                        )
                                    ]
                                    for acronym, indicator in indicators.items()
                                },
                            ]
                        )
                    )
                ),
            ],
            ignore_index=True,
        )
        save_preferences(preferences)

# %%
