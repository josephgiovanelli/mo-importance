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

from pymoo.indicators.hv import Hypervolume
from performance.r2 import R2
from performance.spacing import Spacing
from performance.spread import Spread

from utils.argparse import parse_args
from utils.input import ConfDict, create_configuration
from utils.output import (
    load_encoded,
    check_preferences,
    load_preferences,
    save_preferences,
    adapt_to_mode,
)


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

    paretos = load_encoded(ConfDict()["output_folder"])
    ref_point, ideal_point = [0, 0], [0, 0]
    for obj_idx in range(len(ConfDict()["objectives"])):
        ref_point[obj_idx] = adapt_to_mode(
            ConfDict()["objectives"][obj_idx]["upper_bound"]
            if ConfDict()["obj_modes"][obj_idx] == "min"
            else ConfDict()["objectives"][obj_idx]["lower_bound"],
            ConfDict()["obj_modes"][obj_idx],
        )

        if ConfDict()["obj_modes"][obj_idx] == "max":
            for pareto in paretos.values():
                for conf in pareto:
                    conf[obj_idx] = adapt_to_mode(
                        conf[obj_idx],
                        ConfDict()["obj_modes"][obj_idx],
                    )

    indicators = {
        "hv": {
            "indicator": Hypervolume(ref_point=ref_point),
            "mode": getattr(pd.Series, "idxmax"),
        },
        "sp": {"indicator": Spacing(), "mode": getattr(pd.Series, "idxmin")},
        "ms": {
            "indicator": Spread(nadir=ref_point, ideal=ideal_point),
            "mode": getattr(pd.Series, "idxmax"),
        },
        "r2": {
            "indicator": R2(ideal=ideal_point),
            "mode": getattr(pd.Series, "idxmin"),
        },
    }

    preferences = pd.DataFrame()

    for pair in combinations:
        scores = [
            {
                f"pair_{idx}": pair[idx],
                f"score_{idx}_{acronym}": indicator["indicator"](encoded),
            }
            for idx, encoded in enumerate(
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
