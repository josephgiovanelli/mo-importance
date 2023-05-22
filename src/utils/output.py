import math
import os
from utils.input import ConfDict

import numpy as np
import pandas as pd

import json


def adapt_to_mode(value, mode):
    return value if mode == "min" else 1 - value


def adapt_paretos(paretos):
    for obj_idx in range(len(ConfDict()["objectives"])):
        if ConfDict()["obj_modes"][obj_idx] == "max":
            for pareto in paretos:
                for conf in pareto:
                    conf["evaluation"][
                        ConfDict()["obj_metrics"][obj_idx]
                    ] = adapt_to_mode(
                        conf["evaluation"][ConfDict()["obj_metrics"][obj_idx]],
                        ConfDict()["obj_modes"][obj_idx],
                    )


def update_config(paretos):
    for obj_idx in range(len(ConfDict()["objectives"])):
        for bound in ["upper_bound", "lower_bound"]:
            func = np.max if bound == "upper_bound" else np.min
            if bound not in ConfDict()["objectives"][obj_idx]:
                ConfDict()["objectives"][obj_idx][bound] = func(
                    [
                        conf["evaluation"][ConfDict()["obj_metrics"][obj_idx]]
                        for pareto in paretos
                        for conf in pareto
                        if not math.isnan(
                            conf["evaluation"][ConfDict()["obj_metrics"][obj_idx]]
                        )
                    ]
                )


def save_paretos(paretos, file_name):
    with open(os.path.join(ConfDict()["output_folder"], f"{file_name}.json"), "w") as f:
        json.dump({idx: pareto for idx, pareto in enumerate(paretos)}, f)


def save_preferences(preferences):
    preferences.to_csv(
        os.path.join(ConfDict()["output_folder"], "preferences.csv"), index=False
    )


def check_dump():
    return os.path.isfile(os.path.join(ConfDict()["output_folder"], "dump.json"))


def check_encoded():
    return os.path.isfile(os.path.join(ConfDict()["output_folder"], "encoded.json"))


def check_preferences():
    return os.path.isfile(os.path.join(ConfDict()["output_folder"], "preferences.csv"))


def check_pictures():
    return all(
        [
            os.path.isfile(os.path.join(ConfDict()["output_folder"], f"{idx}.png"))
            for idx in range(ConfDict()["random_samples"])
        ]
    )


def load_dump():
    with open(os.path.join(ConfDict()["output_folder"], "dump.json")) as file:
        dump = json.load(file)
    return dump.values()


def load_encoded():
    with open(os.path.join(ConfDict()["output_folder"], "encoded.json")) as file:
        encoded = json.load(file)
    return encoded


def load_preferences():
    return pd.read_csv(os.path.join(ConfDict()["output_folder"], "preferences.csv"))
