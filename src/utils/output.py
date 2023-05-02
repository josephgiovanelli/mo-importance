from utils.input import ConfDict

import numpy as np


def adapt_to_mode(value, mode):
    return value if mode == "min" else 1 - value


def adapt_paretos(paretos):
    for obj_idx in range(len(ConfDict()["objectives"])):
        if ConfDict()["obj_modes"][obj_idx] == "max":
            for pareto in paretos:
                for conf in pareto:
                    conf[1][ConfDict()["obj_metrics"][obj_idx]] = adapt_to_mode(
                        conf[1][ConfDict()["obj_metrics"][obj_idx]],
                        ConfDict()["obj_modes"][obj_idx],
                    )

        for bound in ["upper_bound", "lower_bound"]:
            func = np.max if bound == "upper_bound" else "lower_bound"
            if bound not in ConfDict()["objectives"][obj_idx]:
                ConfDict()["objectives"][obj_idx][bound] = func(
                    [
                        conf[1][ConfDict()["obj_metrics"][obj_idx]]
                        for pareto in paretos
                        for conf in pareto
                    ]
                )
