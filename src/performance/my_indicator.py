import abc

import numpy as np

from pymoo.core.indicator import Indicator


from utils.input import ConfDict


class MyIndicator(Indicator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # what should an indicator return if no solutions are provided is defined here
        self.default_if_empty = 0.0

    def __call__(self, F, *args, **kwargs):
        return self.do(F, *args, **kwargs)

    def do(self, F, *args, **kwargs):
        # if it is a 1d array
        if F.ndim == 1:
            F = F[None, :]

        # if no points have been provided just return the default
        if len(F) == 0:
            return self.default_if_empty

        # do the normalization - will only be done if zero_to_one is enabled
        F = self.normalization.forward(F)

        return self._do(
            np.array(
                [
                    [
                        f[idx] / max(1, ConfDict()["objectives"][idx]["upper_bound"])
                        for idx in range(len(f))
                    ]
                    for f in F
                ]
            ),
            *args,
            **kwargs
        )

    @abc.abstractmethod
    def _do(self, F, *args, **kwargs):
        return
