import pfevaluator
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pymoo.core.indicator import Indicator
from pymoo.indicators.distance_indicator import (
    at_least_2d_array,
    derive_ideal_and_nadir_from_pf,
)


# =========================================================================================================
# Implementation
# =========================================================================================================


class Spread(Indicator):
    def __init__(self, ideal=None, nadir=None):
        """Spacing indicator
        The smaller the value this indicator assumes, the most uniform is the distribution of elements on the pareto front.

        Parameters
        ----------

        ideal : 1d array, optional
            Ideal point, by default None

        nadir : 1d array, optional
            Nadir point, by default None
        """

        super().__init__(ideal=ideal, nadir=nadir)

        self.metrics = ["MS"]

    def do(self, F, *args, **kwargs):
        """Obtain the spacing indicator given a Pareto front

        Parameters
        ----------
        F : numpy.array (n_samples, n_obj)
            Pareto front

        Returns
        -------
        float
            Spacing indicator
        """
        return super().do(F, *args, **kwargs)

    def _do(self, F, *args, **kwargs):
        return pfevaluator.metric_tpfront(
            pareto_front=F,
            reference_front=np.array([self.nadir, self.ideal]),
            metrics=["MS"],
        )["MS"]
