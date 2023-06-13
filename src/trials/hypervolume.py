from pymoo.indicators.hv import Hypervolume
from performance.r2 import R2
from performance.spacing import Spacing
import numpy as np
from performance.spread import Spread
import pfevaluator

from scipy.spatial import distance


def calc_crowding_distance(F):
    infinity = 1e14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:
        # sort each column and get index
        I = np.argsort(F, axis=0, kind="mergesort")

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate(
            [np.full((1, n_obj), -np.inf), F]
        )

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = (
            np.sum(
                dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)],
                axis=1,
            )
            / n_obj
        )

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


A = np.array([[2, 0.5], [4, 0.4], [6, 0.3], [8, 0.2]])
B = np.array([[1, 0.5], [2, 0.4], [3, 0.3], [5, 0.2]])

ref_point = np.array([10, 1])
ideal_point = np.array([0, 0])

indHV = Hypervolume(ref_point=ref_point)
print("HV", indHV(A))
print("HV", indHV(B))

indSP = Spacing()
print("SP", indSP(A))
print("SP", indSP(B))

indMS = Spread(nadir=ref_point, ideal=ideal_point)
print("MS", indMS(A))
print("MS", indMS(B))

indR2 = R2(ideal=ideal_point)
print("R2", indR2(A))
print("R2", indR2(B))
