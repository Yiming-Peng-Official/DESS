from collections.abc import Iterable

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.subset_selection import normalize, SubsetSelection


class DistanceBasedSubsetSelection(SubsetSelection):
    """
    The distance based subset selection method. There are two variants proposed in [1] and [2]. The basic idea is to
    iteratively choose a solution with maximum distance to nearest selected solution.

    Parameters
    ----------
    population: population
    based_on: "F" or "X"
        specify the distance is calculated based on decision values or objective values
    extreme_point: "all" or "one"
        specify how the extreme points will be selected. when set to 'all', all extreme solutions will be selected,
        when set to 'one', DSS randomly selects one extreme solution.
    dist_metric: str
        the distance metric, can be one of ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                                'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                                'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule']

    References
    ----------
    [1] R. Tanabe, H. Ishibuchi, and A. Oyama, “Benchmarking multi- and many-objective evolutionary algorithms under two optimization scenarios,” IEEE Access, vol. 5, pp. 19 597–19 619, September 2017.
    [2] H. K. Singh, K. S. Bhattacharjee, and T. Ray, “Distance-based subset selection for benchmarking in evolutionary multi/many-objective optimization,” IEEE Transactions on Evolutionary Computation, vol. 23, no. 5, pp. 904–912, October 2019.
    """

    def __init__(self, based_on="F", extreme_point="one", dist_metric="euclidean"):
        assert (based_on in ["F", "X"]), "DSS can only based on fitness ('F') or decision variables ('X')!"
        assert (extreme_point in ["one", "all"]), "DSS either randomly selects one extreme points or selects all of " \
                                                  "extreme points! "
        assert (dist_metric in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
                                'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                                'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule']), "Distance metric is invalid!"
        self.base_on = based_on
        self.extreme_points = extreme_point
        self.dist_metric = dist_metric

    def select_extreme_points(self, V):
        [_, D] = V.shape
        if self.extreme_points is "one":
            d = np.random.randint(0, D)
            return np.argmin(V[:, d])
        else:
            return np.argmin(V, axis=0)

    def _do(self, pop, n_select, **kwargs):
        selected = np.full(len(pop), False)
        V = pop.get(self.base_on)
        # normalization
        V = normalize(V)
        # select extreme points
        I = self.select_extreme_points(V)
        selected[I] = True
        # every step select an unselected solution with maximum distance to the nearest neighborhood distance in the
        # selected set util desired number of solutions are selected.
        cnt = len(I) if isinstance(I, Iterable) else 1
        while cnt < n_select:
            remain = np.where(~selected)[0]  # indices of unselected solutions
            # calculate distance from each unselected solution to nearest selected solution.
            PD = cdist(V[~selected], V[selected], metric=self.dist_metric)
            ND = np.min(PD, axis=1)
            # maximize such distance
            j = np.argmax(ND, axis=0)
            i = remain[j]
            selected[i] = True
            cnt = cnt + 1

        return selected
