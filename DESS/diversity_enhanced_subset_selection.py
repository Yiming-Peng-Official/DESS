import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from subset_selection import SubsetSelection, normalize
from pymoo.subset_selection.distance_based_subset_selection import DistanceBasedSubsetSelection


class DBCANClustering:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def do(self, V):
        """
        Cluster all solution
        Parameters
        ----------
        eps: neighborhood radius
        min_samples: minimum number of solutions within radius eps
        V: candidate solutions

        Returns
        -------
        cluster labels
        """
        # do clustering
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return model.fit(V).labels_


class DiversityEnhancedSubsetSelection(SubsetSelection):
    def __init__(self, delta_max, clustering, objective_selector=DistanceBasedSubsetSelection()):
        self.objective_selector = objective_selector
        self.delta_max = delta_max
        self.clustering = clustering
        super().__init__()

    def _do(self, pop, n_select, **kwargs):
        # normalize
        X = normalize(pop.get('X'))
        F = normalize(pop.get('F'))
        pop_size = len(pop)

        n_var, n_obj = X.shape[1], F.shape[1]

        # Step 1: objective space selection
        S_o = self.objective_selector.do(pop, n_select)

        # Step 2: select equivalent decision vectors
        links = np.full(pop_size, 0)
        Graph = np.full((pop_size, pop_size), False)

        for yi in S_o:
            I = self.select_equivalent_dvs(X, F, yi)
            for xi in I:
                links[xi] += 1
                Graph[xi, yi] = True

        # Step 3: final subset selection
        selected = np.full(len(pop), False)
        cnt = 0
        while cnt < n_select:
            # (1) randomly select a boundary solution in the decision space as an initial solution
            if cnt == 0:
                S_e = np.where(links > 0)[0]
                i = int(np.argmin(X[S_e, np.random.randint(0, n_var)]))
                xi = S_e[i]
            # (2) DSS in the decision space
            else:
                # only select decision vectors which are associated with some objective vectors
                remaining = np.where((links > 0) & (~selected))[0]
                # if all objective vectors in S_o have already been covered, apply only distance-based selection
                if not any(remaining):
                    remaining = np.where(~selected)[0]
                PD = cdist(X[remaining], X[selected])
                ND = np.min(PD, axis=1)  # distance to nearest neighbors in the selected set
                # maximize such distance
                j = np.argmax(ND, axis=0)
                xi = remaining[j]
            # select this solution
            selected[xi] = True
            # select a connected objective vector yi in S_o which is closest to xi in the objective space
            OV = np.argwhere(Graph[xi, :]).flatten()
            if len(OV) > 0:
                D = cdist(F[OV], np.asarray([F[xi]]))
                j = np.argmin(D)
                yi = OV[j]
                # remove all children of yi
                links[Graph[:, yi]] -= 1
                Graph[:, yi] = False
            # increase the counter
            cnt += 1
        return selected

    def select_equivalent_dvs(self, X, F, yi):
        # find all candidate equivalent decision vectors for an objective vector yi
        D = cdist(F, np.asarray([F[yi]]), metric="chebyshev")
        candidate_indices = np.where(D <= self.delta_max)[0]
        # clustering according to decision variables
        cluster_labels = self.clustering.do(X[candidate_indices])
        n_clusters = np.max(cluster_labels) + 1
        # for each cluster, select the solution closest to solution i in the objective space
        selected = np.zeros(n_clusters, dtype=np.int)
        for c in range(n_clusters):
            element_indices = candidate_indices[cluster_labels == c]
            D = cdist(F[element_indices], np.asarray([F[yi]]))
            j = np.argmin(D)
            selected[c] = element_indices[j]
        return selected
