from abc import abstractmethod

import numpy as np


def normalize(V):
    V_max = np.max(V, axis=0) + np.random.rand() * 1e-8
    V_min = np.min(V, axis=0) + np.random.rand() * 1e-8
    return (V - V_min) / (V_max - V_min)


class SubsetSelection:
    def do(self, pop, n_select, **kwargs):
        if n_select >= len(pop):
            return np.full(len(pop), True)

        selected = self._do(pop, n_select, **kwargs)
        return np.where(selected)[0]

    @abstractmethod
    def _do(self, pop, n_select, **kwargs):
        pass
