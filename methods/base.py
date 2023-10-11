from oracles.base import BaseOracle
from decentralized.topologies import Topologies
import numpy as np

class BaseDecentralizedMethod:
    def __init__(self, oracles: list[BaseOracle], topology: Topologies, x_0: np.array):
        assert len(oracles) == x_0.shape[0], "X0 should have the same number of rows as the number of oracles"
        self.oracles: list[BaseOracle] = oracles
        self.topology: Topologies = topology
        self.x = x_0.copy()

    def run(self):
        pass

    def step(self):
        pass



