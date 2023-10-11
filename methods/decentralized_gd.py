import numpy as np

from decentralized.topologies import Topologies
from methods.base import BaseDecentralizedMethod
from oracles.base import BaseOracle


class DecentralizedGradientDescent(BaseDecentralizedMethod):
    def __init__(self, oracles: list[BaseOracle], topology: Topologies, x_0: np.array, stepsize: float,
                 max_iter: int):
        super().__init__(oracles, topology, x_0)
        self.step_size: float = stepsize
        self.max_iter: int = max_iter

    def step(self):
        assert self.topology.matrix_type.startswith("gossip"), "works with gossip matrices only"
        gossip_matrix = next(self.topology)
        x_next = gossip_matrix @ self.x
        gradients = np.vstack([oracle.grad(x_next[i]) for i, oracle in enumerate(self.oracles)])
        self.x = x_next - self.step_size * gradients

    def run(self):
        for _ in range(self.max_iter):
            self.step()
