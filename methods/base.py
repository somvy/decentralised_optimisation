from oracles.base import BaseOracle
from decentralized.topologies import Topologies
import numpy as np


class BaseDecentralizedMethod:
    def __init__(self, oracles: list[BaseOracle], topology: Topologies):
        self.oracles: list[BaseOracle] = oracles
        self.n: int = len(oracles)
        self.topology: Topologies = topology
        self.logs: list[dict[str, float]] = []

    def run(self, log: bool = False):
        pass

    def step(self):
        pass
