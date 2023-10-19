from typing import Any

import torch
from torch import tensor
import numpy as np

from decentralized.topologies import Topologies
from methods.base import BaseDecentralizedMethod
from oracles.base import BaseOracle


class DecentralizedGradientDescent(BaseDecentralizedMethod):
    def __init__(self, oracles: list[BaseOracle], topology: Topologies, stepsize: float,
                 max_iter: int):
        super().__init__(oracles, topology)
        self.step_size: float = stepsize
        self.max_iter: int = max_iter

    def step(self, k: int = 1):
        # assert self.topology.matrix_type.startswith("gossip"), "works with gossip matrices only yet"
        total_parameters: list[list[tensor]] = [oracle.get_params() for oracle in self.oracles]
        total_gradients: list[list[tensor]] = [oracle.grad() for oracle in self.oracles]
        layers_count: int = len(total_parameters[0])
        new_params_by_layer: list[tensor] = []
        gossip_matrix: tensor = tensor(next(self.topology), dtype=total_parameters[0][0].dtype)
        # accumulate params and grads by layer, then update params
        for layer_num in range(layers_count):
            x: tensor = torch.stack([oracle_params[layer_num] for oracle_params in total_parameters])
            # apply gossip matrix to params
            n, *d = x.shape
            x_next = torch.matmul(gossip_matrix, x.view(n, -1)).view(n, *d)

            layer_gradients = torch.stack([oracle_grads[layer_num] for oracle_grads in total_gradients])
            x_next -= self.step_size / np.sqrt(k) * layer_gradients
            new_params_by_layer.append(x_next)

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [new_params_by_layer[layer_num][oracle_num] for layer_num in range(layers_count)]
            )

    def run(self, log: bool = False):
        if log:
            self.logs.append(self.log())
        for k in range(1, self.max_iter + 1):
            # print([oracle.get_params() for oracle in self.oracles])
            self.step(k)
            if log:
                self.logs.append(self.log())

    def log(self) -> dict[str, Any]:
        losses = [oracle() for oracle in self.oracles]
        return {
            "loss": sum(losses) / len(losses),
            "losses": losses
        }
