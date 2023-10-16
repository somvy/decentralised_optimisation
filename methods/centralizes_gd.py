from typing import Any

import numpy as np
import torch
from torch import tensor

from decentralized.topologies import Topologies
from methods.base import BaseDecentralizedMethod
from oracles.base import BaseOracle


class CentralizedGradientDescent(BaseDecentralizedMethod):
    def __init__(self, oracles: list[BaseOracle], topology: Topologies, stepsize: float,
                 max_iter: int):
        super().__init__(oracles, topology)
        self.step_size: float = stepsize
        self.max_iter: int = max_iter

    def step(self):
        total_parameters: list[list[tensor]] = [oracle.get_params() for oracle in self.oracles]
        total_gradients: list[list[tensor]] = [oracle.grad() for oracle in self.oracles]
        layers_count: int = len(total_parameters[0])
        new_params_by_layer: list[tensor] = []

        for layer_num in range(layers_count):
            x: tensor = torch.stack([oracle_params[layer_num] for oracle_params in total_parameters])
            layer_gradients = torch.stack([oracle_grads[layer_num] for oracle_grads in total_gradients])
            layer_gradients = layer_gradients.mean(axis=0)
            x_next = x - self.step_size * layer_gradients
            new_params_by_layer.append(x_next)

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [new_params_by_layer[layer_num][oracle_num] for layer_num in range(layers_count)]
            )

    def run(self, log: bool = False):
        for _ in range(self.max_iter):
            self.step()
            if log:
                self.logs.append(self.log())

    def log(self) -> dict[str, Any]:
        losses = [oracle() for oracle in self.oracles]
        return {
            "loss": sum(losses) / len(losses),
            "losses": losses
        }