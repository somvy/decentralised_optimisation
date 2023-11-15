from typing import Any

import torch
from torch import Tensor
from wandb.sdk.wandb_run import Run

from decentralized.topologies import Topologies
from methods import BaseDecentralizedMethod
from oracles.base import BaseOracle


class DecentralizedGradientDescent(BaseDecentralizedMethod):
    def __init__(
            self,
            oracles: list[BaseOracle],
            topology: Topologies,
            wandbrun: Run,
            stepsize: float,
            max_iter: int
    ):
        assert topology.matrix_type.startswith(
            "mixing"), "Decentralized GD works with mixing matrices only!"
        super().__init__(oracles, topology, wandbrun)
        self.step_size: float = stepsize
        self.max_iter: int = max_iter

    def step(self, k: int = 1):
        x: Tensor = self.to_vector_form([oracle.get_params() for oracle in self.oracles])
        grad: Tensor = self.to_vector_form([oracle.grad() for oracle in self.oracles])
        mixing_matrix: Tensor = Tensor(next(self.topology))

        # apply mixing matrix to params
        x_next: Tensor = torch.matmul(mixing_matrix, x) - self.step_size * grad

        self.wandb.log({
            "x diff norm": (x - x_next).norm().mean(),
            "consensus 0-1": (x[0] - x[1]).norm().mean(),
            "consensus 0-last": (x[0] - x[-1]).norm().mean(),
            "grad norm": grad.norm().mean(),
            "grad max": grad.norm().max()
        },
            step=self.step_num
        )

        x_next_list: list[list[Tensor]] = self.to_list_form(x_next)
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(x_next_list[oracle_num])

    def log(self) -> dict[str, Any]:
        losses = [oracle() for oracle in self.oracles]
        return {
            "loss": sum(losses) / len(losses),
            "losses": losses
        }
