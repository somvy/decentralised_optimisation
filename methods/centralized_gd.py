from typing import Any

from torch import Tensor

from decentralized.topologies import Topologies
from methods.base import BaseDecentralizedMethod
from oracles.base import BaseOracle
from wandb.sdk.wandb_run import Run


class CentralizedGradientDescent(BaseDecentralizedMethod):
    def __init__(self, oracles: list[BaseOracle], topology: Topologies, stepsize: float,
                 max_iter: int, wandbrun: Run):
        super().__init__(oracles, topology, wandbrun)
        self.step_size: float = stepsize
        self.max_iter: int = max_iter
        self.x: list[list[Tensor]] = [oracle.get_params() for oracle in self.oracles]

    def step(self):
        x: Tensor = self.to_vector_form(self.x)
        grad: Tensor = self.to_vector_form([oracle.grad() for oracle in self.oracles]).mean(dim=0)
        x_next: Tensor = x - self.step_size * grad
        self.wandb.log(
            {
                "grad norm": grad.norm(),
                "grad max": grad.mean().max(),
                "x diff": (x_next - x).norm()
            },
            step=self.step_num
        )

        self.x = self.to_list_form(x_next)
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(self.x[oracle_num])

    def log(self) -> dict[str, Any]:
        losses = [oracle() for oracle in self.oracles]
        return {
            "loss": sum(losses) / len(losses)
        }
