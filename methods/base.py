from oracles import BaseOracle
from decentralized.topologies import Topologies
from wandb.sdk.wandb_run import Run
from tqdm.auto import trange
import torch
from torch import Tensor


class BaseDecentralizedMethod:
    def __init__(self, oracles: list[BaseOracle], topology: Topologies, wandb: Run):
        self.oracles: list[BaseOracle] = oracles
        self.n: int = len(oracles)
        self.topology: Topologies = topology
        self.logs: list[dict[str, float]] = []
        self.wandb: Run = wandb
        self.max_iter: int = 0
        self.step_num = 0
        self.param_dims: list[torch.Size] = [
            p.shape for p in self.oracles[0].get_params()
        ]

    def run(self, log: bool = False, disable_tqdm=True):
        for k in trange(1, self.max_iter + 1, disable=disable_tqdm):
            self.step_num = k
            self.step()
            if log:
                log = self.log()
                self.logs.append(log)
                if self.wandb:
                    self.wandb.log(log, step=self.step_num)

    @staticmethod
    def to_vector_form(x: list[list[Tensor]]) -> Tensor:
        return torch.stack(
            [
                torch.cat([param.view(-1) for param in oracle_params])
                for oracle_params in x
            ]
        )

    def to_list_form(self, x: Tensor) -> list[list[Tensor]]:
        def to_tensor_list(x: Tensor):
            "x: vector of params"
            xs: list[Tensor] = []
            for param_dim in self.param_dims:
                xs.append(x[: param_dim.numel()].reshape(param_dim))
                x = x[param_dim.numel() :]
            return xs

        return list(map(to_tensor_list, x))

    def step(self):
        pass

    def log(self) -> dict[str, float]:
        pass
