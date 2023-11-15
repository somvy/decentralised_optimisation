from torch import Tensor, Size
from tqdm.auto import trange
import torch
from oracles.base import BaseOracle
from wandb.wandb_run import Run


class SingleGD:
    def __init__(self, oracle: BaseOracle, wandbrun: Run, max_iter: int, lr: float) -> None:
        self.oracle: BaseOracle = oracle
        self.wandb: Run = wandbrun
        self.step_num: int = 0
        self.param_dims: list[Size] = [p.shape for p in self.oracle.get_params()]
        self.max_iter: int = max_iter
        self.lr: float = lr

    def step(self):
        x: Tensor = torch.cat([param.view(-1) for param in self.oracle.get_params()])
        grad: Tensor = torch.cat([g.view(-1) for g in self.oracle.grad()])
        x_next = x - self.lr * grad

        self.wandb.log(
            {"x_diff": (x - x_next).norm().mean(),
             "grad norm": grad.norm().mean(),
             "grad max": grad.norm().max()
             },
            step=self.step_num

        )
        x_next_list: list[Tensor] = []
        for param_dim in self.param_dims:
            x_next_list.append(x_next[:param_dim.numel()].reshape(param_dim))
            x_next = x_next[param_dim.numel():]

        self.oracle.set_params(x_next_list)

    def log(self):
        return {
            "loss": self.oracle()
        }

    def run(self, log: bool = False, disable_tqdm=True):
        for k in trange(1, self.max_iter + 1, disable=disable_tqdm):
            self.step_num = k
            self.step()
            if log:
                log = self.log()
                if self.wandb:
                    self.wandb.log(log, step=self.step_num)
