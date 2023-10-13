import torch
import numpy as np
from torch import tensor
from oracles.base import BaseOracle


class L1RegressionOracle(BaseOracle):
    def __init__(self, X: tensor, y: tensor, device: str = 'cpu', W_init: tensor = None,
                 b_init: tensor = None):
        self.X = X
        self.y = y
        if W_init is None:
            W_init = torch.ones(X.shape[1], 1, dtype=X.dtype, requires_grad=True) / np.sqrt(X.shape[1])

        elif not isinstance(tensor, W_init):
            W_init = tensor(W_init)
        if b_init is None:
            b_init = torch.ones(1, 1, dtype=X.dtype, requires_grad=True)
        elif not isinstance(tensor, b_init):
            b_init = tensor(b_init)

        self.W = W_init.clone().detach().requires_grad_(True).to(device)
        self.b = b_init.clone().detach().requires_grad_(True).to(device)
        self.loss = torch.nn.L1Loss(reduction="mean")

    @torch.no_grad()
    def __call__(self):
        return self.__f().item()

    def __f(self):
        preds = self.X @ self.W + self.b
        return self.loss(preds[:, 0], self.y)

    def grad(self) -> list[tensor]:
        if self.W.grad is not None and self.b.grad is not None:
            self.W.grad.data.zero_()
            self.b.grad.data.zero_()
        loss = self.__f()
        loss.backward()
        return [self.W.grad, self.b.grad]

    def get_params(self) -> list[tensor]:
        return [self.W.detach(), self.b.detach()]

    def set_params(self, params: list[tensor]):
        assert len(params) == 2
        assert self.W.shape == params[0].shape
        assert self.b.shape == params[1].shape
        self.W.data = params[0]
        self.b.data = params[1]
