import torch
import numpy as np
from torch import Tensor
from oracles.base import BaseOracle
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


class L1RegressionOracle(BaseOracle):
    def __init__(self, X: Tensor, y: Tensor, device: str = 'cpu', W_init: Tensor = None,
                 b_init: Tensor = None):
        self.X = X
        self.y = y
        if W_init is None:
            W_init = torch.ones(X.shape[1], 1, dtype=X.dtype, requires_grad=True) / np.sqrt(X.shape[1])

        elif not isinstance(W_init, Tensor):
            W_init = Tensor(W_init)

        if b_init is None:
            b_init = torch.ones(1, 1, dtype=X.dtype, requires_grad=True)
        elif not isinstance(b_init, Tensor):
            b_init = Tensor(b_init)

        self.W = W_init.clone().detach().requires_grad_(True).to(device)
        self.b = b_init.clone().detach().requires_grad_(True).to(device)
        self.loss = torch.nn.L1Loss(reduction="mean")

    @torch.no_grad()
    def __call__(self):
        return self.__f().item()

    def __f(self):
        preds = self.X @ self.W + self.b
        return self.loss(preds[:, 0], self.y)

    def grad(self) -> list[Tensor]:
        if self.W.grad is not None and self.b.grad is not None:
            self.W.grad.data.zero_()
            self.b.grad.data.zero_()
        loss = self.__f()
        loss.backward()
        return [self.W.grad, self.b.grad]

    def get_params(self) -> list[Tensor]:
        return [self.W.detach(), self.b.detach()]

    def set_params(self, params: list[Tensor]):
        assert len(params) == 2
        assert self.W.shape == params[0].shape
        assert self.b.shape == params[1].shape
        self.W.data = params[0]
        self.b.data = params[1]

    def metrics(self):
        preds = (self.X @ self.W + self.b).detach().numpy()
        y = np.maximum(self.y, 0)
        return {
            "f1": f1_score(y, preds > .5),
            "roc-auc": roc_auc_score(y, preds),
            "acc": accuracy_score(y, preds > .5)
        }
