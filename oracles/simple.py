from oracles.base import BaseOracle
from torch import tensor


class QuadraticOracle(BaseOracle):
    """
    Oracle for quadratic function 1/2 x.T A x - b.T x.

    Parameters
    ----------
    A: dim_size x dim_size

    b: dim_size x 1
    """

    def __init__(self, A: tensor, b: tensor, x: tensor) -> None:
        self.A = A
        self.b = b
        self.x = x

    def __call__(self):
        return 0.5 * (self.x.T @ self.A) @ self.x - self.b @ self.x

    def grad(self) -> list[tensor]:
        return [0.5 * (self.x.unsqueeze(1) @ self.x.T.unsqueeze(0)), -self.x]

    def get_params(self) -> list[tensor]:
        return [self.A, self.b]

    def set_params(self, params: list[tensor]):
        assert len(params) == 2
        assert self.A.shape == params[0].shape
        self.A = params[0]
        assert self.b.shape == params[1].shape
        self.b = params[1]

    def subgrad(self) -> list[tensor]:
        return self.grad()
