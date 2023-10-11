from oracles.base import BaseOracle


class QuadraticOracle(BaseOracle):
    """
    Oracle for quadratic function 1/2 x.T A x - b.T x.

    Parameters
    ----------
    A: dim_size x dim_size

    b: dim_size x 1
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def __call__(self, x):
        return 0.5 * (x.T @ self.A) @ x - self.b @ x

    def grad(self, x):
        return self.A @ x - self.b

    def subgrad(self, x):
        return self.grad(x)
