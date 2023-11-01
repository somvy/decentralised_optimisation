from typing import Callable, Any

import torch
from torch import Tensor
from copy import deepcopy

from wandb.sdk.wandb_run import Run

from methods.base import BaseDecentralizedMethod
from tqdm.auto import trange


class PROXNSADOM(BaseDecentralizedMethod):
    def __init__(self, oracles, topology, max_iter,
                 eta, theta, r, gamma, wandbrun: Run, saddle_lr=5e-4):
        """

        :param oracles:
        :param topology:
        :param stepsize:
        :param max_iter:
        :param params:
        r - regularization param
        eta, theta

        """
        super().__init__(oracles, topology)
        self.param_dims: list[torch.Size] = [p.shape for p in self.oracles[0].get_params()]
        self.max_iter: int = max_iter
        self.eta = eta
        self.theta = theta
        self.gamma = gamma
        self.r = r
        self.saddle_lr = saddle_lr
        self.wandb = wandbrun
        self.step_num = 0

        # oracles x layers x params
        self.x: list[list[Tensor]] = [oracle.get_params() for oracle in self.oracles]
        self.xu: list[list[Tensor]] = deepcopy(self.x)
        self.y: list[list[Tensor]] = deepcopy(self.x)
        # y upper
        self.yu: list[list[Tensor]] = deepcopy(self.x)

        self.a: float = 0
        # initialize z in L orthogonal
        # oracles x layers x params
        self.z: list[list[Tensor]] = [[torch.zeros_like(param) for param in oracles] for oracles in self.x]
        self.zu: list[list[Tensor]] = deepcopy(self.z)
        self.m: list[list[Tensor]] = deepcopy(self.z)

    def grad_f(self, x: Tensor) -> Tensor:
        """
        x: n x d matrix of params
        return n x d matrix of gradients
        """
        # print("grad f ", x.shape, self.param_dims)
        x: list[list[Tensor]] = self.to_list_form(x)

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(x[oracle_num])

        grad: list[list[Tensor]] = [oracle.grad() for oracle in self.oracles]

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(self.x[oracle_num])

        return self.to_vector_form(grad)

    def step(self):
        gossip_matrix = torch.Tensor(next(self.topology))

        prev_alpha = 1 / self.a if self.step_num != 1 else 1

        self.a = (1 + (1 + 4 * self.a ** 2) ** 0.5) / 2
        alpha = 1 / self.a
        eta = self.eta * self.a
        theta = self.theta * self.a

        # n x d
        x, y, z, xu, yu, zu, m = map(
            self.to_vector_form, (self.x, self.y, self.z, self.xu, self.yu, self.zu, self.m)
        )
        yl = alpha * y + (1 - alpha) * yu
        zl = alpha * z + (1 - alpha) * zu
        g_grad = self.grad_g(yl, zl)
        gossip_nesterov = gossip_matrix @ (m - theta * g_grad)
        z_next = z + gossip_nesterov
        m_next = m - theta * g_grad - gossip_nesterov

        x_next, y_next = self.solve_saddle(x, y, yl, zl, eta, theta)
        xnorm_diff = (x_next - x).norm()
        ynorm_diff = (y_next - y).norm()
        step_logs = {
            "xnorm_diff": xnorm_diff,
            "ynorm_diff": ynorm_diff,
            "consensus: ": torch.norm(x_next[0] - x_next[-1])
        }
        if self.wandb:
            self.wandb.log(step_logs, step=self.step_num)
        else:
            print(step_logs)

        if xnorm_diff.isnan() or xnorm_diff.isinf():
            raise StopIteration("xnorm_diff is too low or high, stopping iteration")
        yu_next = yl + alpha * (y_next - y)
        zu_next = zl - self.gamma * gossip_matrix @ g_grad

        # усреднить по x

        xu_next = alpha ** 2 * (xu / prev_alpha + self.a * x_next)
        self.x, self.y, self.z, self.xu, self.yu, self.zu, self.m = map(
            self.to_list_form, (x_next, y_next, z_next, xu_next, yu_next, zu_next, m_next)
        )

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(self.x[oracle_num])

    def grad_g(self, y, z):

        return 1 / self.r * (y + z)

    @staticmethod
    def to_vector_form(x: list[list[Tensor]]) -> Tensor:

        return torch.stack([torch.cat([param.view(-1) for param in oracle_params])
                            for oracle_params in x])

    def to_list_form(self, x: Tensor) -> list[list[Tensor]]:
        def to_tensor_list(x: Tensor):
            "x: vector of params"
            xs: list[Tensor] = []
            for param_dim in self.param_dims:
                xs.append(x[:param_dim.numel()].reshape(param_dim))
                x = x[param_dim.numel():]
            return xs

        return list(map(to_tensor_list, x))

    def run(self, log: bool = False, disable_tqdm=True):
        loop = trange(1, self.max_iter + 1, disable=disable_tqdm)
        for k in loop:
            self.step_num = k
            self.step()
            if log:
                log = self.log()
                self.logs.append(log)
                if self.wandb:
                    self.wandb.log(log, step=self.step_num)

    def log(self) -> dict[str, Any]:
        losses = [float(oracle()) for oracle in self.oracles]

        return {
            "loss": sum(losses) / len(losses),
            "losses": losses,
            "eta": self.eta,
            "theta": self.theta,
            "gamma": self.gamma,
            "a": self.a
        }

    def G(self, y, z):
        return 1 / (2 * self.r) * (y + z).norm() ** 2

    def gradG(self, y, z):
        return 1 / self.r * (y + z)

    def inner_saddle(self, x, y, xk, yk, yk_, zk_, eta, theta, f):
        return 1 / (2 * eta) * (x - xk).norm() ** 2 + f(x) - x @ y - (self.gradG(yk_, zk_) @ y) - 1 / (
                2 * theta) * (y - yk).norm() ** 2

    def solve_saddle(self, xk, yk, yk_, zk_, eta, theta):
        x = xk.clone()
        y = yk.clone()
        grad_x = lambda X, Y: 1 / eta * (X - xk) + self.grad_f(X) - Y

        for k in range(1, 800):
            x = x - self.saddle_lr / (k ** (1 / 2)) * grad_x(x, yk)    
        y = yk - x * theta - self.gradG(yk_, zk_) * theta
        
        after_grad = grad_x(x, y)
        inner_saddle_log = {
            "inner saddle grad norm": after_grad.norm() / after_grad.numel(),
            "inner saggle grad max": after_grad.max()
        }
        if self.wandb:
            self.wandb.log(inner_saddle_log, step=self.step_num)
        return x, y
