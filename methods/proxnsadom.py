from typing import Callable, Any

import torch
from torch import Tensor
from copy import deepcopy

from wandb.sdk.wandb_run import Run
from oracles.base import BaseOracle
from decentralized.topologies import Topologies
from methods import BaseDecentralizedMethod


class PROXNSADOM(BaseDecentralizedMethod):
    def __init__(
            self,
            oracles: list[BaseOracle],
            topology: Topologies,
            max_iter: int,
            reg: float,
            wandbrun: Run,
            saddle_lr: float = 5e-4,
            saddle_iters: int = 1000
    ):
        """

        :param oracles:
        :param topology:
        :param stepsize:
        :param max_iter:
        :param params:
        reg - regularization param
        eta, theta

        """
        super().__init__(oracles, topology, wandbrun)
        self.max_iter: int = max_iter

        chi: float = topology.chi
        self.eta: float = 1 / (64 * reg * chi ** 2)
        self.theta: float = reg / (16 * chi ** 2)
        self.gamma: float = reg / 2
        self.r: float = reg
        self.saddle_lr: float = saddle_lr
        self.saddle_iters: int = saddle_iters

        self.wandb.config.update({
            "saddle_lr": saddle_lr,
            "sadle iters": saddle_iters,
            "topology": topology.topology_type,
            "topology matrix": topology.matrix_type,
            "topology chi": topology.chi,
            "model params count": sum([p.numel() for p in self.oracles[0].get_params()])

        })
        self.step_num: int = 0

        # oracles x layers x params
        self.x: list[list[Tensor]] = [oracle.get_params() for oracle in self.oracles]
        # x upper - x weigted mean by iters
        self.xu: list[list[Tensor]] = deepcopy(self.x)
        # x upper mean by nodes
        self.xu_mean: list[Tensor] = deepcopy(self.x[0])
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
        g_grad = self.gradG(yl, zl)
        gossip_nesterov = gossip_matrix @ (m - theta * g_grad)
        z_next = z + gossip_nesterov
        m_next = m - theta * g_grad - gossip_nesterov

        x_next, y_next = self.solve_saddle(x, y, yl, zl, eta, theta)
        xnorm_diff = (x_next - x).norm()
        ynorm_diff = (y_next - y).norm()
        step_logs = {
            "xnorm diff": xnorm_diff,
            "ynorm diff": ynorm_diff,
            "consensus": torch.norm(x_next[0] - x_next[-1])
        }

        if xnorm_diff.isnan() or xnorm_diff.isinf():
            raise StopIteration("xnorm_diff is too low or high, stopping iteration")

        yu_next = yl + alpha * (y_next - y)
        zu_next = zl - self.gamma * gossip_matrix @ g_grad

        # усредняем х по итерациям
        xu_next = alpha ** 2 * (xu / prev_alpha + self.a * x_next)
        self.x, self.y, self.z, self.xu, self.yu, self.zu, self.m = map(
            self.to_list_form, (x_next, y_next, z_next, xu_next, yu_next, zu_next, m_next)
        )

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(self.xu[oracle_num])
        step_logs["x mean by iters loss"] = sum([float(oracle()) for oracle in self.oracles]) / len(
            self.oracles)

        self.xu_mean = [torch.mean(torch.stack(layer_weights), dim=0) for layer_weights in zip(*self.xu)]
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(self.xu_mean)
        step_logs["x mean by iters and nodes loss"] = sum([float(oracle()) for oracle in self.oracles]) / len(
            self.oracles)

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(self.x[oracle_num])
        step_logs["x loss"] = sum([float(oracle()) for oracle in self.oracles]) / len(self.oracles)

        self.wandb.log(step_logs, step=self.step_num)

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
        # grad_x = lambda X, Y: 1 / eta * (X - xk) + self.grad_f(X) - Y
        grad_x = lambda x, y: (1 / eta * (x - xk) + 3 * theta * x + 3 * theta * self.gradG(yk_, zk_)
                               - yk + self.grad_f(x))

        for k in range(1, self.saddle_iters + 1):
            x = (x - self.saddle_lr
                 # / (k ** (1 / 2))
                 * grad_x(x, yk))
        y = yk - x * theta - self.gradG(yk_, zk_) * theta
        # y = yk - x * theta - self.gradG(yk_, zk_) * theta
        # x = x - self.saddle_lr / (k ** (1 / 2)) * grad_x(x, y)
        after_grad = grad_x(x, y)
        inner_saddle_log = {
            "inner saddle grad norm": after_grad.norm() / after_grad.numel(),
            "inner saggle grad max": after_grad.max()
        }
        if self.wandb:
            self.wandb.log(inner_saddle_log, step=self.step_num)
        return x, y
s
