from typing import Any

import torch
from torch import Tensor
from copy import deepcopy
from methods import BaseDecentralizedMethod
from oracles import BaseOracle
from decentralized.topologies import Topologies
from wandb.sdk.wandb_run import Run


class DecentralizedCommunicationSliding(BaseDecentralizedMethod):
    def __init__(self, oracles: list[BaseOracle], topology: Topologies, wandbrun: Run, max_iter: int):
        super().__init__(oracles, topology, wandbrun)
        self.max_iter = max_iter

        self.x_all: list[list[Tensor]] = [oracle.get_params() for oracle in self.oracles]
        self.x_ergodic = [[torch.zeros_like(tmp) for tmp in el] for el in self.x_all]
        # initialize z in L orthogonal
        # oracles x layers x params
        self.x_prev = deepcopy(self.x_all)
        self.x_hat = deepcopy(self.x_all)
        self.y = deepcopy(self.x_all)
        self.params = {}
        self.__calculate_params(k=1)

    def __calculate_params(self, k=1):
        chi = self.topology.chi
        mu = self.oracles[0].regularization
        self.params = {
            'mu': mu,
            'eta': k * mu / 2,
            'T': 100,
            'alpha': k / (k + 1),
            'teta': k + 1,
            'tau': 4 * self.topology.norm ** 2 / ((k + 1) * mu)
        }

    def compute_grad_in_u(self, u):
        x = [oracle.get_params() for oracle in self.oracles]  # node x layer x param
        layers_count = len(x[0])
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [u[oracle_num][layer_num] for layer_num in range(layers_count)]
            )
        gradients = [oracle.grad() for oracle in self.oracles]
        # set params back
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                x[oracle_num]
            )
        return gradients

    def communication_sliding(self, x, w):
        layers_count = len(x[0])
        oracles_count = len(self.oracles)
        u = deepcopy(x)
        u_hat = [[torch.zeros_like(tmp) for tmp in el] for el in x]
        for t in range(self.params['T']):
            h = self.compute_grad_in_u(u)
            u_new = []
            for o in range(oracles_count):
                tmp = []
                for l in range(layers_count):
                    self.params['beta'] = t / 2 + (t + 2) / 2 * self.params['mu'] / self.params['eta']
                    u_tensor = 2 * self.params['eta'] * self.params['beta'] * u[o][l]
                    u_tensor += 2 * self.params['eta'] * x[o][l]
                    u_tensor -= w[o][l] + h[o][l]
                    u_tensor /= 2 * self.params['eta'] * (1 + self.params['beta'])
                    tmp.append(u_tensor)
                    u_hat[o][l] += (t + 1) * u_tensor
                u_new.append(tmp)
            u = u_new
        for o in range(oracles_count):
            for l in range(layers_count):
                u_hat[o][l] /= self.params['T'] * (self.params['T'] - 1) / 2 + self.params['T']
        return u, u_hat

    def update_ergodic(self, x_hat, k=0):
        layers_count = len(x_hat[0])
        oracles_count = len(self.oracles)
        for o in range(oracles_count):
            for l in range(layers_count):
                if k != 1:
                    self.x_ergodic[o][l] *= (k - 1) * (k + 2) / 2
                self.x_ergodic[o][l] += (k + 1) * x_hat[o][l]
                self.x_ergodic[o][l] /= k * (k + 3) / 2

    def step(self, k=0):
        assert self.topology.matrix_type.startswith("gossip"), "works with gossip matrices only yet"
        gossip_matrix: Tensor = torch.tensor(next(self.topology),
                                             dtype=self.x_all[0][0].dtype) * self.topology.lambda_max
        layers_count = len(self.x_all[0])
        oracles_counter = range(len(self.oracles))
        updated_params_by_layer = {"x": [], "x_prev": [], "x_hat": [], "y": [], "w": []}

        # accumulate params and grads by layer
        for layer_num in range(layers_count):
            x: Tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.x_all])
            x_prev: Tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.x_prev])
            x_hat: Tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.x_hat])
            y: Tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.y])

            x_wave = self.params['alpha'] * (x_hat - x_prev) + x
            n, *d = x_wave.shape
            v = torch.matmul(gossip_matrix, x_wave.view(n, -1)).view(n, *d)
            y = y + 1. / self.params['tau'] * v
            n, *d = y.shape
            w = torch.matmul(gossip_matrix, y.view(n, -1)).view(n, *d)
            updated_params_by_layer['x_prev'].append(x)
            updated_params_by_layer['y'].append(y)
            updated_params_by_layer['w'].append(w)

        self.x_prev = self.change_dim_layer_to_oracle(updated_params_by_layer["x_prev"])
        self.y = self.change_dim_layer_to_oracle(updated_params_by_layer["y"])
        x, x_hat = self.communication_sliding(self.x_all,
                                              self.change_dim_layer_to_oracle(updated_params_by_layer["w"]))
        self.x_hat = x_hat
        self.x_all = x
        # set x_new in oracles
        self.update_ergodic(x_hat, k)
        x = self.change_dim_layer_to_oracle(x_hat)
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [x[layer_num][oracle_num] for layer_num in range(layers_count)]
            )

    def run(self, log: bool = False):
        if log:
            self.logs.append(self.log())
        for i in range(self.max_iter):
            self.__calculate_params(k=i + 1)
            self.step(k=i + 1)
            if log:
                self.logs.append(self.log())

    def log(self) -> dict[str, Any]:
        losses = [oracle() for oracle in self.oracles]
        return {
            "loss": sum(losses) / len(losses),
            "losses": losses
        }

    @staticmethod
    def change_dim_layer_to_oracle(x: list[list[Tensor]]):
        """
        changes dimensions from layer x node x param to node x layer x param
        :param x:
        :return:
        """
        return [list(t) for t in zip(*x)]
