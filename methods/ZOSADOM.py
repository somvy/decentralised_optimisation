from typing import Any

import torch
from torch import Tensor
from copy import deepcopy
import numpy as np
from methods.base import BaseDecentralizedMethod


class ZOSADOM(BaseDecentralizedMethod):
    def __init__(self, oracles, topology, max_iter):
        super().__init__(oracles, topology)
        self.max_iter = max_iter

        self.x_f_all: list[list[Tensor]] = [oracle.get_params() for oracle in self.oracles]
        # initialize z in L orthogonal
        # oracles x layers x params
        self.z_all = [[torch.zeros_like(param) for param in oracles] for oracles in self.x_f_all]
        self.y_f_all = deepcopy(self.x_f_all)
        self.y_all = deepcopy(self.x_f_all)
        self.z_f_all = deepcopy(self.z_all)
        self.m_all = deepcopy(self.x_f_all)
        self.__calculate_params()
    
    def __calculate_params(self):
        L = np.sqrt(self.oracles[0].dimension) * self.oracles[0].lipschitz / self.oracles[0].gamma
        mu = self.oracles[0].regularization
        chi = self.topology.chi
        self.params = {}
        self.params['zeta'] = 0.5
        self.params['beta'] = 1. / (2 * L)
        self.params['nu'] = mu / 2
        self.params['alpha'] = mu / 4
        self.params['tau2'] = np.sqrt(mu / L)
        self.params['pi'] = self.params['beta'] / 16
        self.params['tau1'] = 1. / (1. / self.params['tau2'] + 0.5)
        self.params['eta'] = 1. / ((1. / self.params['beta'] + L) * self.params['tau2'])
        self.params['sigma2'] = np.sqrt(self.params['beta'] * mu) / (16 * chi)
        self.params['sigma1'] = 1. / (1 / self.params['sigma2'] + 0.5)
        self.params['kappa'] = self.params['nu'] / (14 * self.params['sigma2'] * chi * chi)
        self.params['teta'] = self.params['nu'] / (4 * self.params['sigma2'])
        

    def __get_grad_in_x_g(self) -> list[list[Tensor]]:
        x = [oracle.get_params() for oracle in self.oracles] # node x layer x param
        layers_count = len(x[0])
        x_f = self.x_f_all
        # calculate gradients in x_g
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [self.params["tau1"] * x[oracle_num][layer_num] + (1 - self.params["tau1"]) * x_f[oracle_num][
                    layer_num].reshape(x[oracle_num][layer_num].shape)
                 for layer_num in range(layers_count)]
            )
        gradients = [oracle.grad() for oracle in self.oracles]
        # set params back
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                x[oracle_num]
            )
        return gradients

    def step(self):
        assert self.topology.matrix_type.startswith("gossip"), "works with gossip matrices only yet"

        x_all: list[list[Tensor]] = [oracle.get_params() for oracle in
                                     self.oracles]  # node x layer x param
        gossip_matrix: Tensor = torch.tensor(next(self.topology), dtype=x_all[0][0].dtype)
        layers_count = len(x_all[0])
        oracles_counter = range(len(self.oracles))
        grad_all = self.__get_grad_in_x_g()  # node x layer x param
        updated_params_by_layer = {"x": [], "y": [], "z": [], "m": [], "x_f": [], "y_f": [], "z_f": []}

        # accumulate params and grads by layer
        for layer_num in range(layers_count):
            x = torch.stack([x_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            y = torch.stack([self.y_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            z = torch.stack([self.z_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            x_f = torch.stack([self.x_f_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            y_f = torch.stack([self.y_f_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            z_f = torch.stack([self.z_f_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            m = torch.stack([x_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            grad = torch.stack([grad_all[oracle_num][layer_num] for oracle_num in oracles_counter]).squeeze()
            x_g = self.params['tau1'] * x + (1 - self.params['tau1']) * x_f
            y_g = self.params['sigma1'] * y + (1 - self.params['sigma1']) * y_f
            z_g = self.params['sigma1'] * z + (1 - self.params['sigma1']) * z_f

            const1 = y + self.params['teta'] * self.params['beta'] * (grad - self.params['nu'] * x_g)
            const1 -= self.params['teta'] / self.params['nu'] * (y_g + z_g)
            const2 = x + self.params['eta'] * self.params['alpha'] * x_g
            const2 -= self.params['eta'] * (grad - self.params['nu'] * x_g)

            x_new = const2 + self.params['eta'] / (1 + self.params['teta'] * self.params['beta']) * const1
            denum = 1 + self.params['eta'] * self.params['alpha']
            denum += self.params['teta'] * self.params['eta'] / (1 + self.params['teta'] * self.params['beta'])
            x_new /= denum
            y_new = const1 - self.params['teta'] * x_new
            y_new /= 1 + self.params['teta'] * self.params['beta']
            
            step = gossip_matrix @ (self.params['kappa'] / self.params['nu'] * (y_g + z_g) + m)

            z_new = z + self.params['kappa'] * self.params['pi'] * (z_g - z)
            z_new -= step
            m_new = self.params['kappa'] / self.params['nu'] * (y_g + z_g) + m - step

            x_f = x_g + self.params['tau2'] * (x_new - x)
            y_f = y_g + self.params['sigma2'] * (y_new - y)
            z_f = z_g - self.params['zeta'] * (gossip_matrix @ y_g + z_g)

            updated_params_by_layer["x"].append(x_new)
            updated_params_by_layer["y"].append(y_new)
            updated_params_by_layer["z"].append(z_new)
            updated_params_by_layer["m"].append(m_new)
            updated_params_by_layer["x_f"].append(x_f)
            updated_params_by_layer["y_f"].append(y_f)
            updated_params_by_layer["z_f"].append(z_f)

        self.x_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["x_f"])

        self.y_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["y_f"])
        self.z_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["z_f"])
        self.z_all = self.change_dim_layer_to_oracle(updated_params_by_layer["z"])
        self.y_all = self.change_dim_layer_to_oracle(updated_params_by_layer["y"])
        self.m_all = self.change_dim_layer_to_oracle(updated_params_by_layer["m"])
        
        # set x_new in oracles
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [updated_params_by_layer["x"][layer_num][oracle_num].reshape(x_all[oracle_num][layer_num].shape)
                         for layer_num in range(layers_count)]
            )

    def run(self, log: bool = False):
        if log:
            self.logs.append(self.log())
        for _ in range(self.max_iter):
            self.step()
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
