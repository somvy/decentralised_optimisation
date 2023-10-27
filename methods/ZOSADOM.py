from typing import Any

import torch
from torch import Tensor
from copy import deepcopy
import numpy as np
from methods.base import BaseDecentralizedMethod
from tqdm.auto import trange


class ZOSADOM(BaseDecentralizedMethod):
    def __init__(self, oracles, topology, max_iter):
        super().__init__(oracles, topology)
        self.max_iter = max_iter

        self.x_all: list[list[Tensor]] = [oracle.get_params() for oracle in self.oracles]
        # initialize z in L orthogonal
        # oracles x layers x params
        self.z_all = [[torch.zeros_like(tmp) for tmp in el] for el in self.x_all]
        self.y_all = deepcopy(self.x_all)
        self.m_all = deepcopy(self.z_all)
        self.x_f_all = deepcopy(self.x_all)
        self.y_f_all = deepcopy(self.x_all)
        self.z_f_all = deepcopy(self.z_all)
        self.__calculate_params()
    
    def __calculate_params(self):
        L = np.sqrt(self.oracles[0].dimension) * self.oracles[0].lipschitz / self.oracles[0].gamma
        mu = self.oracles[0].regularization
        chi = self.topology.chi
        #print(chi)
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
    
    def compute_grad_in_x_g(self):
        x = self.x_all # node x layer x param
        x_f = self.x_f_all
        layers_count = len(x[0])
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [self.params['tau1'] * x[oracle_num][layer_num] + 
                 (1 - self.params['tau1']) * x_f[oracle_num][layer_num]
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
        gossip_matrix: Tensor = torch.tensor(next(self.topology), dtype=self.x_all[0][0].dtype)
        layers_count = len(self.x_all[0])
        oracles_counter = range(len(self.oracles))
        updated_params_by_layer = {"x": [], "y": [], "z": [], "m": [], "x_f": [], "y_f": [], "z_f": []}
        grad_all = self.compute_grad_in_x_g()
        # accumulate params and grads by layer
        for layer_num in range(layers_count):
            x: tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.x_all])
            y: tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.y_all])
            z: tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.z_all])
            x_f: tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.x_f_all])
            y_f: tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.y_f_all])
            z_f: tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.z_f_all])
            m: tensor = torch.stack([oracle_params[layer_num] for oracle_params in self.m_all])
            grad: tensor = torch.stack([oracle_params[layer_num] for oracle_params in grad_all])
            x_g = self.params['tau1'] * x + (1 - self.params['tau1']) * x_f
            y_g = self.params['sigma1'] * y + (1 - self.params['sigma1']) * y_f
            z_g = self.params['sigma1'] * z + (1 - self.params['sigma1']) * z_f
            # grad = torch.zeros_like(grad)
            
            x_new = grad - self.params['nu'] * x_g - y + self.params['teta'] / self.params['nu'] * (y_g + z_g)
            x_new *= self.params['eta'] / (1 + self.params['teta'] * self.params['beta'])
            x_new = x + self.params['eta'] * self.params['alpha'] * x_g - x_new
            denum = 1 + self.params['eta'] * self.params['alpha']
            denum += self.params['eta'] * self.params['teta'] / (1 + self.params['teta'] * self.params['beta'])
            x_new /= denum
            
            y_new = self.params['teta'] * ((y_g + z_g) / self.params['nu'] + x)
            y_new = y + self.params['teta'] * self.params['beta'] * (grad - self.params['nu'] * x_g) - y_new
            y_new /= 1 + self.params['teta'] * self.params['beta']
            
            x_f_new = x_g + self.params['tau2'] * (x_new - x)
            y_f_new = y_g + self.params['sigma2'] * (y_new - y)
            delta_m = self.params['kappa'] * (y_g + z_g) / self.params['nu'] + m
            delta_z = y_g + z_g
            
            n, *d = delta_m.shape
            delta_m = torch.matmul(gossip_matrix, delta_m.view(n, -1)).view(n, *d)
    
            n, *d = delta_z.shape
            delta_z = torch.matmul(gossip_matrix, delta_z.view(n, -1)).view(n, *d)
            
            z_new = z + self.params['kappa'] * self.params['pi'] * (z_g - z) - delta_m
            m_new = m + self.params['kappa'] / self.params['nu'] * (y_g + z_g) - delta_m
            
            z_f_new = z_g - self.params['zeta'] * delta_z
            updated_params_by_layer['x'].append(x_new)
            updated_params_by_layer['y'].append(y_new)
            updated_params_by_layer['z'].append(z_new)
            updated_params_by_layer['m'].append(m_new)
            updated_params_by_layer['x_f'].append(x_f_new)
            updated_params_by_layer['y_f'].append(y_f_new)
            updated_params_by_layer['z_f'].append(z_f_new)
            

        self.x_all = self.change_dim_layer_to_oracle(updated_params_by_layer["x"])
        self.y_all = self.change_dim_layer_to_oracle(updated_params_by_layer["y"])
        self.z_all = self.change_dim_layer_to_oracle(updated_params_by_layer["z"])
        self.m_all = self.change_dim_layer_to_oracle(updated_params_by_layer["m"])
        self.x_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["x_f"])
        self.y_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["y_f"])
        self.z_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["z_f"])
        # set x_new in oracles
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [self.x_all[oracle_num][layer_num] for layer_num in range(layers_count)]
            )

    def run(self, log: bool = False, disable_tqdm=True):
        loop = trange(1, self.max_iter + 1, disable=disable_tqdm)
        if log:
            self.logs.append(self.log())
        for i in loop:
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
