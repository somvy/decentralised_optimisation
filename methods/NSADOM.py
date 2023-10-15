import torch
from torch import tensor
from copy import deepcopy
from methods.base import BaseDecentralizedMethod


class NSADOM(BaseDecentralizedMethod):
    def __init__(self, oracles, topology, stepsize, max_iter, **params):
        super().__init__(oracles, topology)
        self.step_size = stepsize
        self.max_iter = max_iter
        self.params = params

        self.x_f_all = [oracle.get_params() for oracle in self.oracles]
        self.y_f_all = deepcopy(self.x_f_all)
        self.y_all = deepcopy(self.x_f_all)
        # initialize z in L orthogonal
        self.z_all = [[torch.zeros_like(param) for param in oracles] for oracles in self.x_f_all]
        self.m_all = deepcopy(self.x_f_all)
        self.z_f_all = deepcopy(self.z_all)

    def __get_grad_in_x_g(self):
        x = [oracle.get_params() for oracle in self.oracles]
        layers_count = len(x[0])
        x_f = self.x_f_all

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [self.params["tau1"] * x[oracle_num][layer_num] + (1 - self.params["tau1"]) * x_f[oracle_num][
                    layer_num]
                 for layer_num in range(layers_count)]
            )
        gradients = [oracle.grad() for oracle in self.oracles]

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [x[layer_num][oracle_num] for layer_num in range(layers_count)]
            )
        return gradients

    def step(self):
        assert self.topology.matrix_type.startswith("gossip"), "works with gossip matrices only yet"
        gossip_matrix: tensor = tensor(next(self.topology), dtype=torch.float)

        x_all = [oracle.get_params() for oracle in self.oracles]
        layers_count = len(x_all[0])
        grad_all = self.__get_grad_in_x_g()
        # total_gradients: list[list[tensor]] = [oracle.grad() for oracle in self.oracles]
        new_params_by_layer = []

        # accumulate params and grads by layer
        for layer_num in range(layers_count):
            x = torch.stack([x_all[layer_num] for oracle_params in x_all])
            y = torch.stack([self.y_all[layer_num] for oracle_params in x_all])
            z = torch.stack([self.z_all[layer_num] for oracle_params in x_all])
            x_f = torch.stack([self.x_f_all[layer_num] for oracle_params in x_all])
            y_f = torch.stack([self.y_f_all[layer_num] for oracle_params in x_all])
            z_f = torch.stack([self.z_f_all[layer_num] for oracle_params in x_all])
            m = torch.stack([x_all[layer_num] for oracle_params in x_all])
            grad = torch.stack([grad_all[layer_num] for oracle_params in x_all])

            x_g = self.params['tau1'] * x + (1 - self.params['tau1']) * x_f
            y_g = self.params['sigma1'] * y + (1 - self.params['sigma1']) * y_f
            z_g = self.params['sigma1'] * z + (1 - self.params['sigma1']) * z_f

            const1 = y - self.params['teta'] * self.params['beta'] * (grad - self.params['nu'] * x_g)
            const1 -= self.params['teta'] / self.params['nu'] * (y_g + z_g)
            const2 = x + self.params['eta'] * self.params['alpha'] * x_g
            const2 -= self.params['eta'] * (grad - self.params['nu'] * x_g)

            x_new = const2 + self.params['eta'] / (1 + self.params['teta'] * self.params['beta']) * const1
            denum = 1 + self.params['eta'] * self.params['alpha']
            denum += self.params['teta'] * self.params['eta'] / (
                    1 + self.params['teta'] * self.params['beta'])
            x_new /= denum
            y_new = const1 - self.params['teta'] * x_new
            y_new /= 1 + self.params['teta'] * self.params['beta']

            step = torch.einsum("nn,nd...->nd...", gossip_matrix,
                                self.params['gamma'] / self.params['nu'] * (y_g + z_g) + m)

            z_new = z + self.params['gamma'] * self.params['delta'] * (z_g - z)
            z_new -= step
            m_new = self.params['gamma'] / self.params['nu'] * (y_g + z_g) + m - step

            x_f = x_g + self.params['tau2'] * (x_new - x)
            y_f = y_f + self.params['sigma2'] * (y_new - y)
            z_f = z_g - self.params['zeta'] * torch.einsum("nn,nd...->nd...", gossip_matrix, y_g + z_g)
        # put x_f, y_f, z_f in self.x_f_all, self.y_f_all, self.z_f_all
        # put z_new,y_new, m_new in self.z_all, self.y_all, self.m_all
        # x_new in oracles

        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [new_params_by_layer[layer_num][oracle_num] for layer_num in range(layers_count)]
            )

    def run(self, log: bool = False):
        for _ in range(self.max_iter):
            self.step()
            if log:
                self.log()

    def log(self):
        losses = [oracle() for oracle in self.oracles]
        print(f"losses: {sum(losses) / len(losses)}| {losses}")
