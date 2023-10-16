import torch
from torch import Tensor
from copy import deepcopy
from methods.base import BaseDecentralizedMethod


class NSADOM(BaseDecentralizedMethod):
    def __init__(self, oracles, topology, stepsize, max_iter, params):
        super().__init__(oracles, topology)
        self.step_size = stepsize
        self.max_iter = max_iter
        self.params = params

        self.x_f_all: list[list[Tensor]] = [oracle.get_params() for oracle in self.oracles]
        self.y_f_all = deepcopy(self.x_f_all)
        self.y_all = deepcopy(self.x_f_all)
        # initialize z in L orthogonal
        # oracles x layers x params
        self.z_all = [[torch.zeros_like(param) for param in oracles] for oracles in self.x_f_all]
        self.z_f_all = deepcopy(self.z_all)
        self.m_all = deepcopy(self.x_f_all)

    def __get_grad_in_x_g(self) -> list[list[Tensor]]:
        x = [oracle.get_params() for oracle in self.oracles] # node x layer x param
        layers_count = len(x[0])
        x_f = self.x_f_all
        # calculate gradients in x_g
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [self.params["tau1"] * x[oracle_num][layer_num] + (1 - self.params["tau1"]) * x_f[oracle_num][
                    layer_num]
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
        gossip_matrix: Tensor = Tensor(next(self.topology))

        x_all: list[list[Tensor]] = [oracle.get_params() for oracle in
                                     self.oracles]  # node x layer x param
        layers_count = len(x_all[0])
        oracles_counter = range(len(self.oracles))
        grad_all = self.__get_grad_in_x_g()  # node x layer x param
        # total_gradients: list[list[tensor]] = [oracle.grad() for oracle in self.oracles]
        updated_params_by_layer = {"x": [], "y": [], "z": [], "m": [], "x_f": [], "y_f": [], "z_f": []}

        # accumulate params and grads by layer
        for layer_num in range(layers_count):
            x = torch.stack([x_all[oracle_num][layer_num] for oracle_num in oracles_counter])
            y = torch.stack([self.y_all[oracle_num][layer_num] for oracle_num in oracles_counter])
            z = torch.stack([self.z_all[oracle_num][layer_num] for oracle_num in oracles_counter])
            x_f = torch.stack([self.x_f_all[oracle_num][layer_num] for oracle_num in oracles_counter])
            y_f = torch.stack([self.y_f_all[oracle_num][layer_num] for oracle_num in oracles_counter])
            z_f = torch.stack([self.z_f_all[oracle_num][layer_num] for oracle_num in oracles_counter])
            m = torch.stack([x_all[oracle_num][layer_num] for oracle_num in oracles_counter])
            grad = torch.stack([grad_all[oracle_num][layer_num] for oracle_num in oracles_counter])

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

            updated_params_by_layer["x"].append(x_new)
            updated_params_by_layer["y"].append(y_new)
            updated_params_by_layer["z"].append(z_new)
            updated_params_by_layer["m"].append(m_new)
            updated_params_by_layer["x_f"].append(x_f)
            updated_params_by_layer["y_f"].append(y_f)
            updated_params_by_layer["z_f"].append(z_f)

        # put x_f, y_f, z_f in self.x_f_all, self.y_f_all, self.z_f_all
        # put z_new,y_new, m_new in self.z_all, self.y_all, self.m_all
        print(len(self.x_f_all), len(self.x_f_all[0]))
        self.x_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["x_f"])
        print(len(self.x_f_all), len(self.x_f_all[0]))

        self.y_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["y_f"])
        self.z_f_all = self.change_dim_layer_to_oracle(updated_params_by_layer["z_f"])
        self.z_all = self.change_dim_layer_to_oracle(updated_params_by_layer["z"])
        self.y_all = self.change_dim_layer_to_oracle(updated_params_by_layer["y"])
        self.m_all = self.change_dim_layer_to_oracle(updated_params_by_layer["m"])

        # set x_new in oracles
        for oracle_num, oracle in enumerate(self.oracles):
            oracle.set_params(
                [updated_params_by_layer["x"][layer_num][oracle_num] for layer_num in range(layers_count)]
            )

    def run(self, log: bool = False):
        for _ in range(self.max_iter):
            self.step()
            if log:
                self.log()

    def log(self):
        losses = [oracle() for oracle in self.oracles]
        print(f"losses: {sum(losses) / len(losses)}| {losses}")

    @staticmethod
    def change_dim_layer_to_oracle(x: list[list[Tensor]]):
        """
        changes dimensions from layer x node x param to node x layer x param
        :param x:
        :return:
        """
        return [list(t) for t in zip(*x)]
