import torch
from torch import tensor
import numpy as np
from torch import tensor


def sample_spherical(npoints, ndim, device, dtype):
    vec = np.random.randn(npoints, ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return torch.tensor(vec, device=device, dtype=dtype)


import torch
from oracles.base import BaseOracle


class BinarySVC(BaseOracle):
    def __init__(self, X, y, device="cpu", w_init=None, b_init=None, regularization=1,
                 grad_type="grad", grad_batch_size=55, gamma=1e-4):
        """
        :param X: n x d
        :param y:  -1 , 1, n x 1
        :param alpha:
        :param device:
        :param w_init:
        :param b_init:
        """
        self.d = X.shape[1]
        self.X = X.to(device)
        self.y = y.to(device)
        if w_init is None:
            self.w = torch.ones(self.d, 1, dtype=X.dtype) / (self.d ** 0.5)
        else:
            self.w = w_init
        if b_init is None:
            self.b = torch.ones(1, 1, dtype=X.dtype)
        else:
            self.b = b_init
        for param in [self.w, self.b]:
            param.requires_grad = True
            param = param.to(device)
        self.regularization = regularization
        self.grad_type = grad_type
        self.grad_batch_size = grad_batch_size
        self.gamma = gamma
        self.lipschitz = 1
        self.dimension = 0
        params = self.get_params()
        for param in params:
            self.dimension += np.prod(param.shape)

    def __f(self):
        margin = (1 - self.y * (self.X @ self.w - self.b))
        margin = torch.cat([margin, torch.zeros(self.X.shape[0], 1)], axis=1).max(axis=1)[0]
        return (margin + self.regularization / 2 * (self.w.T @ self.w)).mean()

    def __call__(self):
        with torch.no_grad():
            return self.__f().item()

    def __grad(self):
        self.w.grad = None
        self.b.grad = None
        loss = self.__f()
        loss.backward()
        return [self.w.grad, self.b.grad]

    def get_params(self):
        return [self.w.detach(), self.b.detach()]

    def set_params(self, params):
        self.w = params[0].clone()
        self.b = params[1].clone()
        for _ in [self.w, self.b]:
            _.requires_grad = True
        
    def __randomized_subgrad(self) -> list[tensor]:
        if self.grad_type == "smoothed-two-point":
            params = self.get_params()
            dimension = self.dimension
            noise = sample_spherical(self.grad_batch_size, dimension, params[0].device, params[0].dtype)
            grad = torch.zeros_like(noise[0])
            for i in range(self.grad_batch_size):
                current_noise = noise[i]
                left_params = []
                right_params = []
                start = 0
                for param in params:
                    size = np.prod(param.shape)
                    left_params.append(param - self.gamma * current_noise[start:start + size].reshape(param.shape))
                    right_params.append(param + self.gamma * current_noise[start:start + size].reshape(param.shape))
                    start += size
                self.set_params(left_params)
                f_left = self.__call__()
                self.set_params(right_params)
                f_right = self.__call__()
                grad += dimension / self.gamma / 2 * (f_right - f_left) * current_noise
            grad /= self.grad_batch_size
            result = []
            start = 0
            for param in params:
                size = np.prod(param.shape)
                result.append(grad[start:start + size].reshape(param.shape))
                start += size
            self.set_params(params)
            return result
        
    def grad(self) -> list[tensor]:
        if self.grad_type == "grad":
            return self.__grad()
        if self.grad_type == "sub_grad":
            return self.__subgrad()
        return self.__randomized_subgrad()
