import torch
import numpy as np
from torch import tensor


def sample_spherical(npoints, ndim, device, dtype):
    vec = np.random.randn(npoints, ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return torch.tensor(vec, device=device, dtype=dtype)


def is_smoothing_needed(params, eps=1e-5):
    for param in params:
        if (torch.abs(param) < eps).any():
            return True
    return False


class L1RegressionOracle:
    
    def __init__(self, X, y, device='cpu', W_init=None, b_init=None, regularization=1,
                 grad_type="grad", grad_batch_size=55, gamma=1e-4):
        self.grad_type = grad_type
        self.grad_batch_size = grad_batch_size
        self.gamma = gamma
        self.X = X
        self.y = y
        if W_init is None:
            W_init = torch.ones(X.shape[1], 1, dtype=X.dtype, requires_grad=True) / np.sqrt(X.shape[1])
        elif type(W_init) != torch.tensor:
            W_init = torch.tensor(W_init)
        if b_init is None:
            b_init = torch.ones(1, 1, dtype=X.dtype, requires_grad=True)
        elif type(b_init) != torch.tensor:
            b_init = torch.tensor(b_init)
            
        self.W = W_init.clone().detach().requires_grad_(True).to(device)
        self.b = b_init.clone().detach().requires_grad_(True).to(device)
        self.loss = torch.nn.L1Loss(reduction="mean")
        self.lipschitz = torch.linalg.norm(torch.concat([X, torch.ones(X.shape[1])[None, :]], dim=0), ord=2)
        self.regularization = regularization
        self.dimension = 0
        params = self.get_params()
        for param in params:
            self.dimension += np.prod(param.shape)
    
    @torch.no_grad()
    def __call__(self):
        return self.__f().item()
    
    def __f(self):
        preds = self.X @ self.W + self.b
        return self.loss(preds[:, 0], self.y) + self.regularization / 2 * (self.W * self.W).sum()
    
    def __grad(self):
        if self.W.grad is not None and self.b.grad is not None:
            self.W.grad.data.zero_()
            self.b.grad.data.zero_()
        loss = self.__f()
        loss.backward()
        return [self.W.grad, self.b.grad]
    
    def get_params(self):
        return [self.W.detach(), self.b.detach()]
    
    def set_params(self, params):
        self.W.data = params[0]
        self.b.data = params[1]
        
    def __randomized_subgrad(self) -> list[tensor]:
        if self.grad_type == "smoothed-two-point":
            params = self.get_params()
            if is_smoothing_needed(params):
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
            else:
                return self.__grad()
        
    def grad(self) -> list[tensor]:
        if self.grad_type == "grad":
            return self.__grad()
        if self.grad_type == "sub_grad":
            return self.__subgrad()
        return self.__randomized_subgrad()
