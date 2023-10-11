import torch
from oracles.base import BaseOracle


class BinarySVC(BaseOracle):
  def __init__(self, X, y, alpha: float, device="cpu", w_init=None, b_init=None):
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
      self.w = torch.ones(self.d, 1) / (self.d ** 0.5)
    else:
      self.w = w_init
    if b_init is None:
      self.b = torch.ones(1, 1)
    else:
      self.b = b_init

    for param in [self.w, self.b]:
      param.requires_grad = True
      param = param.to(device)

    self.alpha = alpha

  def __f(self):
    margin = (1 - self.y * (self.X @ self.w - self.b))
    margin = torch.cat([margin, torch.zeros(self.X.shape[0], 1)], axis=1).max(axis=1)[0]
    return (margin + self.alpha / 2 * (self.w.T @ self.w)).mean()

  def __call__(self):
    with torch.no_grad():
      return self.__f().item()

  def grad(self):
    self.w.grad = None
    self.b.grad = None
    loss = self.__f()
    loss.backward()
    return [self.w.grad, self.b.grad]

  def get_params(self):
    return [self.w.detach(), self.b.detach()]

  def set_params(self, params):
    self.w = params[0]
    self.b = params[1]
    for _ in [self.w, self.b]:
      _.requires_grad = True
