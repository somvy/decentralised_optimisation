import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

class CifarResnet:

  def __init__(self, Xy: Dataset, batch_size=128, device="cpu"):
    self.data = DataLoader(Xy, shuffle=False, batch_size=batch_size)
    self.iter_data = iter(self.data)
    self.bs = batch_size
    self.net = resnet18(pretrained=True)
    for p in self.net.parameters():
      p.require_grad = True
    self.net = self.net.to(device)
    self.loss = nn.CrossEntropyLoss()
    self.device = device

  def __f(self):
    try:
      X, y = next(self.iter_data)
    except:
      self.iter_data = iter(self.data)
      X, y = next(self.iter_data)
    X, y = X.to(self.device), y.to(self.device)
    y_pred = self.net(X)
    return self.loss(y_pred, y)

  def __call__(self):
    with torch.no_grad():
      return self.__f()

  def grad(self):
    for p in self.net.parameters():
      p.grad = None
    loss = self.__f()
    loss.backward()
    grads = [p.grad.clone() for p in self.net.parameters()]
    for p in self.net.parameters():
      p.grad = None
    return grads

  def get_params(self):
    return [p for p in self.net.parameters()]

  def set_params(self, params):
    for (p_old, p_new) in zip(self.net.parameters(), params):
      p_old.data = p_new.clone().detach()
      p_old.requires_grad = True
