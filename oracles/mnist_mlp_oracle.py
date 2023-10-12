import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      # input shape = 28*28
      # neurons in first dense layer = 64
      nn.Linear(28*28, 64),
      # relu activation
      nn.ReLU(),
      # 64 = neurons in first dense layer
      # 32 = neurons in second dense layer
      nn.Linear(64, 32),
      nn.ReLU(),
      # 32 = neurons in second dense layer
      # 10 = neurons in output layer (number of classes)
      nn.Linear(32, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

class MNISTMLP:

  def __init__(self, Xy: Dataset, init_network: nn.Module, batch_size=128, device="cpu"):
    self.data = DataLoader(Xy, shuffle=False, batch_size=batch_size)
    self.bs = batch_size
    self.net = MLP()
    self.net.load_state_dict(deepcopy(init_network.state_dict()))
    for p in self.net.parameters():
      p.require_grad = True
    self.net = self.net.to(device)
    self.loss = nn.CrossEntropyLoss()
    self.device = device
  
  def __f(self):
    loss = 0
    n_samples = 0
    for (X, y) in tqdm(self.data):
      X, y = X.to(self.device), y.to(self.device)
      n_samples += X.shape[0]
      y_pred = self.net(X)
      loss += self.loss(y_pred, y)
    loss /= n_samples
    return loss

  def __call__(self):
    with torch.no_grad():
      return self.__f()
    
  def grad(self):
    for p in self.net.parameters():
      p.grad = None
    loss = self.__f()
    loss.backward()
    return [p.grad for p in self.net.parameters()]

  def get_params(self):
    return [p for p in self.net.parameters()]

  def set_params(self, params):
    for (p_old, p_new) in zip(self.net.parameters(), params):
      p_old = p_new
      p_old.requires_grad = True
