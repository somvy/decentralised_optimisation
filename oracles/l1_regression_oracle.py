import torch

class L1RegressionOracle:
    
    def __init__(self, X, y, device='cpu', W_init=None, b_init=None):
        self.X = X
        self.y = y
        if W_init is None:
            W_init = torch.zeros(X.shape[1], 1, dtype=X.dtype)
        else:
            W_init = torch.tensor(W_init, dtype=X.dtype)
        if b_init is None:
            b_init = torch.zeros(1, 1, dtype=X.dtype)
        else:
            b_init = torch.tensor(b_init, dtype=X.dtype)
        
        self.W = torch.autograd.Variable(W_init, requires_grad=True).to(device)
        self.b = torch.autograd.Variable(b_init, requires_grad=True).to(device)
        self.loss = torch.nn.L1Loss(reduction="mean")
    
    @torch.no_grad()
    def __call__(self):
        preds = self.X @ self.W + self.b
        return self.loss(preds[:, 0], self.y)
    
    def __f(self):
        preds = self.X @ self.W + self.b
        return self.loss(preds[:, 0], self.y)
    
    def grad(self):
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
