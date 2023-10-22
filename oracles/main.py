from oracles.mnist_mlp_oracle import MNISTMLP, MLP
from torchvision.datasets import MNIST
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from oracles.l1_regression_oracle import L1RegressionOracle
from oracles.binary_svc_oracle import BinarySVC
import libsvmdata
from torchvision import transforms
import torch


def get_smv_oracles(n):
    X, y = libsvmdata.fetch_libsvm('a1a')
    X = torch.Tensor(X.todense())
    y = torch.Tensor(y)
    oracles = []
    start = 0
    step = X.shape[0] // n

    for _ in range(n):
        oracle = BinarySVC(X[start:start + step], y[start:start + step])
        oracles.append(oracle)
        start += step
    return oracles


def get_l1_oracles(n):
    X, y = libsvmdata.fetch_libsvm('abalone_scale')
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    oracles = []
    start = 0
    step = X.shape[0] // n

    for _ in range(n):
        oracle = L1RegressionOracle(X[start:start + step], y[start:start + step])
        oracles.append(oracle)
        start += step
    return oracles


def get_mnist_oracles(n):
    train_dataset = MNIST(".", download=True, transform=transforms.ToTensor(), train=True)
    skf = KFold(n_splits=n, shuffle=False)
    init_model = MLP()

    oracles = []
    for train_ids, test_ids in skf.split(train_dataset):
        subset = Subset(train_dataset, test_ids)
        oracle = MNISTMLP(Xy=subset, init_network=init_model)
        oracles.append(oracle)

    return oracles
