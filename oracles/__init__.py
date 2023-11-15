from oracles.base import BaseOracle
from oracles.binary_svc_oracle import BinarySVC
from oracles.l1_regression_oracle import L1RegressionOracle
from oracles.mnist_mlp_oracle import MNISTMLP, MLP
from oracles.resnet_cifar_oracle import CifarResnet
from oracles.simple import QuadraticOracle
from oracles.main import get_svm_oracles, get_l1_oracles, get_mnist_oracles
