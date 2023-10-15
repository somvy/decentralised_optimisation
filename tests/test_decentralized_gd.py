import numpy as np
from decentralized.topologies import Topologies
from oracles.simple import QuadraticOracle
from oracles.binary_svc_oracle import BinarySVC
from oracles.l1_regression_oracle import L1RegressionOracle
from methods.decentralized_gd import DecentralizedGradientDescent
import torch
from utils import print_metrics, save_plot


def test_decentralized_gd_simple_task():
    np.random.seed(0xCAFEBABE)
    dim_size = 20
    num_nodes = 10

    topology: Topologies = Topologies(n=num_nodes, topology_type="full", matrix_type="gossip-metropolis")
    x_0 = torch.rand(dim_size, 1)
    oracles = [
        QuadraticOracle(
            A=(num_node + 1) / num_nodes * torch.eye(dim_size),
            b=torch.zeros(dim_size, 1),
            x=x_0)
        for num_node in range(num_nodes)]

    method = DecentralizedGradientDescent(oracles=oracles, topology=topology, stepsize=1e-5,
                                          max_iter=100)

    method.run(log=True)
    save_plot(method.logs, "dgd_simple.png")

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] <= 5e-2)


def test_decentralized_gd_svm():
    np.random.seed(0xCAFEBABE)
    dim_size = 20
    num_nodes = 10
    samples_on_each_node = 100

    topology: Topologies = Topologies(n=num_nodes, topology_type="star", matrix_type="gossip-metropolis")
    oracles = []

    for i in range(num_nodes):
        points = torch.randn(samples_on_each_node, dim_size)
        X = torch.cat([points + 3, points - 3])
        y = torch.cat([torch.ones(samples_on_each_node), -torch.ones(samples_on_each_node)])
        oracles.append(BinarySVC(X=X, y=y, alpha=0.2))

    method = DecentralizedGradientDescent(oracles=oracles, topology=topology, stepsize=1e-7,
                                          max_iter=1000)
    method.run(log=True)
    save_plot(method.logs, "dgd_svm.png")

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] < 5e-2)


def test_decentralized_gd_l1regression():
    np.random.seed(0xCAFEBABE)
    dim_size = 20
    num_nodes = 10
    samples_on_each_node = 100

    topology: Topologies = Topologies(n=num_nodes, topology_type="star", matrix_type="gossip-metropolis")
    oracles = []
    w_init = torch.randn(dim_size, 1)

    for i in range(num_nodes):
        points = torch.randn(samples_on_each_node, dim_size)
        X = torch.cat([points + 2, points - 2])
        y = torch.cat([torch.ones(samples_on_each_node), -torch.ones(samples_on_each_node)])
        oracles.append(L1RegressionOracle(X=X, y=y, W_init=w_init))

    print_metrics([oracle.metrics() for oracle in oracles])

    method = DecentralizedGradientDescent(oracles=oracles, topology=topology, stepsize=1e-7, max_iter=1000)

    method.run(log=True)
    save_plot(method.logs, "dgd_l1reg.png")

    print_metrics([oracle.metrics() for oracle in oracles])

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] < 5e-2)



