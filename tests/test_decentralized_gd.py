import numpy as np
from decentralized.topologies import Topologies
from oracles.simple import QuadraticOracle
from oracles.binary_svc_oracle import BinarySVC
from oracles.l1_regression_oracle import L1RegressionOracle
from methods.decentralized_gd import DecentralizedGradientDescent
import torch


def test_decentralized_gd_simple_task():
    np.random.seed(0xCAFEBABE)
    dim_size = 20
    num_nodes = 10

    topology: Topologies = Topologies(n=num_nodes, topology_type="ring", matrix_type="gossip-metropolis")
    x_0 = torch.rand(dim_size)
    oracles = [
        QuadraticOracle(
            A=(num_node + 1) / num_nodes * torch.eye(dim_size),
            b=torch.zeros(dim_size),
            x=x_0)
        for num_node in range(num_nodes)]

    method = DecentralizedGradientDescent(oracles=oracles, topology=topology, stepsize=1e-5,
                                          max_iter=1000)

    method.run()

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] < 5e-2)


def test_decentralizes_gd_svm():
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
    method.run()

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] < 5e-2)


def test_decentralized_gd_l1regression():
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
        oracles.append(L1RegressionOracle(X=X, y=y))

    method = DecentralizedGradientDescent(oracles=oracles, topology=topology, stepsize=1e-7, max_iter=10000)
    method.run()

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] < 5e-2)



