from collections import defaultdict

from methods.NSADOM import NSADOM
import numpy as np
from decentralized.topologies import Topologies
from oracles.binary_svc_oracle import BinarySVC
import torch
from utils import print_metrics, save_plot

def test_nsadom_smv():
    np.random.seed(0xCAFEBABE)
    dim_size = 20
    num_nodes = 10
    samples_on_each_node = 100

    topology: Topologies = Topologies(n=num_nodes, topology_type="star", matrix_type="gossip-metropolis")
    oracles = []

    for i in range(num_nodes):
        points = torch.randn(samples_on_each_node, dim_size)
        X = torch.cat([points + 1, points - 1])
        y = torch.cat([torch.ones(samples_on_each_node), -torch.ones(samples_on_each_node)])
        oracles.append(BinarySVC(X=X, y=y, alpha=0.2))
    params = defaultdict(lambda: 1)
    method = NSADOM(oracles=oracles, topology=topology, stepsize=1e-7, max_iter=1000, params=params)
    method.run(log=True)
    save_plot(method.logs, "nsadom_svm.svg")

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] < 5e-2)
