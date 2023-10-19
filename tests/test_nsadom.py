from collections import defaultdict

import numpy as np
from decentralized.topologies import Topologies
from oracles.l1_regression_oracle import L1RegressionOracle
import torch
from utils import print_metrics, save_plot
import libsvmdata




def test_nsadom_smv():
    np.random.seed(0xCAFEBABE)
    num_nodes = 10

    topology: Topologies = Topologies(n=num_nodes, topology_type="star", matrix_type="gossip-metropolis")
    oracles = []

    X, y = libsvmdata.fetch_libsvm('abalone_scale')

    method = NSADOM(oracles=oracles, topology=topology, stepsize=1e-7, max_iter=1000, params=params)
    method.run(log=True)
    save_plot(method.logs, "nsadom_svm.svg")

    for oracle in oracles:
        assert torch.all(oracle.get_params()[0] < 5e-2)
