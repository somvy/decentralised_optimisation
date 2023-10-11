import numpy as np
from decentralized.topologies import Topologies
from oracles.simple import QuadraticOracle
from methods.decentralized_gd import DecentralizedGradientDescent


def test_decentralized_gd():
    np.random.seed(0xCAFEBABE)
    dim_size = 20
    num_nodes = 10

    topology: Topologies = Topologies(n=num_nodes, topology_type="ring", matrix_type="gossip-metropolis", n_graphs=5)
    x_0 = np.random.rand(dim_size)
    X_0 = np.vstack([x_0] * num_nodes)
    oracles = [QuadraticOracle(A=(num_node / num_nodes) * np.eye(dim_size), b=np.zeros(dim_size))
               for num_node in range(1, num_nodes + 1)]

    method = DecentralizedGradientDescent(oracles=oracles, topology=topology, x_0=X_0, stepsize=1e-7,
                                          max_iter=1000)

    method.run()
    print(next(topology))
    print("chi:", topology.chi, "\n")
    print(method.x)
    print(method.x[0] - method.x[1])
    assert np.all((method.x ** 2).sum(axis=1) <= 0.05)




