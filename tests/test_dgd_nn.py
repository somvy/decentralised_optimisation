import os
import sys

sys.path.append(os.getcwd())

from decentralized.topologies import Topologies
from methods.decentralized_gd import DecentralizedGradientDescent
from oracles.mnist_mlp_oracle import MNISTMLP, MLP
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
from sklearn.model_selection import KFold


def test_decentralized_gd_nn():
    num_nodes = 3
    topology: Topologies = Topologies(n=num_nodes, topology_type="full", matrix_type="gossip-metropolis")

    init_model = MLP()
    train_dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
    skf = KFold(n_splits=num_nodes, shuffle=False)

    oracles = []

    for train_ids, test_ids in skf.split(train_dataset):
        subset = Subset(train_dataset, test_ids)
        oracle = MNISTMLP(Xy=subset, init_network=init_model)
        oracles.append(oracle)

    method = DecentralizedGradientDescent(oracles=oracles, topology=topology, stepsize=1e-3, max_iter=5000)
    loss = sum([oracle() for oracle in oracles]) / len(oracles)
    method.run()
    loss_after = sum([oracle() for oracle in oracles]) / len(oracles)
    print(f"losses: {loss_after} | {loss}")
    assert loss_after < loss


if __name__ == '__main__':
    test_decentralized_gd_nn()
