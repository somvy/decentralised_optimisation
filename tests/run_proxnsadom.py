from collections import defaultdict
from typing import Any

from methods.proxnsadom import PROXNSADOM
import numpy as np
from decentralized.topologies import Topologies
from oracles.l1_regression_oracle import L1RegressionOracle
import torch
from utils import print_metrics, save_plot
import libsvmdata
import matplotlib.pyplot as plt


def get_oracles(X, y, n):
    oracles = []
    start = 0
    step = X.shape[0] // n
    for _ in range(n):
        oracles.append(L1RegressionOracle(X[start:start + step], y[start:start + step]))
        start += step
    return oracles


def proxnsadom_plot(logs: dict[str, float]):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot([log["loss"] for log in logs])
    axs[0, 0].set_title("loss")
    axs[1, 0].plot([log["theta"] for log in logs])
    axs[1, 0].set_title("theta")
    # axs[1, 0].sharex(axs[0, 0]
    axs[0, 1].plot([log["eta"] for log in logs])
    axs[0, 1].set_title("eta")
    axs[1, 1].plot([log["a"] for log in logs])
    axs[1, 1].set_title("a")
    # fig.tight_layout()
    # plt.xlabel("Communication rounds")
    # plt.ylabel("Objective value")
    # plt.title(f"Convergence on {task_name} and {topology_name} topology")
    plt.legend()
    fig.savefig("proxnsadom_hyper.svg")


def run():
    np.random.seed(0xCAFEBABE)
    num_nodes = 10

    topology: Topologies = Topologies(n=num_nodes, topology_type="ring", matrix_type="gossip-metropolis")

    X, y = libsvmdata.fetch_libsvm('abalone_scale')
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    oracles = get_oracles(X, y, num_nodes)

    reg = 1
    chi = topology.chi
    # chi = 1.5
    print("topology chi :", chi)
    eps = 1e-5

    method = PROXNSADOM(
        oracles=oracles,
        topology=topology,
        max_iter=10,
        eta=1 / (64 * reg * chi ** 2),
        theta=reg / (16 * chi ** 2),
        gamma=reg / 2,
        r=reg
    )
    print("running...")
    method.run(log=True, disable_tqdm=False)

    save_plot(method.logs, "proxnsadom_svm.svg")
    proxnsadom_plot(method.logs)
    print("\n".join([str(log) for log in method.logs]))

    # print_metrics(method.logs)
    print("logs saved to %s" % "proxnsadom_svm.svg")


if __name__ == "__main__":
    run()
