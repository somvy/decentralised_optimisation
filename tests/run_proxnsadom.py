from collections import defaultdict
from typing import Any

from methods.proxnsadom import PROXNSADOM
import numpy as np
from decentralized.topologies import Topologies
from oracles.l1_regression_oracle import L1RegressionOracle
from oracles.binary_svc_oracle import BinarySVC
import torch
from utils import print_metrics, save_plot
import libsvmdata
import matplotlib.pyplot as plt


def get_oracles(X, y, n):
    oracles = []
    start = 0
    step = X.shape[0] // n
    dim = X.shape[1]
    # w_init = torch.zeros(dim, 1, dtype=X.dtype) / (dim ** 0.5)
    # b_init = torch.zeros(1, 1, dtype=X.dtype)

    for _ in range(n):
        # oracle = L1RegressionOracle(X[start:start + step], y[start:start + step])
        oracle = BinarySVC(X[start:start + step], y[start:start + step],
                           # w_init=w_init, b_init=b_init
                           )
        oracles.append(oracle)
        start += step
    return oracles


def proxnsadom_plot(logs: list[dict[str, float]]):
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


def get_method(oracles, topology, reg, chi, saddle_lr):
    return PROXNSADOM(
        oracles=oracles,
        topology=topology,
        max_iter=50,
        eta=1 / (64 * reg * chi ** 2),
        theta=reg / (16 * chi ** 2),
        gamma=reg / 2,
        r=reg,
        saddle_lr=saddle_lr
    )


def grid_search(oracles, topology, reg):
    chi = topology.chi
    stats = []
    saddle_lr_candidates = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5,
                            3e-6, 5e-5, 7e-5, 1e-4, 3e-4,
                            5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2,
                            3e-2, 5e-2, 7e-2, 1e-1]

    for saddle_lr in saddle_lr_candidates:
        method = get_method(oracles, topology, reg, chi, saddle_lr)
        print("running... for saddle lr", saddle_lr)
        try:
            method.run(log=True, disable_tqdm=False)
            loss = method.log()["loss"]
            print("loss", loss)
            stats.append(loss)
        except StopIteration:
            pass

    best_saddle_lr = saddle_lr_candidates[torch.argmin(torch.FloatTensor(stats))]
    print(stats)
    print("### best saddle lr:", best_saddle_lr)


def run_single(oracles, topology, reg):
    chi = topology.chi
    print("topology chi :", chi)
    method = get_method(oracles, topology, reg, chi, saddle_lr=5e-7)
    print("running...")
    method.run(log=True, disable_tqdm=False)

    filename = "proxnsadom_svm.svg"
    save_plot(method.logs, filename)
    proxnsadom_plot(method.logs)

    # print_metrics(method.logs)
    print("logs saved to %s" % filename)


def main():
    np.random.seed(0xCAFEBABE)
    num_nodes = 10
    topology: Topologies = Topologies(n=num_nodes, topology_type="ring", matrix_type="gossip-laplacian")

    # X, y = libsvmdata.fetch_libsvm('abalone_scale')
    X, y = libsvmdata.fetch_libsvm('a1a')
    X = torch.Tensor(X.todense())
    y = torch.Tensor(y)
    oracles = get_oracles(X, y, num_nodes)
    run_single(oracles, topology, reg=1)


if __name__ == "__main__":
    main()
