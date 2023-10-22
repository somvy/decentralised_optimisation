import wandb
from wandb.wandb_run import Run

from methods.proxnsadom import PROXNSADOM
import numpy as np
from decentralized.topologies import Topologies

import torch
from utils import print_metrics, save_plot
from oracles.main import get_mnist_oracles
import matplotlib.pyplot as plt


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


def get_method(oracles, topology, reg, chi, wandbrun: Run, saddle_lr):
    return PROXNSADOM(
        oracles=oracles,
        topology=topology,
        max_iter=500,
        eta=1 / (64 * reg * chi ** 2),
        theta=reg / (16 * chi ** 2),
        gamma=reg / 2,
        r=reg,
        wandbrun=wandbrun,
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


def run_single(oracles, topology, reg, run: Run = None):
    chi = topology.chi
    saddle_lr = 5e-2
    method = get_method(oracles, topology, reg, chi, wandbrun=run, saddle_lr=saddle_lr)

    try:
        method.run(log=True, disable_tqdm=False)
    finally:
        filename = "proxnsadom_svm.svg"
        save_plot(method.logs, filename, method_name="PROXNSADOM", task_name="svm")
        proxnsadom_plot(method.logs)
        print("logs saved to %s" % filename)


def main():
    np.random.seed(0xCAFEBABE)
    num_nodes = 10
    reg = 2e-1
    run = wandb.init(project="decentralized opt",
                     config={
                         "task": "svm",
                         "method": "proxnsadom",
                         "reg": reg,
                         "num_nodes": num_nodes
                     })
    topology: Topologies = Topologies(n=num_nodes, topology_type="fully connected",
                                      matrix_type="gossip-laplacian")
    oracles = get_mnist_oracles(num_nodes)
    run_single(oracles, topology, reg, run=run)


if __name__ == "__main__":
    main()
