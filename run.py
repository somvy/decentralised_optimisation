import wandb
from methods import PROXNSADOM, CentralizedGradientDescent, SingleGD, DecentralizedGradientDescent

from decentralized.topologies import Topologies
import config
import torch
from oracles.main import get_oracles


def grid_search(method: PROXNSADOM):
    stats = []
    saddle_lr_candidates: list[float] = [
        1e-8, 5e-8, 1e-7, 5e-7, 1e-6,
        5e-6, 1e-5, 3e-6, 5e-5, 7e-5,
        1e-4, 3e-4, 5e-4, 7e-4, 1e-3,
        3e-3, 5e-3, 7e-3, 1e-2, 3e-2,
        5e-2, 7e-2, 1e-1
    ]

    for saddle_lr in saddle_lr_candidates:
        method.saddle_lr = saddle_lr
        print("running method for saddle lr: ", saddle_lr)
        try:
            method.run(log=True, disable_tqdm=False)
            loss = method.log()["loss"]
            stats.append(loss)
        except StopIteration:
            pass

    best_saddle_lr = saddle_lr_candidates[torch.argmin(torch.FloatTensor(stats))]
    print(stats)
    print("### best saddle lr:", best_saddle_lr)


def main():
    num_nodes = 5
    reg = 1e-1
    max_iter = 200000
    saddle_iters = 1000
    saddle_lr = 5e-4
    task = "mnist"
    topology_type = "fully connected"
    method_name = "proxnsadom"

    wandbrun = wandb.init(
        project=config.WANDB_PROJECT_NAME,
        config={
            "task": task,
            "topology": topology_type,
            "method": method_name,
            "reg": reg,
            "num_nodes": num_nodes
        })

    matrix_type = "mixing-laplacian" if method_name in (
        "centralized gd", "decentralized gd") else "gossip-laplacian"
    topology: Topologies = Topologies(
        n=num_nodes,
        topology_type=topology_type,
        matrix_type=matrix_type
    )

    oracles = get_oracles(task, num_nodes)

    match method_name:
        case "proxnsadom":
            method = PROXNSADOM(
                oracles=oracles,
                topology=topology,
                max_iter=max_iter,
                reg=reg,
                wandbrun=wandbrun,
                saddle_iters=saddle_iters,
                saddle_lr=saddle_lr
            )
        case "centralized gd":
            method = CentralizedGradientDescent(
                oracles=oracles,
                topology=topology,
                max_iter=max_iter,
                stepsize=1e-3,
                wandbrun=wandbrun
            )
        case "decentralized gd":
            method = DecentralizedGradientDescent(
                oracles=oracles,
                topology=topology,
                max_iter=max_iter,
                stepsize=1e-3,
                wandbrun=wandbrun,
            )
        case "single gd":
            method = SingleGD(
                oracle=oracles[0],
                wandbrun=wandbrun,
                max_iter=max_iter,
                lr=1e-3
            )
        case _:
            raise ValueError("No such method!")

    method.run(log=True, disable_tqdm=False)


if __name__ == "__main__":
    main()
