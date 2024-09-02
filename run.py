import wandb
import hydra
from methods import (
    PROXNSADOM,
    CentralizedGradientDescent,
    SingleGD,
    DecentralizedGradientDescent,
    DecentralizedCommunicationSliding,
    ZOSADOM,
)
from omegaconf import DictConfig
from decentralized.topologies import Topologies
import torch
from oracles.main import get_oracles


def grid_search(method: PROXNSADOM):
    stats = []
    saddle_lr_candidates: list[float] = [
        1e-8,
        5e-8,
        1e-7,
        5e-7,
        1e-6,
        5e-6,
        1e-5,
        3e-6,
        5e-5,
        7e-5,
        1e-4,
        3e-4,
        5e-4,
        7e-4,
        1e-3,
        3e-3,
        5e-3,
        7e-3,
        1e-2,
        3e-2,
        5e-2,
        7e-2,
        1e-1,
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


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    wandbrun = wandb.init(
        project=cfg.WANDB_PROJECT_NAME,
        config={
            "task": cfg.task,
            "topology": cfg.topology_type,
            "method": cfg.method_name,
            "reg": cfg.reg,
            "num_nodes": cfg.num_nodes,
        },
    )

    matrix_type = (
        "mixing-laplacian"
        if cfg.method_name in ("centralized gd", "decentralized gd")
        else "gossip-laplacian"
    )
    topology: Topologies = Topologies(
        n=cfg.num_nodes, topology_type=cfg.topology_type, matrix_type=matrix_type
    )

    oracles = get_oracles(cfg.task, cfg.num_nodes)

    match cfg.method_name:
        case "proxnsadom":
            method = PROXNSADOM(
                oracles=oracles,
                topology=topology,
                max_iter=cfg.max_iter,
                reg=cfg.reg,
                wandbrun=wandbrun,
                saddle_iters=cfg.saddle_iters,
                saddle_lr=cfg.saddle_lr,
            )
        case "centralized gd":
            method = CentralizedGradientDescent(
                oracles=oracles,
                topology=topology,
                    max_iter=cfg.max_iter,
                stepsize=1e-3,
                wandbrun=wandbrun,
            )
        case "decentralized gd":
            method = DecentralizedGradientDescent(
                oracles=oracles,
                topology=topology,
                max_iter=cfg.max_iter,
                stepsize=1e-3,
                wandbrun=wandbrun,
            )
        case "single gd":
            method = SingleGD(
                    oracle=oracles[0], wandbrun=wandbrun, max_iter=cfg.max_iter, lr=1e-3
            )
        case "decentralized cs":
            method = DecentralizedCommunicationSliding(
                oracles=oracles,
                topology=topology,
                max_iter=cfg.max_iter,
                wandbrun=wandbrun,
            )
        case "zosadom":
            method = ZOSADOM(
                oracles=oracles,
                topology=topology,
                max_iter=cfg.max_iter,
                wandbrun=wandbrun,
            )
        case _:
            raise ValueError(
                "No such method!, enter one of the following: proxnsadom, centralized gd, decentralized gd, single gd"
            )

    method.run(log=True, disable_tqdm=False)


if __name__ == "__main__":
    main()
