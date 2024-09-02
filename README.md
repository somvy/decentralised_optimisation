This repo implements algorithms for decentralised optimisation
from the paper https://arxiv.org/abs/2405.18031 

Specifically:

* Centralised Gradient Descent (for comparison)
* Decentralised Gradient Descent
* Decentralised Communication Sliding
* ZO-SADOM 
* ProxNSADOM (new)


### Requirements:

        python >= 3.10

To reproduce them follow the instructions:

## clone the repo, create the virtual env and install the libraries
(i prefer uv, but feel free to use your favorite package manager)

        git clone https://github.com/somvy/siriusopt2023 && cd siriusopt2023
        python -m pip install uv 
        uv venv && . .venv/bin/activate
        uv pip install -r requirements.txt
        wandb login

## run experiments on some method 

logs are saved to wandb

specify desirable configuration (method, task, network topology, number of nodes etc) in config.yaml

    export PYTHONPATH=.
    python run.py


## Or, if you prefer the Jupyter format, checkout the L1_regression.ipynb

