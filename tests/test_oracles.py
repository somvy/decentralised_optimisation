from oracles.mnist_mlp_oracle import MNISTMLP, MLP
from torchvision.datasets import MNIST
from torchvision import transforms
from utils import save_plot


def test_nn():
    init_model = MLP()
    train_dataset = MNIST(".", download=True, transform=transforms.ToTensor(), train=True)
    oracle = MNISTMLP(Xy=train_dataset, init_network=init_model)

    iters = 100000
    stepsize = 3e-4
    log = []

    for _ in range(iters):
        x = oracle.get_params()
        grads = oracle.grad()
        new_params = [param - stepsize * grad for param, grad in zip(x, grads)]
        oracle.set_params(new_params)
        log.append({"loss": oracle()})

    save_plot(log, "nn.png")
