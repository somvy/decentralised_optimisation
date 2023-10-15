import matplotlib.pyplot as plt


def print_metrics(metrics: list[dict[str, float]]) -> None:
    keys = metrics[0].keys()
    accumulators = {key: [] for key in keys}
    for metric in metrics:
        for key in keys:
            accumulators[key].append(metric[key])

    print(", ".join(
        f"{key}: {sum(accumulators[key]) / len(accumulators[key])}"
        for key in keys
    ))


def save_plot(logs: list[dict[str, float]], filename: str):

    plt.plot([log["loss"] for log in logs])
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig(filename)
