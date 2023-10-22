import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
font = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 14
}

matplotlib.rc('font', **font)


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


def save_plot(logs: list[dict[str, float]], filename: str, method_name: str = "GD",
              topology_name: str = "star", task_name: str = "regression"):
    plt.plot([log["loss"] for log in logs], label=method_name)
    plt.xlabel("Communication rounds")
    plt.ylabel("Objective value")
    plt.title(f"Convergence on {task_name} and {topology_name} topology")
    plt.legend()
    plt.savefig(filename)