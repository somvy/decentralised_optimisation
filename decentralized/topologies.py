import numpy as np
import scipy
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


class Topologies:
    """
    Topology generator for distributed optimization, representing a generator that returns the mixing matrix of the current network configuration at each next() call. Thus, the generator can be used in the following way:

    Return type: np.array, dtype=np.float64

    for current_matrix in Topologies(params):
        .....

    The following topologies are supported:
        1) Fully connected graph
        2) Ring
        3) Star
        4) Alternating rings and stars (changing after a specified number of iterations)
        5) Cyclic sequence of a given number of connected random graphs

    It also supports the ability to limit the number of iterations.

    """

    def __init__(
        self,
        n: int,
        topology_type: str,
        matrix_type: str,
        n_graphs: int = None,
        plot_graphs: bool = False,
        max_iter: int = None,
        seed_list=None,
    ):
        """
        n - number of vertices/workers
        topology_type - type of topology: fully connected (complete graph), ring, star,
                                        ring-star (alternating between ring and star), random (random topologies)
        matrix_type - mixing-metropolis, mixing-laplacian, gossip-metropolis, gossip-laplacian
        n_graphs - how often to alternate between ring and star for ring-star, number of random topologies for random
        plot_graphs - whether to plot the graphs
        max_iter - maximum number of calls (None if infinite)
        seed_list - list of seeds of length n_graphs for generating random graphs
        """
        self.n: int = n
        self.topology_type: str = topology_type
        assert self.topology_type in [
            "fully connected",
            "ring",
            "star",
            "ring-star",
            "random",
        ]
        self.matrix_type: str = matrix_type
        assert self.matrix_type in [
            "mixing-metropolis",
            "mixing-laplacian",
            "gossip-metropolis",
            "gossip-laplacian",
        ]
        self.n_graphs = n_graphs
        if self.topology_type == "ring-star" or self.topology_type == "random":
            assert isinstance(self.n_graphs, int) and self.n_graphs > 0
        self.seed_list = seed_list
        if self.seed_list is None and self.topology_type == "random":
            self.seed_list = np.arange(self.n_graphs, dtype=int) + 42
        self.plot_graphs = plot_graphs
        self.max_iter = max_iter
        self.current_iter = 0
        self.__regime: bool = True
        self.chi = -1
        self.lambda_max = 1
        self.norm = None
        self.matrices = self.get_matrices()

    def get_matrices(self) -> list[nx.Graph]:
        result: list[nx.Graph] = []

        if self.topology_type == "fully connected":
            result = [nx.complete_graph(self.n)]
        if self.topology_type == "ring":
            result = [nx.cycle_graph(self.n)]
        if self.topology_type == "star":
            result = [nx.star_graph(self.n - 1)]
        if self.topology_type == "ring-star":
            result = [nx.cycle_graph(self.n), nx.star_graph(self.n - 1)]
        if self.topology_type == "random":
            for i in range(self.n_graphs):
                g = nx.gnp_random_graph(n=self.n, p=0.5, seed=int(self.seed_list[i]))
                if not nx.is_connected(g):
                    g = nx.complement(g)
                result.append(g)

        if self.plot_graphs:
            self.__plot_graphs(result)
        return self.__graph_to_matrix(result)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_iter == self.max_iter:
            raise StopIteration
        it = self.current_iter
        self.current_iter += 1

        if self.topology_type != "ring-star":
            return self.matrices[it % len(self.matrices)]
        else:
            if it % self.n_graphs == 0:
                self.__regime = not self.__regime
            return self.matrices[self.__regime]

    def __graph_to_matrix(self, graphs):
        result = []
        for G in graphs:
            if self.matrix_type.endswith("metropolis"):
                W = np.zeros((self.n, self.n))
                for i, j in G.edges:
                    if i != j:
                        W[i, j] = 1.0 / (max(G.degree[i], G.degree[j]) + 1)
                        W[j, i] = 1.0 / (max(G.degree[i], G.degree[j]) + 1)
                W += np.diag(np.ones(W.shape[0]) - W.sum(axis=1))

            if self.matrix_type.endswith("laplacian"):
                L = nx.laplacian_matrix(G).todense()
                self.norm = np.linalg.norm(L, ord=2)
                self.lambda_max = np.linalg.eigvalsh(L)[-1]
                W = np.eye(L.shape[0]) - L / self.lambda_max

            ones = np.ones(W.shape[0])
            assert np.allclose(W @ ones, ones)
            assert np.allclose(W, W.T)

            lambda2 = np.linalg.eigvalsh(W)
            lambda2 = lambda2[: lambda2.shape[0] - 1]
            lambda2 = lambda2[lambda2 < 1][-1]
            assert lambda2 < 1
            self.chi = max(
                self.chi, 1.0 / (1 - lambda2)
            )  # eq to 1. / \lambda_min+ for gossip

            if self.matrix_type.startswith("gossip"):  # gossip matrices
                W = np.eye(W.shape[0]) - W
                lambda_max = np.linalg.eigvalsh(W)[-1]
                W /= lambda_max

            result.append(W)
        return result

    def __plot_graphs(self, graphs):
        fig, axes = plt.subplots(nrows=len(graphs))
        for i in range(len(graphs)):
            if len(graphs) == 1:
                nx.draw_networkx_edges(
                    graphs[i],
                    nx.rescale_layout_dict(nx.circular_layout(graphs[i]), 10),
                    ax=axes,
                )
            else:
                nx.draw_networkx_edges(
                    graphs[i],
                    nx.rescale_layout_dict(nx.circular_layout(graphs[i]), 10),
                    ax=axes[i],
                )
