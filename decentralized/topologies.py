import numpy as np
import scipy
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


class Topologies:
    """
    Генератор топологий для распределенной оптимизации, представляющий собой генератор, который при каждом вызове next 
    возвращает смешивающую матрицу текущей конфигурации сети. Таким образом можно использовать генератор в следующем варианте:
    
    Возвращаемый тип: np.array, dtype=np.float64
    
    for current_matrix in Topologies(params):
        .....
    
    Поддерживаются следующие топологии:
        1) Полный граф
        2) Кольцо
        3) Звезда
        4) Чередующиеся кольца и звезды (сменяются через заданное число итераций)
        5) Зацикленная последовательность из заданного количества связных случайных графов
    
    Также поддерживается возможность ограничить количество итераций.
    
    """

    def __init__(self, n: int, topology_type: str, matrix_type: str, n_graphs: int = None,
                 plot_graphs: bool =False, max_iter: int = None):
        """
        n - количество вершин/воркеров
        topology_type - вид топологии: full (полный граф), ring (кольцо), star (звезда),
                                        ring-star (чередуется кольцо и звезда), random (случайные топологии)
        matrix_type - mixing-metropolis, mixing-laplacian, gossip-metropolis, gossip-laplacian
        n_graphs - через сколько чередуется кольцо и звезда для ring-star, количество случайных топологий при random
        plot_graps - рисовать ли графы
        max_iter - максимальное количество вызовов (None, если бесконечно)
        """
        self.n: int = n
        self.topology_type: str = topology_type
        assert self.topology_type in ["full", "ring", "star", "ring-star", "random"]
        self.matrix_type: str = matrix_type
        assert self.matrix_type in ["mixing-metropolis", "mixing-laplacian", "gossip-metropolis",
                                    "gossip-laplacian"]
        self.n_graphs = n_graphs
        if self.topology_type == "ring-star" or self.topology_type == "random":
            assert (type(self.n_graphs) == int) and (self.n_graphs > 0)
        self.plot_graphs = plot_graphs
        self.max_iter = max_iter
        self.current_iter = 0
        self.__regime: bool = True
        self.chi = -1
        self.matrices = self.get_matrices()

    def get_matrices(self) -> list[nx.Graph]:
        result: list[nx.Graph] = []

        match self.topology_type:
            case  "full":
                result = [nx.complete_graph(self.n)]
            case "ring":
                result = [nx.cycle_graph(self.n)]
            case "star":
                result = [nx.star_graph(self.n - 1)]
            case "ring-star":
                result = [nx.cycle_graph(self.n), nx.star_graph(self.n - 1)]

            case "random":
                for _ in range(self.n_graphs):
                    g = nx.gnp_random_graph(self.n, 0.5)
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
                for (i, j) in G.edges:
                    if i != j:
                        W[i, j] = 1. / (max(G.degree[i], G.degree[j]) + 1)
                        W[j, i] = 1. / (max(G.degree[i], G.degree[j]) + 1)
                W += np.diag(np.ones(W.shape[0]) - W.sum(axis=1))

            if self.matrix_type.endswith("laplacian"):
                L = nx.laplacian_matrix(G).todense()
                v = np.linalg.eigvalsh(L)[-1]
                W = np.eye(L.shape[0]) - L / v

            ones = np.ones(W.shape[0])
            assert np.allclose(W @ ones, ones)
            assert np.allclose(W, W.T)

            lambda2 = np.linalg.eigvalsh(W)
            print("lambda 2 = ", lambda2)
            lambda2 = lambda2[:lambda2.shape[0] - 1]
            lambda2 = lambda2[lambda2 < 1][-1]
            assert lambda2 < 1
            self.chi = max(self.chi, 1. / (1 - lambda2))  # eq to 1. / \lambda_min+ for gossip

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
                nx.draw_networkx_edges(graphs[i],
                                       nx.rescale_layout_dict(nx.circular_layout(graphs[i]), 10),
                                       ax=axes)
            else:
                nx.draw_networkx_edges(graphs[i],
                                       nx.rescale_layout_dict(nx.circular_layout(graphs[i]), 10),
                                       ax=axes[i])
