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
    
    for current_gossip in Topologies(params):
        .....
    
    Поддерживаются следующие топологии:
        1) Полный граф
        2) Кольцо
        3) Звезда
        4) Чередующиеся кольца и звезды (сменяются через заданное число итераций)
        5) Зацикленная последовательность из заданного количества связных случайных графов
    
    Также поддерживается возможность ограничить количество итераций.
    
    """
    
    def __init__(self, n, topology_type, gossip_matrix_type, n_graphs=None, 
                 plot_graphs=False, max_iter=None):
        """
        n - количество вершин/воркеров
        topology_type - вид топологии: full (полный граф), ring (кольцо), star (звезда),
                                        ring-star (чередуется кольцо и звезда), random (случайные топологии)
        gossip_matrix_type - metropolis (алгоритм Метрополиса), laplacian (через Лапласиан)
        n_graphs - через сколько чередуется кольцо и звезда для ring-star, количество случайных топологий при random
        plot_graps - рисовать ли графы
        max_iter - максимальное количество вызовов (None, если бесконечно)
        """
        self.n = n
        self.topology_type = topology_type
        assert self.topology_type in ["full", "ring", "star", "ring-star", "random"]
        self.gossip_matrix_type = gossip_matrix_type
        assert self.gossip_matrix_type in ["metropolis", "laplacian"]
        self.n_graphs = n_graphs
        if self.topology_type == "ring-star" or self.topology_type == "random":
            assert (type(self.n_graphs) == int) and (self.n_graphs > 0)
        self.plot_graphs = plot_graphs
        self.gossip_matrices = self.get_matrices()
        self.max_iter = max_iter
        self.current_iter = 0
        self.__regime = 1
        
    def get_matrices(self):
        result = None
        if self.topology_type == "full":
            result = [nx.complete_graph(self.n)]
        if self.topology_type == "ring":
            result = [nx.cycle_graph(self.n)]
        if self.topology_type == "star":
            result = [nx.star_graph(self.n - 1)]
        if self.topology_type == "ring-star":
            result = [nx.cycle_graph(self.n), nx.star_graph(self.n - 1)]
        if self.topology_type == "random":
            result = []
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
            return self.gossip_matrices[it % len(self.gossip_matrices)]
        else:
            if it % self.n_graphs == 0:
                self.__regime = 1 - self.__regime
            return self.gossip_matrices[self.__regime]
            
        

    def __graph_to_matrix(self, graphs):
        result = []
        for G in graphs:
            if self.gossip_matrix_type == "metropolis":
                W = np.zeros((self.n, self.n))
                for (i, j) in G.edges:
                    if i != j:
                        W[i, j] = 1. / (max(G.degree[i], G.degree[j]) + 1)
                        W[j, i] = 1. / (max(G.degree[i], G.degree[j]) + 1)
                W += np.diag(np.ones(W.shape[0]) - W.sum(axis=1))
            if self.gossip_matrix_type == "laplacian":
                L = nx.laplacian_matrix(G).todense()
                v = np.linalg.eigvalsh(L)[-1]
                W = np.eye(L.shape[0]) - L / v
            p = np.ones(W.shape[0])
            assert np.allclose(W @ p, p)
            assert np.allclose(W, W.T)
            assert np.linalg.eigvalsh(W)[-2] < 1 + 1e-6
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
