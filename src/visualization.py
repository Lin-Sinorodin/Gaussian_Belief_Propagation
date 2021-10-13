import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

from .gabp_mrf import find_edges
from .utils import get_running_statistics


def set_plot_options():
    # restore matplotlib default rc parameters
    plt.rcParams.update(plt.rcParamsDefault)

    plt.matplotlib.rc('figure', figsize=(9, 5))
    plt.matplotlib.rc('grid', linestyle='dashed', linewidth=1, alpha=0.25)
    plt.matplotlib.rc('font', family='serif', size=12)
    plt.matplotlib.rc('legend', fontsize=12)

    # Check if latex is installed. Source: https://stackoverflow.com/a/40895025
    from distutils.spawn import find_executable
    if find_executable('latex'):
        plt.matplotlib.rc('text', usetex=True)

    # Change ticks
    plt.rcParams['xtick.major.size'] = 7.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.major.size'] = 7.0
    plt.rcParams['ytick.minor.size'] = 3.0
    plt.rcParams['ytick.direction'] = 'inout'


def get_plot_colors():
    return ['#6d82ee', '#97964a', '#ffd44f', '#f4777f', '#ad8966']


class NetworkxGraph:
    def __init__(self, A):
        self.A = A
        self.graph = self._initialize_graph()

    def _initialize_graph(self) -> nx.graph:
        """initialize a networkx graph object from data matrix A"""
        graph = nx.Graph()
        for i in range(self.A.shape[0]):
            graph.add_node(i)

        edges = find_edges(self.A)
        for node in edges:
            graph.add_edge(node[0], node[1])

        return graph

    def draw_graph(self, title, color, node_size=20):
        """draw a networkx graph"""
        nx.draw(self.graph, node_size=node_size, node_color=color)
        plt.title(title)
        plt.show()


class AnalyzeResult:
    @staticmethod
    def plot_gabp_convergence(iter_dist, color):
        """plot the distance between each iteration on the GaBP algorithm"""
        fig, ax = plt.subplots()

        plt.semilogy(iter_dist, color=color, label='Total', marker='+')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L2 Distance')
        ax.set_title('L2 Distance Between Following Iterations ')
        plt.show()

    @staticmethod
    def plot_time_vs_iterations(num_iters: int, dims: list):
        colors = get_plot_colors()
        fig, ax = plt.subplots()

        for i, dim in enumerate(dims):
            running_time, num_iter = get_running_statistics(dim, num_iters=num_iters)
            linregress = stats.linregress(running_time, num_iter)

            plt.scatter(running_time, num_iter, color=colors[i], marker='+', label=dim)
            plt.plot(running_time, linregress.intercept + linregress.slope*running_time, color=colors[i], alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', title='Number of Nodes', ncol=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Iterations till Convergence')
        ax.set_title('Running Time V.S Iterations')
        ax.grid()
        plt.show()
