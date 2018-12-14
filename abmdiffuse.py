import numpy as np
import networkx as nx
import time
import random


class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, g=nx.gnm_random_graph(10000, 30000), num_runs=30):
        if not nx.is_directed(g):
            self.g = g.to_directed()
        self.p, self.q = p, q
        self.nodes_array = np.array(self.g)
        self.num_runs = num_runs
        for i in self.g:
            self.g.node[i]['prede'] = list(self.g.predecessors(i))
    
    def decide(self, i):
        num_adopt_prede = sum([self.g.node[k]['state'] for k in self.g.node[i]['prede']])
        prob = 1 - (1 - self.p)*(1 - self.q)**num_adopt_prede
        return prob > random.random()

    def update(self, non_node_array):
        len_nodes = len(non_node_array)
        state_array = np.zeros(len_nodes, dtype=np.bool)
        for i in range(len_nodes):
            node = non_node_array[i]
            if self.decide(node):
                self.g.node[node]['state'] = True
                state_array[i] = True
        
        return np.sum(state_array), non_node_array[state_array==False]

    def single_diffuse(self):
        for i in self.g:
            self.g.node[i]['state'] = False
        non_node_array = self.nodes_array[:]
        num_of_adopt = []
        for i in range(self.num_runs):
            num, non_node_array = self.update(non_node_array)
            num_of_adopt.append(num)

        return num_of_adopt

    def repete_diffuse(self, repetes=10):  # 多次扩散
        return [self.single_diffuse() for i in range(repetes)]


if __name__ == '__main__':
    t1 = time.perf_counter()
    p, q = 0.001, 0.05
    diffu = Diffuse(p, q)
    diff_cont = diffu.repete_diffuse(repetes=10)
    print(f"参数设置: p--{p}, q--{q} network--{diffu.g.number_of_nodes()}")
    print(f"用时{time.perf_counter() - t1:.2f}秒")

