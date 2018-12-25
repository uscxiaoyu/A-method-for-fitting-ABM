# coding=utf-8
import numpy as np
import networkx as nx
import time
import random
from abmdiffuse import Diffuse


class Diffuse_ih(Diffuse):  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p_tuple, q_tuple, g=nx.gnm_random_graph(10000, 30000), num_runs=35):
        self.g = g.to_directed() if not nx.is_directed(g) else g
        self.p_mu, self.p_sigma = p_tuple
        self.q_mu, self.q_sigma = q_tuple
        self.nodes_array = np.array(self.g)
        self.num_runs = num_runs
        for i in self.g:
            self.g.node[i]['prede'] = list(self.g.predecessors(i))
            self.g.node[i]['p'] = self.p_mu + self.p_sigma * np.random.randn()
            self.g.node[i]['q'] = self.q_mu + self.q_sigma * np.random.randn()
    
    def decide(self, i):
        num_adopt_prede = sum([self.g.node[k]['state'] for k in self.g.node[i]['prede']])
        prob = 1 - (1 - self.g.node[i]['p'])*(1 - self.g.node[i]['q'])**num_adopt_prede
        return prob > random.random()


if __name__ == '__main__':
    t1 = time.perf_counter()
    p_tuple, q_tuple = (0.001, 0.0005), (0.05, 0.01)
    diffu = Diffuse_ih(p_tuple, q_tuple)
    diffu_cont = diffu.repete_diffuse(repetes=10)
    print(f"参数设置: p--{p_tuple}, q--{q_tuple} network--{diffu.g.number_of_nodes()}")
    print(f"用时{time.perf_counter() - t1:.2f}秒")

