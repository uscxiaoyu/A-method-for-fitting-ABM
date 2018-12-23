# coding=utf-8
import numpy as np
import networkx as nx
import time
import random
from abmdiffuse import Diffuse


class Diffuse_ne(Diffuse):  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, alpha, g=nx.gnm_random_graph(10000, 30000), num_runs=35):
        self.g = g.to_directed() if not nx.is_directed(g) else g
        self.p = max(p, 0.00005)
        self.q = min(q, 1)
        self.nodes_array = np.array(self.g)
        self.num_runs = num_runs
        self.alpha = alpha
        for i in self.g:
            self.g.node[i]['prede'] = list(self.g.predecessors(i))
            self.g.node[i]['num_prede'] = len(self.g.node[i]['prede'])

    def decide(self, i):
        num_adopt_prede = sum([self.g.node[k]['state']
                               for k in self.g.node[i]['prede']])
        if self.g.node[i]['num_prede']:
            mi = num_adopt_prede/(self.g.node[i]['num_prede']**self.alpha)
        else:
            mi = 0
        prob = 1 - (1 - self.p)*(1 - self.q)**mi
        return prob > random.random()


if __name__ == '__main__':
    t1 = time.perf_counter()
    p, q, alpha = 0.01, 0.5, 1
    diffu = Diffuse_ne(p, q, alpha)
    diffu_cont = diffu.repete_diffuse(repetes=10)
    print(f"参数设置: p--{p}, q--{q}, alpha--{alpha} network--{diffu.g.number_of_nodes()}")
    print(f"用时{time.perf_counter() - t1:.2f}秒")
