# coding=utf-8
import numpy as np
import networkx as nx
import time
import random
from abmdiffuse import Diffuse


class Diffuse_ih(Diffuse):  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, sigma=0.1, g=nx.gnm_random_graph(10000, 30000), num_runs=35):
        self.g = g.to_directed() if not nx.is_directed(g) else g
        self.p, self.q = p, q
        self.sigma = sigma
        self.nodes_array = np.array(self.g)
        self.num_runs = num_runs
        for i in self.g:
            self.g.node[i]['prede'] = list(self.g.predecessors(i))
            self.g.node[i]['p'] = self.p*(1 + self.sigma*np.random.randn())
            self.g.node[i]['q'] = self.q*(1 + self.sigma*np.random.randn())
    
    def decide(self, i):
        num_adopt_prede = sum([self.g.node[k]['state'] for k in self.g.node[i]['prede']])
        prob = 1 - (1 - self.g.node[i]['p'])*(1 - self.g.node[i]['q'])**num_adopt_prede
        return prob > random.random()


if __name__ == '__main__':
    import pylab as pl
    t1 = time.perf_counter()
    p, q = 0.001, 0.05
    diffu = Diffuse_ih(p=p, q=q)
    diffu_cont = diffu.repete_diffuse(repetes=20)
    print(f"参数设置: p--{p}, q--{q} sigma--{diffu.sigma}")
    print(f"用时{time.perf_counter() - t1:.2f}秒")
    fig = pl.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    for line in diffu_cont:
        ax.plot(line, 'k-', lw=0.5, alpha=0.5)
    ax.plot(np.mean(diffu_cont, axis=0), 'r-', lw=2)
    pl.show()

