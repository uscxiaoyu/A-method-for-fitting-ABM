# coding=utf-8
import multiprocessing
import numpy as np
import networkx as nx
import time
import random
import multiprocessing

class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, alpha=0, sigma=0, g=nx.gnm_random_graph(10000, 30000), num_runs=35, multi_proc=False):
        '''
        p: 创新系数
        q: 模仿系数
        alpha: 邻居效应
        sigma: 个体差异性
        g: 网络
        num_runs: 仿真时间步
        '''
        self.multi_proc = multi_proc
        self.g = g.to_directed() if not nx.is_directed(g) else g
        self.nodes_array = np.array(self.g)
        self.num_runs = num_runs
        self.alpha = alpha
        self.sigma = sigma
        for i in self.g:
            self.g.nodes[i]['prede'] = list(self.g.predecessors(i))
            self.g.nodes[i]['num_prede'] = len(self.g.nodes[i]['prede'])
            self.g.nodes[i]['p'] = p*(1 + self.sigma*np.random.randn())
            self.g.nodes[i]['q'] = q*(1 + self.sigma*np.random.randn())

    def decide(self, i):
        num_adopt_prede = sum([self.g.nodes[k]['state'] for k in self.g.nodes[i]['prede']])
        prob = 1 - (1 - self.g.nodes[i]['p'])*(1 - self.g.nodes[i]['q'])**num_adopt_prede
        if self.g.nodes[i]['num_prede']:
            mi = num_adopt_prede/(self.g.nodes[i]['num_prede']**self.alpha)
        else:
            mi = 0
        prob = 1 - (1 - self.g.nodes[i]['p'])*(1 - self.g.nodes[i]['q'])**mi
        return prob > random.random()

    def update(self, non_node_array):
        len_nodes = len(non_node_array)
        state_array = np.zeros(len_nodes, dtype=np.bool)
        for i in range(len_nodes):
            node = non_node_array[i]
            if self.decide(node):
                self.g.nodes[node]['state'] = True
                state_array[i] = True
        return np.sum(state_array), non_node_array[state_array == False]

    def single_diffuse(self):
        for i in self.g:
            self.g.nodes[i]['state'] = False
        non_node_array = self.nodes_array[:]
        num_of_adopt = []
        for i in range(self.num_runs):
            num, non_node_array = self.update(non_node_array)
            num_of_adopt.append(num)

        return num_of_adopt

    def repete_diffuse(self, repetes=10):  # 多次扩散
        if self.multi_proc:
            if repetes < 5:
                pool = multiprocessing.Pool(processes=repetes)
            else:
                pool = multiprocessing.Pool(processes=5)
            proc = []
            for i in range(repetes):
                proc.append(pool.apply_async(self.single_diffuse))

            pool.close()
            pool.join()
            return [res.get() for res in proc]
        else:
            return [self.single_diffuse() for i in range(repetes)]

if __name__ == '__main__':
    t1 = time.perf_counter()
    p, q, alpha = 0.01, 0.5, 0
    diffu = Diffuse(p, q, alpha=alpha)
    diffu_cont = diffu.repete_diffuse(repetes=10)
    print(f"参数设置: p--{p}, q--{q}, alpha--{alpha} network--{diffu.g.number_of_nodes()}")
    print(f"用时{time.perf_counter() - t1:.2f}秒")
