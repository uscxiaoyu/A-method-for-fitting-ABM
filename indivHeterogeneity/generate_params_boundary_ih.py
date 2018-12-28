# coding=utf-8
from abmdiffuse import Diffuse
from bassestimate import BassEstimate
from pymongo import MongoClient
from generate_params_boundary import Gen_para
import numpy as np
import networkx as nx
import time


class Gen_para_ih(Gen_para):
    def __init__(self, g, p_cont=(0.001, 0.02), q_cont=(0.08, 0.1), delta=(0.0005, 0.01)):
        self.p_cont = p_cont
        self.q_cont = q_cont
        self.d_p, self.d_q = delta
        self.g = g
        self.num_nodes = self.g.number_of_nodes()

    def add_data(self, p, q):
        diff = Diffuse(p, q, g=self.g)
        x = np.mean(diff.repete_diffuse(), axis=0)
        max_idx = np.argmax(x)
        s = x[:(max_idx + 2)]
        para_range = [[1e-6, 0.1], [1e-5, 0.8], [sum(s), 4*self.num_nodes]]
        bassest = BassEstimate(s, para_range)
        bassest.t_n = 1000
        res = bassest.optima_search(c_n=200, threshold=10e-6)
        return res[1:3]  # P, Q


def func(p, q, g):
    diff = Diffuse(p, q, g=g, num_runs=40)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.indivHeter

    facebook_graph = nx.read_gpickle('dataSources/facebook.gpickle')
    epinions_graph = nx.read_gpickle('dataSources/epinions.gpickle') 

    sigma_cont = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    for j, sigma in enumerate(sigma_cont):
        t1 = time.perf_counter()
        print(j+1, f"sigam:{sigma}")
        p_cont = (0.0003, 0.02)
        q_cont = (0.076*3.0/(j + 3), 0.12*3.0/(j + 3))  # 小心设置
        delta = (0.0001, 0.001)
        ger_samp = Gen_para(sigma=sigma, p_cont=p_cont, q_cont=q_cont, delta=delta)
        bound = ger_samp.identify_range()
        prj.insert_one({"_id": sigma, "para_boundary": bound})
        print(f'  time: {time.perf_counter() - t1:.2f}s')
