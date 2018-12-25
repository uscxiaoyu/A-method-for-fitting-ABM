# coding=utf-8
from abmdiffuse import Diffuse
from bassestimate import BassEstimate
from pymongo import MongoClient
from generate_params_boundary import Gen_para
import numpy as np
import networkx as nx
import time


class Gen_para_ih(Gen):
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


def generate_random_graph(degre_sequance):
    G = nx.configuration_model(degre_sequance, create_using=None, seed=None)
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G


def func(p, q, g):
    diff = Diffuse(p, q, g=g, num_runs=40)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.networks

    expon_seq = np.load('dataSources/exponential_sequance.npy')
    gauss_seq = np.load('dataSources/gaussian_sequance.npy')
    logno_seq = np.load('dataSources/lognormal_sequance.npy')
    facebook_graph = nx.read_gpickle('dataSources/facebook.gpickle')
    epinions_graph = nx.read_gpickle('dataSources/epinions.gpickle')   
    g_cont = [nx.barabasi_albert_graph(10000, 3), generate_random_graph(expon_seq), 
              generate_random_graph(gauss_seq), nx.gnm_random_graph(10000, 100000), 
              nx.gnm_random_graph(10000, 30000), nx.gnm_random_graph(10000, 40000), 
              nx.gnm_random_graph(10000, 50000), nx.gnm_random_graph(10000, 60000),
              nx.gnm_random_graph(10000, 70000), nx.gnm_random_graph(10000, 80000), 
              nx.gnm_random_graph(10000, 90000), generate_random_graph(logno_seq),
              nx.watts_strogatz_graph(10000, 6, 0), nx.watts_strogatz_graph(10000, 6, 0.1),
              nx.watts_strogatz_graph(10000, 6, 0.3), nx.watts_strogatz_graph(10000, 6, 0.5),
              nx.watts_strogatz_graph(10000, 6, 0.7), nx.watts_strogatz_graph(10000, 6, 0.9),
              nx.watts_strogatz_graph(10000, 6, 1),
              facebook_graph, epinions_graph]

    txt_cont = ['barabasi_albert_graph(10000,3)', 'exponential_graph(10000,3)', 
                'gaussian_graph(10000,3)', 'gnm_random_graph(10000,100000)', 
                'gnm_random_graph(10000,30000)', 'gnm_random_graph(10000,40000)', 
                'gnm_random_graph(10000,50000)', 'gnm_random_graph(10000,60000)',
                'gnm_random_graph(10000,70000)', 'gnm_random_graph(10000,80000)', 
                'gnm_random_graph(10000,90000)', 'lognormal_graph(10000,3)',
                'watts_strogatz_graph(10000,6,0)', 'watts_strogatz_graph(10000,6,0.1)',
                'watts_strogatz_graph(10000,6,0.3)', 'watts_strogatz_graph(10000,6,0.5)',
                'watts_strogatz_graph(10000,6,0.7)', 'watts_strogatz_graph(10000,6,0.9)',
                'watts_strogatz_graph(10000,6,1.0)',
                'facebook_graph', 'epinions_graph']

    bound_dict = {}
    for j, g in enumerate(g_cont):
        t1 = time.perf_counter()
        print(j+1, txt_cont[j])
        p_cont = (0.0003, 0.02)  
        q_cont = (0.076*3.0/(j + 4), 0.12*3.0/(j + 4))  # 小心设置
        delta = (0.0001, 0.001)
        ger_samp = Gen_para(g=g, p_cont=p_cont, q_cont=q_cont, delta=delta)
        bound = ger_samp.identify_range()
        prj.insert_one({"_id": txt_cont[j], "para_boundary": bound})
        print(f'  time: {time.perf_counter() - t1:.2f}s')
