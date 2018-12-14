# coding=utf-8
from abmdiffuse import Diffuse
from bassestimate import BassEstimate
import numpy as np
import networkx as nx
import time
import random
import pickle


class Gen_para:
    def __init__(self, g, p_cont=(0.001, 0.02), q_cont=(0.08, 0.1), delta=(0.0005, 0.01)):
        self.p_cont = p_cont
        self.q_cont = q_cont
        self.d_p, self.d_q = delta
        self.g = g

    def add_data(self, p, q):
        diff = Diffuse(p, q, g=self.g)
        x = np.mean(diff.repete_diffuse(), axis=0)
        max_idx = np.argmax(x)
        s = x[: (max_idx + 2)]
        para_range = [[1e-6, 0.1], [1e-5, 0.8], [2000, 20000]]
        bassest = BassEstimate(s, para_range)
        bassest.t_n = 1000
        res = bassest.optima_search(c_n=200, threshold=10e-8)
        return res[:2]

    def identify_range(self):
        min_p, max_p = self.p_cont
        min_q, max_q = self.q_cont
        est_cont = [self.add_data(p, q) for p, q in ((min_p, min_q), (max_p, max_q))]
        i = 1
        while True:
            min_P, min_Q = est_cont[0]
            max_P, max_Q = est_cont[1]
            print(i, ' P:%.4f~%.4f' % (min_P, max_P), ' Q:%.4f~%.4f' % (min_Q, max_Q))
            c1, c2 = 0, 0
            if min_P > 0.0007 or min_p > 0.0005:  # in case of min_p < 0
                if min_p - self.d_p > 0:
                    min_p -= self.d_p
                else:
                    min_p *= 0.8
                c1 += 1
            if min_Q > 0.38:
                min_q -= self.d_q
                c1 += 1
            if max_P < 0.03:
                max_p += self.d_p
                c2 += 1
            if max_Q < 0.53:
                max_q += self.d_q
                c2 += 1

            i += 1

            if c1 + c2 != 0:  # check which ends should be updated
                if c1 != 0:
                    est_cont[0] = self.add_data(min_p, min_q)
                if c2 != 0:
                    est_cont[1] = self.add_data(max_p, max_q)
            else:
                break

            if i == 25:
                break

        return [(min_p, max_p), (min_q, max_q)], [(min_P, max_P), (min_Q, max_Q)]

    def generate_sample(self, n_p=10, n_q=20):
        rg_p, rg_q = self.identify_range()
        sp_cont = [(p, q) for p in np.linspace(rg_p[0], rg_p[1], n_p) for q in np.linspace(rg_q[0], rg_q[1], n_q)]
        return sp_cont


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
    """
    expon_seq = np.load('exponential_sequance.npy')
    gauss_seq = np.load('gaussian_sequance.npy')
    logno_seq = np.load('lognormal_sequance.npy')
    g_cont = [nx.barabasi_albert_graph(10000, 3), generate_random_graph(expon_seq), generate_random_graph(gauss_seq),
              nx.gnm_random_graph(10000, 100000), nx.gnm_random_graph(10000, 30000),
              nx.gnm_random_graph(10000, 40000), nx.gnm_random_graph(10000, 50000), nx.gnm_random_graph(10000, 60000),
              nx.gnm_random_graph(10000, 70000), nx.gnm_random_graph(10000, 80000), nx.gnm_random_graph(10000, 90000),
              generate_random_graph(logno_seq),
              nx.watts_strogatz_graph(10000, 6, 0), nx.watts_strogatz_graph(10000, 6, 0.1),
              nx.watts_strogatz_graph(10000, 6, 0.3), nx.watts_strogatz_graph(10000, 6, 0.5),
              nx.watts_strogatz_graph(10000, 6, 0.7), nx.watts_strogatz_graph(10000, 6, 0.9),
              nx.watts_strogatz_graph(10000, 6, 1)]

    txt_cont = ['barabasi_albert_graph(10000,3)', 'exponential_graph(10000,3)', 'gaussian_graph(10000,3)',
                'gnm_random_graph(10000,100000)', 'gnm_random_graph(10000,30000)',
                'gnm_random_graph(10000,40000)', 'gnm_random_graph(10000,50000)', 'gnm_random_graph(10000,60000)',
                'gnm_random_graph(10000,70000)', 'gnm_random_graph(10000,80000)', 'gnm_random_graph(10000,90000)',
                'lognormal_graph(10000,3)',
                'watts_strogatz_graph(10000,6,0)', 'watts_strogatz_graph(10000,6,0.1)',
                'watts_strogatz_graph(10000,6,0.3)', 'watts_strogatz_graph(10000,6,0.5)',
                'watts_strogatz_graph(10000,6,0.7)', 'watts_strogatz_graph(10000,6,0.9)',
                'watts_strogatz_graph(10000,6,1.0)']

    """

    bound_dict = {'gnm_random_graph(10000,30000)': [(0.00038, 0.02182), (0.076, 0.12)]}
    g_cont = [nx.gnm_random_graph(10000, 40000), nx.gnm_random_graph(10000, 50000), nx.gnm_random_graph(10000, 60000),
              nx.gnm_random_graph(10000, 70000), nx.gnm_random_graph(10000, 80000), nx.gnm_random_graph(10000, 90000),
              nx.gnm_random_graph(10000, 100000)]

    txt_cont = ['gnm_random_graph(10000,40000)', 'gnm_random_graph(10000,50000)', 'gnm_random_graph(10000,60000)',
                'gnm_random_graph(10000,70000)', 'gnm_random_graph(10000,80000)', 'gnm_random_graph(10000,90000)',
                'gnm_random_graph(10000,100000)']

    for j, g in enumerate(g_cont):
        t1 = time.clock()
        print(j + 1, txt_cont[j])
        p_cont = (0.0003, 0.02)
        q_cont = (0.076 * 3.0 / (j + 4), 0.12 * 3.0 / (j + 4))
        delta = (0.00031, 0.008 * 3.0 / (j + 4))
        ger_samp = Gen_para(g=g, p_cont=p_cont, q_cont=q_cont, delta=delta)
        bound_dict[txt_cont[j]] = ger_samp.identify_range()
        print('  time: %.2f s' % (time.clock() - t1))

    f = open('dataSources/bound(gmm).pkl', 'wb')
    pickle.dump(bound_dict, f)
