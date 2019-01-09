# coding=utf-8
from abmdiffuse import Diffuse
from pymongo import MongoClient
import numpy as np
import networkx as nx
import time
import multiprocessing


def generate_random_graph(degre_sequance):
    G = nx.configuration_model(degre_sequance, create_using=None, seed=None)
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G


def func(p, q, g):
    p = round(p, 5)
    q = round(q, 5)
    diff = Diffuse(p, q, g=g, num_runs=40)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return str([p, q]), list(np.concatenate(([p, q], x)))


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.networks
    expon_seq = np.load('dataSources/exponential_sequance.npy')
    gauss_seq = np.load('dataSources/gaussian_sequance.npy')
    logno_seq = np.load('dataSources/lognormal_sequance.npy')
    facebook_graph = nx.read_gpickle('dataSources/facebook.gpickle')
    epinions_graph = nx.read_gpickle('dataSources/epinions.gpickle')

    txt_cont = {x['_id'] for x in prj.find({}, projection={'_id': 1})}
    prj2 = db.abmDatabase  # 新建一个集合
    e_cont = {x['_id'] for x in prj2.find({})}
    txt_cont = list(txt_cont - e_cont)
    print("剩余:", txt_cont)

    for i, key in enumerate(txt_cont):
        mongo_date = prj.find_one({"_id": key})
        r_p = mongo_date['param_boundary']['p_range']
        r_q = mongo_date['param_boundary']['q_range']
        pq_cont = [(p, q) for p in np.linspace(r_p[0], r_p[1], num=30)
                          for q in np.linspace(r_q[0], r_q[1], num=50)]
        if key == 'exponential_graph(10000,3)':
            g = generate_random_graph(expon_seq)
        elif key == 'gaussian_graph(10000,3)':
            g = generate_random_graph(gauss_seq)
        elif key == 'lognormal_graph(10000,3)':
            g = generate_random_graph(logno_seq)
        elif key in ['facebook_graph', 'epinions_graph']:
            g = eval(key)
        else:
            g = eval("nx." + key)

        t1 = time.perf_counter()
        pool = multiprocessing.Pool(processes=5)
        result = []
        for p, q in pq_cont:
            result.append(pool.apply_async(func, (p, q, g)))
        pool.close()
        pool.join()
        data = []
        for res in result:
            data.append(res.get())

        print(i + 1, key, f'Time: {(time.perf_counter() - t1):.2f} s')
        diffuse_curves = dict(data)
        new_curves = {}
        for i, k in enumerate(diffuse_curves):
            new_curves[str(i)] = diffuse_curves[k]

        prj2.insert_one({"_id": key, "diffuse_curves": new_curves})
        