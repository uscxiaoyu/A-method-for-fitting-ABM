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


def func(p, q, alpha):
    p = round(p, 5)
    q = round(q, 5)
    diff = Diffuse(p, q, alpha=alpha, num_runs=35)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return str([p, q]), list(np.concatenate(([p, q], x)))


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.neighEffects
    alpha_cont = [x['_id'] for x in prj.find(
        {"diffuse_curves": {"$exists": False}}, projection={'_id': 1})]
    for i, alpha in enumerate(alpha_cont):
        mongo_date = prj.find_one({"_id": alpha})
        r_p = mongo_date['param_boundary']['p_range']
        r_q = mongo_date['param_boundary']['q_range']
        pq_cont = [(p, q) for p in np.linspace(r_p[0], r_p[1], num=10)
                   for q in np.linspace(r_q[0], r_q[1], num=15)]
        t1 = time.perf_counter()
        pool = multiprocessing.Pool(processes=5)
        result = []
        for p, q in pq_cont:
            result.append(pool.apply_async(func, (p, q, alpha)))
        pool.close()
        pool.join()
        data = []
        for res in result:
            data.append(res.get())

        print(i + 1, alpha, f'Time: {(time.perf_counter() - t1):.2f} s')
        diffuse_curves = dict(data)
        prj.update_one(
            {"_id": alpha}, {"$set": {"diffuse_curves": diffuse_curves}}, upsert=True)
