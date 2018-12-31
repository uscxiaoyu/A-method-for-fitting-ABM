from abmdiffuse import Diffuse
from bassestimate import BassEstimate
from generate_params_boundary import Gen_para
from pymongo import MongoClient
import networkx as nx
import numpy as np
import time


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.neighEffects
    alpha_cont = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.3, 1.5, 2]
    bound_dict = {}
    g = nx.gnm_random_graph(10000, 30000)
    for j, alpha in enumerate(alpha_cont[1:]):
        t1 = time.perf_counter()
        print(j + 1, alpha)
        p_cont = (0.0005, 0.03)
        q_cont = (0.08*6**alpha, 0.12*6**alpha)
        delta = (0.001*6**alpha, 0.001*6**alpha)
        ger_samp = Gen_para(alpha, p_cont=p_cont, q_cont=q_cont, delta=delta) 
        bound = ger_samp.identify_range()
        prj.insert_one({"_id": alpha, "para_boundary": bound})
        print(f'  time: {time.perf_counter() - t1:.2f}s')
