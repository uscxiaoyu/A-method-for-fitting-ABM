# coding=utf-8
from abmdiffuse import Diffuse
from bassestimate import BassEstimate
from pymongo import MongoClient
from generate_params_boundary import Gen_para
import numpy as np
import networkx as nx
import time


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.indivHeter

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
