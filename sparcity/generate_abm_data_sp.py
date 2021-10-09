# coding=utf-8
from abmdiffuse_sp import Diffuse
from pymongo import MongoClient
import numpy as np
import networkx as nx
import time
import multiprocessing


def func(p, q, g):
    p = round(p, 5)
    q = round(q, 5)
    diff = Diffuse(p, q, g=g, num_runs=35)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return str([p, q]), list(np.concatenate(([p, q], x)))


if __name__ == '__main__':
    client = MongoClient('106.14.27.147')
    db = client.abmDiffusion
    prj = db.sparcity_refine
    # for i, txt in enumerate(txt_cont):
    while True:
        mongo_data = prj.find_one_and_update({"state": 0}, {"$set": {"state": 1}})
        if not mongo_data:
            break
        
        g = eval('nx.' + mongo_data["_id"])
        r_p = mongo_data['param_boundary']['p_range']
        r_q = mongo_data['param_boundary']['q_range']
        pq_cont = [(p, q) for p in np.linspace(r_p[0], r_p[1], num=10)
                   for q in np.linspace(r_q[0], r_q[1], num=15)]
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

        print(mongo_data["_id"], f'Time: {(time.perf_counter() - t1):.2f} s')
        diffuse_curves = dict(data)
        prj.update_one(
            {"_id": mongo_data["_id"]}, {"$set": {"diffuse_curves": diffuse_curves}}, upsert=True)
    
    
        
    # txt_cont = ['gnm_random_graph(10000,1000)', 'gnm_random_graph(10000,2000)',
    #     'gnm_random_graph(10000,4000)', 'gnm_random_graph(10000,6000)',
    #     'gnm_random_graph(10000,8000)', 'gnm_random_graph(10000,10000)', 
    #     'gnm_random_graph(10000,20000)']

    # txt_cont = [x['_id'] for x in prj.find(
    #     {"diffuse_curves": {"$exists": False}}, projection={'_id': 1})]
