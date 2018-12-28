# coding=utf-8
from pymongo import MongoClient
from bassestimate import BassEstimate
import numpy as np
import datetime
import time
import multiprocessing


def func(x, n):
    p, q = x[:2]
    s_full = x[2:]
    max_ix = np.argmax(s_full)
    s = s_full[:max_ix + 1 + n]
    bassest = BassEstimate(s)
    _, P, Q, M = bassest.optima_search(c_n=200, threshold=10e-6)
    r_2 = bassest.r2([P, Q, M])
    return [round(p, 5), round(q, 5), r_2, P, Q, M]

if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    all_data = db.networks.find_one({"_id": "gnm_random_graph(10000,30000)"}, projection={"diffuse_curves":1})
    prj = db.numPoints
    numpoints_cont = [-1, 0, 1, 2, 3, 4, 5]
    for num_points in numpoints_cont:
        diff_data = all_data["diffuse_curves"]
        pool = multiprocessing.Pool(processes=5)
        result = []
        t1 = time.perf_counter()
        for key in diff_data:
            d = np.array(diff_data[key])
            result.append(pool.apply_async(func, (d, num_points)))

        pool.close()
        pool.join()
        d_dict = {}
        for res in result:
            d = res.get()
            d_dict[str(d[:2])] = d

        print(f"{num_points}: Time elapsed {(time.perf_counter() - t1):.2f}s")
        prj.insert_one({"_id": num_points},  {"estimates": {"ctime": datetime.datetime.now(), **d_dict}})
