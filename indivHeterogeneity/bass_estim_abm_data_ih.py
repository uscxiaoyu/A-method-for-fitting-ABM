# coding=utf-8
from pymongo import MongoClient
import bassestimate as eb
import numpy as np
import datetime
import time
import multiprocessing


def func(x, n=1):
    p, q = x[:2]
    s_full = x[2:]
    max_ix = np.argmax(s_full)
    s = s_full[:max_ix + 1 + n]
    bassest = eb.BassEstimate(s)
    _, P, Q, M = bassest.optima_search(c_n=200, threshold=10e-6)
    r_2 = bassest.r2([P, Q, M])
    return [round(p, 5), round(q, 5), r_2, P, Q, M]

if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.indivHeter
    alpha_cont = [x['_id'] for x in prj.find({"estimates": {"$exists": False}}, projection={'_id': 1})]
    for alpha in alpha_cont:
        all_data = prj.find_one({"_id": alpha})
        diff_data = all_data["diffuse_curves"]
        pool = multiprocessing.Pool(processes=5)
        result = []
        t1 = time.perf_counter()
        i = 0
        for key in diff_data:
            d = np.array(diff_data[key])
            result.append(pool.apply_async(func, (d,)))
            i += 1

        pool.close()
        pool.join()
        d_dict = {}
        for res in result:
            d = res.get()
            d_dict[str(d[:2])] = d

        print(f"{alpha}: Time elapsed {(time.perf_counter() - t1):.2f}s")
        prj.update_one({"_id": alpha}, {"$set": {"estimates": {"ctime": datetime.datetime.now(), **d_dict}}}, upsert=True)
