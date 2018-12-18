# coding=utf-8
import time
import datetime
import numpy as np
import multiprocessing
from pymongo import MongoClient
from bassestimate import BassEstimate, BassForecast

def func(s):
    bass_fore = BassForecast(s, n=3, b_idx=8)  # 1步向前和3步向前预测
    res = bass_fore.run()
    return res


def func(x, n=3, b_idx=8):
    p, q = x[:2]
    s_full = x[2:]
    bass_fore = BassForecast(s_full, n=n, b_idx=b_idx)  # 1步向前和3步向前预测
    res = bass_fore.run()
    return [round(p, 5), round(q, 5), res[0], res[1]]

if __name__ == "__main__":
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.networks
    txt_cont = [x['_id'] for x in prj.find(
        {"forecasts": {"$exists": False}}, projection={'_id': 1})]
    
    for txt in txt_cont:
        all_data = prj.find_one({"_id": txt})
        diff_data = all_data["diffuse_curves"]
        pool = multiprocessing.Pool(processes=5)
        result = []
        t1 = time.perf_counter()

        for key in diff_data:
            d = np.array(diff_data[key])
            result.append(pool.apply_async(func, (d,)))

        pool.close()
        pool.join()
        d_dict = {}
        for res in result:
            d = res.get()
            d_dict[str(d[:2])] = d

        print(f"{txt}: Time elapsed {(time.perf_counter() - t1):.2f}s")
        prj.update_one({"_id": txt}, {"$set": {"forecasts": {
                       "ctime": datetime.datetime.now(), **d_dict}}}, upsert=True)

