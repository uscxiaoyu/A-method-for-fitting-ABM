# coding=utf-8
import time
import datetime
import numpy as np
import multiprocessing
from pymongo import MongoClient
from bassestimate import BassForecast


def func(x, n=3):
    p, q = x[:2]
    s_full = x[2:]
    b_idx = np.argmax(s_full) - 1  # 巅峰扩散前1步
    e_idx = np.argmax(s_full) + 3  # 巅峰扩散后3步
    bass_fore = BassForecast(s_full, n=n, b_idx=b_idx, e_idx=e_idx) 
    res = bass_fore.run()   # [1步向前, 3步向前预测]
    return [round(p, 5), round(q, 5), res[0], res[1]]


if __name__ == "__main__":
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.neighEffects
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
