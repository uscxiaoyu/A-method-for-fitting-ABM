#coding=utf-8
#%%
from abmdiffuse import Diffuse
from pymongo import MongoClient
import numpy as np
import networkx as nx
import time
import multiprocessing
import bassestimate as eb
import datetime

#%%
client = MongoClient('localhost', 27017)
db = client.abmDiffusion
prj = db.indivHeter
sigma = 0.1
#%%
# 生成扩散曲线集合
def func_1(p, q, sigma):
    p = round(p, 5)
    q = round(q, 5)
    diff = Diffuse(p=p, q=q, sigma=sigma, num_runs=35, multi_proc=True)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return str([p, q]), list(np.concatenate(([p, q], x)))
#%%
t1 = time.perf_counter()
mongo_date = prj.find_one({"_id": sigma})
r_p = mongo_date['param_boundary']['p_range']
r_q = mongo_date['param_boundary']['q_range']
pq_cont = [(p, q) for p in np.linspace(r_p[0], r_p[1], num=10)
                    for q in np.linspace(r_q[0], r_q[1], num=15)]
data = []
i = 1
for p, q in pq_cont:
    t2 = time.perf_counter()
    res = func_1(p, q, sigma)
    data.append(res)
    print(i, f"{time.perf_counter() - t2: .2f}s")
    i += 1

print(sigma, f'Time: {(time.perf_counter() - t1):.2f} s')
diffuse_curves = dict(data)

#%%
prj.update_one({"_id": sigma}, {"$set": {"diffuse_curves": diffuse_curves}}, upsert=True)

#%%
# 估计参数
def func_2(x, n=1):
    p, q = x[:2]
    s_full = x[2:]
    max_ix = np.argmax(s_full)
    s = s_full[:max_ix + 1 + n]
    bassest = eb.BassEstimate(s)
    _, P, Q, M = bassest.optima_search(c_n=200, threshold=10e-6)
    r_2 = bassest.r2([P, Q, M])
    return [round(p, 5), round(q, 5), r_2, P, Q, M]
#%%
all_data = prj.find_one({"_id": sigma})
diff_data = all_data["diffuse_curves"]
d_dict = {}
i = 1
for key in diff_data:
    t2 = time.perf_counter()
    d = np.array(diff_data[key])
    res = func_2(d)
    d_dict[str(res[:2])] = res
    print(i,  f"{res[2]}, {time.perf_counter() - t2: .2f}s")
    i += 1

print(f"{sigma}: Time elapsed {(time.perf_counter() - t1):.2f}s")
#%%
prj.update_one({"_id": sigma}, {"$set": {"estimates": {"ctime": datetime.datetime.now(), **d_dict}}}, upsert=True)
#%%
x = prj.find_one({"_id": sigma})
x.keys()
#%%
x["estimates"]
#%%
from bassestimate import BassForecast
#%%
def func_3(x, n=3):
    p, q = x[:2]
    s_full = x[2:]
    b_idx = np.argmax(s_full) - 1  # 巅峰扩散前1步
    e_idx = np.argmax(s_full) + 3  # 巅峰扩散后3步
    bass_fore = BassForecast(s_full, n=n, b_idx=b_idx, e_idx=e_idx) 
    res = bass_fore.run()   # [1步向前, 3步向前预测]
    return [round(p, 5), round(q, 5), res[0], res[1]]

txt_cont = [sigma]
  
for txt in txt_cont:
    all_data = prj.find_one({"_id": txt})
    diff_data = all_data["diffuse_curves"]
    pool = multiprocessing.Pool(processes=5)
    result = []
    t1 = time.perf_counter()

    for key in diff_data:
        d = np.array(diff_data[key])
        result.append(pool.apply_async(func_3, (d,)))

    pool.close()
    pool.join()
    d_dict = {}
    for res in result:
        d = res.get()
        d_dict[str(d[:2])] = d

    print(f"{txt}: Time elapsed {(time.perf_counter() - t1):.2f}s")
    prj.update_one({"_id": txt}, {"$set": {"forecasts": {
                        "ctime": datetime.datetime.now(), **d_dict}}}, upsert=True)