#coding=utf-8

#%%
get_ipython().run_line_magic('pylab', 'inline')
from abmdiffuse import Diffuse
from pymongo import MongoClient
from bassestimate import BassEstimate, BassForecast
from generate_params_boundary import Gen_para
import pylab as pl
import numpy as np
import networkx as nx
import time
import multiprocessing

#%%
client = MongoClient('localhost', 27017)
db = client.abmDiffusion
prj = db.networks

#%%
ws0 = prj.find_one({"_id": 'watts_strogatz_graph(10000,6,0)'})

#%%
ws0.keys()

#%%
diff_cont = ws0["diffuse_curves"]

#%%
diff_list = [diff_cont[key] for key in diff_cont]

#%%
import pylab as pl
for line in diff_list:
    pl.plot(line[2:])


#%%
for line in diff_list[:15]:
    pl.plot(line[2:])

#%%
param_bound = ws0["param_boundary"]
print(param_bound)


#%%
t1 = time.perf_counter()
g = nx.watts_strogatz_graph(10000, 6, 0)
p, q = 0.001, 0.1
diff = Diffuse(p, q, g=g)
diff_cont = diff.repete_diffuse(20)
S = np.mean(diff_cont, axis=0)
print(f"Time elasped:{time.perf_counter()-t1:.2f}s")
print(f"网络: nx.watts_strogatz_graph(10000, 6, 0)")
print(f"最大采纳量{np.max(S):.0f}, 最大时间步:{np.argmax(S)}")

m_idx = np.argmax(S)
s = S[:m_idx + 2]
t1 = time.process_time()
para_range = [[1e-5, 0.1], [1e-5, 0.8], [sum(s), 10*sum(s)]]
bassest = BassEstimate(s, para_range)
mse, P, Q, M = bassest.optima_search(c_n=100, threshold=10e-6)
r_2 = bassest.r2([P, Q, M])
print(f'Time elapsed: {(time.process_time() - t1):.2f}s')
print(f'P:{P:.4f}   Q:{Q:.4f}   M:{M:.0f}\nr^2:{r_2:.4f}')


fig = pl.figure(figsize=(10, 6), facecolor='grey')
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Steps")
ax.set_ylabel("Number of Adopters")
ax.plot(np.arange(1, len(S)+1), S, 'r-', lw=0.5)
ax.scatter(np.arange(1, len(S)+1), S, s=50, marker='o', c='k')
text = ax.text(20, np.max(S) * 0.1, f"p={p:.4f}, q={q:.4f}, m={nx.number_of_nodes(g)}\nP={P:.4f}, Q={Q:.4f}, M={M:.0f}", 
                bbox=dict(facecolor='grey', alpha=0.5))

#%%
ws0["param_boundary"]

#%%
t1 = time.perf_counter()
print("ws_0")
p_cont = (0.0005, 0.03)
q_cont = (0.004, 0.008)
delta = (0.0005, 0.0005)
ger_samp = Gen_para(g=g, p_cont=p_cont, q_cont=q_cont, delta=delta)
bound = ger_samp.identify_range()
print(f'time: {time.perf_counter() - t1:.2f}s')

#%%
prj.find_one({"_id":'watts_strogatz_graph(10000,6,0)'},
            projection={"param_boundary":1})

#%%
import datetime
param_boundary = {'p_range': [0.001, 0.02],
                  'q_range': [0.1, 0.2],
                  'P_range': [0.006, 0.0415],
                  'Q_range': [0.1322, 0.6976],
                  'ctime': datetime.datetime.now()}

#%%
prj.update_one({"_id":'watts_strogatz_graph(10000,6,0)'},
                {"$set": {'param_boundary': param_boundary}})


#%% [markdown]
## 生成扩散数据

#%%
def func(p, q, g):
    p = round(p, 5)
    q = round(q, 5)
    diff = Diffuse(p, q, g=g, num_runs=35)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return str([p, q]), list(np.concatenate(([p, q], x)))

#%%
r_p = param_boundary['p_range']
r_q = param_boundary['q_range']
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

print(f'Time: {(time.perf_counter() - t1):.2f} s')
diffuse_curves = dict(data)

#%%
diffuse_curves.keys()

#%%
prj.update_one({"_id": 'watts_strogatz_graph(10000,6,0)'}, 
                {"$set": {"diffuse_curves": diffuse_curves}}, upsert=True)

#%%[markdown]
# 拟合扩散数据

#%%
import bassestimate as eb
def func(x, n=1):
    p, q = x[:2]
    s_full = x[2:]
    max_ix = np.argmax(s_full)
    s = s_full[:max_ix + 1 + n]
    bassest = eb.BassEstimate(s)
    _, P, Q, M = bassest.optima_search(c_n=200, threshold=10e-6)
    r_2 = bassest.r2([P, Q, M])
    return [round(p, 5), round(q, 5), r_2, P, Q, M]


#%%
all_data = prj.find_one({"_id": 'watts_strogatz_graph(10000,6,0)'})
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

print(f"Time elapsed {(time.perf_counter() - t1):.2f}s")
#%%
d_dict

#%%
prj.update_one({"_id": 'watts_strogatz_graph(10000,6,0)'}, 
               {"$set": {"estimates": {"ctime": datetime.datetime.now(), **d_dict}}}, upsert=True)


#%%[markdown]
# 预测扩散数据

#%%
from bassestimate import BassForecast
def func(x, n=3):
    p, q = x[:2]
    s_full = x[2:]
    b_idx = np.argmax(s_full) - 1  # 巅峰扩散前1步
    e_idx = np.argmax(s_full) + 3  # 巅峰扩散后3步
    bass_fore = BassForecast(s_full, n=n, b_idx=b_idx, e_idx=e_idx) 
    res = bass_fore.run()   # [1步向前, 3步向前预测]
    return [round(p, 5), round(q, 5), res[0], res[1]]

#%%
all_data = prj.find_one({"_id": 'watts_strogatz_graph(10000,6,0)'})
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

print(f"Time elapsed {(time.perf_counter() - t1):.2f}s")
#%%
d_dict

#%%
prj.update_one({"_id": 'watts_strogatz_graph(10000,6,0)'}, 
                {"$set": {"forecasts": {
                    "ctime": datetime.datetime.now(), **d_dict}}}, upsert=True)

#%%
