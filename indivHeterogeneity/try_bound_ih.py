#%%
get_ipython().run_line_magic('pylab', 'inline')
from abmdiffuse import Diffuse
from bassestimate import BassEstimate
from pymongo import MongoClient
from generate_params_boundary import Gen_para
import numpy as np
import pylab as pl
import networkx as nx
import datetime
import time


#%%
client = MongoClient('localhost', 27017)
db = client.abmDiffusion
prj = db.indivHeter

#%%
def func(p, q, sigma):
    diff = Diffuse(p, q, sigma=sigma, num_runs=40)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))

#%%[markdown]
#### 扩散示例

#%%
sigma = 0.1
p, q = 0.001, 0.05
t1 = time.perf_counter()
diff = Diffuse(p, q, sigma=sigma)
diff_cont = diff.repete_diffuse()
S = np.mean(diff_cont, axis=0)
print(f"Time elasped:{time.perf_counter()-t1:.2f}s")
print(f"网络:{sigma}")
print(f"最大采纳量{np.max(S):.0f}, 最大时间步:{np.argmax(S)}")

#%%
m_idx = np.argmax(S)
s = S[:m_idx + 2]
t1 = time.process_time()
para_range = [[1e-5, 0.1], [1e-5, 0.8], [sum(s), 10*sum(s)]]
bassest = BassEstimate(s, para_range)
mse, P, Q, M = bassest.optima_search(c_n=100, threshold=10e-6)
r_2 = bassest.r2([P, Q, M])
print(f'Time elapsed: {(time.process_time() - t1):.2f}s')
print(f'P:{P:.4f}   Q:{Q:.4f}   M:{M:.0f}\nr^2:{r_2:.4f}')

#%%
fig = pl.figure(figsize=(10, 6), facecolor='grey')
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Steps")
ax.set_ylabel("Number of Adopters")
ax.plot(np.arange(1, len(S)+1), S, 'r-', lw=0.5)
ax.scatter(np.arange(1, len(S)+1), S, s=50, marker='o', c='k')
text = ax.text(20, np.max(S) * 0.1, f"{sigma}\np={p:.4f}, q={q:.4f}, m={nx.number_of_nodes(diff.g)}\nP={P:.4f}, Q={Q:.4f}, M={M:.0f}", 
               bbox=dict(facecolor='grey', alpha=0.5))

#%%
t1 = time.perf_counter()
print(sigma)
p_cont = (0.0005, 0.03)
q_cont = (0.004, 0.008)
sigma = 0.1
ger_samp = Gen_para(p_cont=p_cont, q_cont=q_cont, sigma=sigma)
bound = ger_samp.identify_range()
print(f'time: {time.perf_counter() - t1:.2f}s')

#%%
bound

#%%
prj.insert_one({'_id':sigma, 'param_boundary':{"ctime": datetime.datetime.now(), **bound}})

#%%
sigma_cont = [0.2, 0.4]
for j, sigma in enumerate(sigma_cont):
    t1 = time.perf_counter()
    print(j+1, f"sigam:{sigma}")
    p_cont = (0.0005, 0.025)
    q_cont = (0.04, 0.1)  # 小心设置
    delta = (0.0001, 0.005)
    ger_samp = Gen_para(sigma=sigma, p_cont=p_cont, q_cont=q_cont, delta=delta)
    bound = ger_samp.identify_range()
    prj.insert_one({"_id": sigma, "param_boundary": bound})
    print(f'  time: {time.perf_counter() - t1:.2f}s')


#%%
list(prj.find())

#%%

#%%
get_ipython().system('mongodump -d abmDiffusion -o "./"')


