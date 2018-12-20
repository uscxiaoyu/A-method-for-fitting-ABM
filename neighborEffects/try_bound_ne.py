#%%
get_ipython().run_line_magic('pylab', 'inline')
from bassestimate import BassEstimate
from pymongo import MongoClient
from neighborEffects.generate_params_boundary_ne import *
import numpy as np
import pylab as pl
import networkx as nx
import datetime
import time

#%%
client = MongoClient('localhost', 27017)
db = client.abmDiffusion
prj = db.neighEffects
alpha_cont = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.3, 1.5]

#%%
i = 0
p, q, alpha = 0.005, 0.1, alpha_cont[i]
t1 = time.perf_counter()
diff = Diffuse_ne(p, q, alpha)
diff_cont = diff.repete_diffuse()
S = np.mean(diff_cont, axis=0)
print(f"Time elasped:{time.perf_counter()-t1:.2f}s")
print(f"邻居效应:{alpha_cont[i]}")
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
text = ax.text(20, np.max(S) * 0.1, f"{alpha_cont[i]}\np={p:.4f}, q={q:.4f}, m={nx.number_of_nodes(diff.g)}\nP={P:.4f}, Q={Q:.4f}, M={M:.0f}", 
                bbox=dict(facecolor='grey', alpha=0.5))


#%%
t1 = time.perf_counter()
print(alpha_cont[i])
p_cont = (0.0005, 0.03)
q_cont = (0.08, 0.12)
delta = (0.001, 0.001)
ger_samp = Gen_para_ne(alpha=alpha, p_cont=p_cont, q_cont=q_cont, delta=delta)
bound = ger_samp.identify_range()
print(f'time: {time.perf_counter() - t1:.2f}s')
print(bound)

#%% [markdown]
#### 插入数据 
#%%
prj.insert_one({"_id": alpha, "para_boundary": bound})

#%%
prj.find_one()

#%%[markdown]
#### 对[0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.3, 1.5]

#%%
for j, alpha in enumerate(alpha_cont[1:], start=1):
    t1 = time.perf_counter()
    print(j + 1, alpha)
    p_cont = (0.0005, 0.03)
    q_cont = (0.08*6**alpha, 0.12*6**alpha)
    delta = (0.001*6**alpha, 0.001*6**alpha)
    ger_samp = Gen_para_ne(alpha, p_cont=p_cont, q_cont=q_cont, delta=delta)
    bound = ger_samp.identify_range()
    prj.insert_one({"_id": alpha, "para_boundary": bound})
    print(f'  time: {time.perf_counter() - t1:.2f}s')
