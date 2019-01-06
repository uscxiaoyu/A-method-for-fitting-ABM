# coding=utf-8
#%%
import time
import numpy as np
import pylab as pl
import random
from abm_estim_diffu_data import estimateABM
get_ipython().run_line_magic('pylab', 'inline')

#%%
random.seed(999)
np.random.seed(999)

year = np.arange(1949, 1962)  # clothers dryers
s = [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]
t1 = time.perf_counter()
est_abm = estimateABM(s, m_p=True)
p0, q0 = est_abm.gener_init_pq()
t2 = time.perf_counter()
print(f"第一阶段: {t2 - t1:.2f}秒")
print(f'    p0:{p0:.5f}, q0:{q0:.5f}')

result = est_abm.solution_search(p0, q0)
t3 = time.perf_counter()
print(f'第二阶段:: {t3 - t2:.2f}秒')
print(f'一共用时: {t3 - t1:.2f}秒')
print(f"R2:{result['fitness']:.4f}    num_nodes:{result['num_nodes']}")

#%%
result.keys()

#%%
print(f"Estimates:{result['params']}, r2:{result['fitness']}, number of nodes:{result['num_nodes']}")

#%%
pl.xlabel('Year')
pl.ylabel('Number of Adopters')
pl.plot(year, result['best_curve'], lw=2)
pl.scatter(year, s, c='grey', s=20)

#%%
p_range = [round(p0 + i * 0.001, 4) for i in range(-6, 18)]
q_range = [round(q0 + i * 0.005, 4) for i in range(-12, 12)]
x, y = np.meshgrid(p_range, q_range)

#%%
p_q_cont = []
for x in result['path']:
    p_q_cont += list(x)

best_solution = tuple(result['params'][:-1])  # p, q
his_cond = result['his_cond']

#%%
# 绘图
fig = pl.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('p', fontsize=15)
ax.set_ylabel('q', fontsize=15)
ax.set_xlim([min(p_range) - 0.001, max(p_range) + 0.001])
ax.set_ylim([min(q_range) - 0.005, max(q_range) + 0.005])

for p in p_range:
    for q in q_range:
        ax.scatter(p, q, s=5, c='k', marker='o', alpha=0.1)

for (p, q) in p_q_cont:
    if (p, q) == best_solution:
        ax.scatter(p, q, s=80, c='r', marker='*')
    elif (p, q) == (p0, q0):
        ax.scatter(p0, q0, s=50, c='b', marker='o')
    else:
        ax.scatter(p, q, s=20, c='k', marker='^', alpha=0.5)

inset_ax1 = fig.add_axes([0.55, 0.73, 0.15, 0.15], facecolor='#FCFAF2')
inset_ax2 = fig.add_axes([0.73, 0.73, 0.15, 0.15], facecolor='#FCFAF2')
#inset_ax3 = fig.add_axes([0.55, 0.55, 0.15, 0.15], facecolor='#FCFAF2')
#inset_ax4 = fig.add_axes([0.73, 0.55, 0.15, 0.15], facecolor='#FCFAF2')

for z in result['path'][0]:
    if z in result['his_cond'][0]:
        inset_ax1.scatter(z[0], z[1], s=5, c='r', marker='o', alpha=1)
    else:
        inset_ax1.scatter(z[0], z[1], s=5, c='k', marker='o', alpha=1)

inset_ax1.set_xlim([0.01, 0.015])
inset_ax1.set_ylim([0.05, 0.088])
inset_ax1.set_xlabel('Iteration 1', fontsize=10)
inset_ax1.set_xticks([])
inset_ax1.set_yticks([])

ax_list = [inset_ax1, inset_ax2]#, inset_ax3, inset_ax4]

for i in range(1, len(his_cond)):
    ax = ax_list[i]
    pq0 = set()
    for j in range(i):
        pq0.update(result['path'][j])

    pq1 = result['path'][i]
    for z in pq0:
        ax.scatter(z[0], z[1], s=5, c='k', marker='o', alpha=0.2)

    for z in pq1:
        if z not in his_cond[i]:
            ax.scatter(z[0], z[1], s=5, c='k', marker='o', alpha=1)

    for z in his_cond[i]:
        ax.scatter(z[0], z[1], s=5, c='r', marker='s', alpha=1)

    ax.set_xlim([0.01, 0.015])
    ax.set_ylim([0.05, 0.088])
    ax.set_xlabel('Iteration %s' % (i+1), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


#%%
p_q_cont


#%%
his_cond

#%%
result['path'][0]

#%%
import pickle

f = open("plot_trace.txt", 'wb')
pickle.dump(result, f)
f.close()

#%%
f = open("plot_trace.txt", 'rb')
res = pickle.load(f)

#%%
res['path']