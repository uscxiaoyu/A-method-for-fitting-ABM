# coding=utf-8
#%%
get_ipython().run_line_magic('pylab', 'inline')
import time
import numpy as np
import pylab as pl
from abm_estim_diffu_data import estimateABM

#%%
# clothers dryers
s = [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]
t = time.perf_counter()
est_abm = estimateABM(s)
p0, q0 = est_abm.gener_p0_q0()
result = est_abm.solution_search(p0, q0)
print(f'总时间: {time.perf_counter() - t:.2f} 秒')

#%%
p_q_cont = result['path']
best_solution = result['params'][:-1]  # p, q

#%%
p_range = [round(p0 + i * 0.001, 4) for i in range(-6, 18)]
q_range = [round(q0 + i * 0.005, 4) for i in range(-12, 12)]
x, y = np.meshgrid(p_range, q_range)

#%%
# 绘图
fig = pl.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('p', fontsize=15)
ax.set_ylabel('q', fontsize=15)
ax.set_xlim([min(p_range) - 0.001, max(p_range) + 0.001])
ax.set_ylim([min(q_range) - 0.005, max(q_range) + 0.005])

for p in p_range:
    for q in q_range:
        ax.scatter(p, q, s=15, c='k', marker='o', alpha=0.1)

for (p, q) in p_q_cont:
    if (p, q) == best_solution:
        ax.scatter(p, q, s=160, c='r', marker='*')
    elif (p, q) == (p0, q0):
        ax.scatter(p0, q0, s=100, c='b', marker='o')
    else:
        ax.scatter(p, q, s=40, c='k', marker='^', alpha=0.5)

inset_ax1 = fig.add_axes([0.55, 0.73, 0.15, 0.15], facecolor='#FCFAF2')
inset_ax2 = fig.add_axes([0.73, 0.73, 0.15, 0.15], facecolor='#FCFAF2')
inset_ax3 = fig.add_axes([0.55, 0.55, 0.15, 0.15], facecolor='#FCFAF2')
inset_ax4 = fig.add_axes([0.73, 0.55, 0.15, 0.15], facecolor='#FCFAF2')

pq = np.array(result['his_path'][0])
inset_ax1.scatter(pq[:, 0], pq[:, 1], s=15, c='k', marker='o', alpha=0.2)

inset_ax1.set_xlim([0.005, 0.015])
inset_ax1.set_ylim([0.05, 0.088])
inset_ax1.set_xlabel('Iteration 1', fontsize=10)
inset_ax1.set_xticks([])
inset_ax1.set_yticks([])

i = 1
for ax in [inset_ax2, inset_ax3, inset_ax4]:
    pq0 = result['his_path'][i-1]
    pq1 = result['his_path'][i]
    temp = []
    for z in pq1:
        if z in pq0:
            ax.scatter(z[0], z[1], s=15, c='k', marker='o', alpha=0.2)
        else:
            ax.scatter(z[0], z[1], s=15, c='k', marker='o', alpha=1)
    solution = result['his_best'][i-1]
    for z in solution:
        ax.scatter(z[0], z[1], s=15, c='r', marker='s', alpha=1)

    ax.set_xlim([0.005, 0.015])
    ax.set_ylim([0.05, 0.088])
    ax.set_xlabel('Iteration %s' % (i+1), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    i = i + 1
