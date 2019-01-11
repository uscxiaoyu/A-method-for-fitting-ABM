# coding=utf-8
#%%
import time
import numpy as np
import pylab as pl
import random
import pickle
from abmestimate import estimateABM
from matplotlib.ticker import ScalarFormatter
get_ipython().run_line_magic('pylab', 'inline')

#%%
f = open("plot_trace.txt", 'rb')
result = pickle.load(f)
p0, q0 = 0.01159, 0.06075

#%%
year = np.arange(1949, 1962)  # clothers dryers
s = [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]

#%%
random.seed(999)
np.random.seed(999)

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
print(f"Estimates:{result['params']}, r2:{result['fitness']:.4f}, number of nodes:{result['num_nodes']}")
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
year = np.arange(1949, 1962)
s = [106, 319, 492, 635, 737, 890, ]

#%%
# 绘图
pl.style.use('grayscale')
fig = pl.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Year', fontsize=20)
ax.set_ylabel('Number of Adopters', fontsize=20)
ax.plot(year, result['best_curve'], lw=2,
        color='red', label='Fitted curve of the ABM')
ax.scatter(year, s, c='grey', s=40, alpha=0.5, label='Empirical data')
ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=True))
ax.text(1949, 1000, "$R^2$=%.4f\n$\hat{p}$=%.5f,$\hat{q}$=%.5f,$\hat{m}$=%d" % (
    result["fitness"], result['params'][0], result['params'][1], result['params'][2]), fontsize=12)
ax.grid(False)
ax.legend(loc='upper left', fontsize=15)

inset_ax1 = fig.add_axes([0.575, 0.2, 0.15, 0.25], facecolor='#FCFAF2')
inset_ax2 = fig.add_axes([0.73, 0.2, 0.15, 0.25], facecolor='#FCFAF2')
ax_list = [inset_ax1, inset_ax2]

for i in range(len(his_cond)):
    ax = ax_list[i]
    pq_set = set()
    for j in range(i+1):
        pq_set.update(result['path'][j])

    for z in pq_set:
        if z == (p0, q0):
            ax.scatter(z[0], z[1], s=60, c='w',
                       edgecolors='k', marker='o', alpha=0.5)
        elif z in his_cond[i]:
            if z == best_solution and i == len(his_cond) - 1:
                ax.scatter(z[0], z[1], s=60, c='r', marker='*')
            else:
                ax.scatter(z[0], z[1], s=30, c='r', marker='^', alpha=1)
        elif z in result['path'][i]:
            ax.scatter(z[0], z[1], s=30, c='k', marker='s', alpha=1)
        else:
            ax.scatter(z[0], z[1], s=30, c='k', marker='s', alpha=0.2)

    ax.set_xlim([0.01, 0.015])
    ax.set_ylim([0.05, 0.08])
    ax.set_title('Iteration %s' % (i+1), fontsize=15)
    ax.set_xlabel('p')
    if i == 0:
        ax.set_ylabel('q')
    ax.set_xticks([])
    ax.set_yticks([])


#%%
p_q_cont


#%%
his_cond

#%%
result['path'][0]

#%%
f = open("plot_trace.txt", 'wb')
pickle.dump(result, f)
f.close()

#%%
res['path']
