# coding=utf-8
import numpy as np
import networkx as nx
import time
import random
from abmdiffuse import Diffuse


if __name__ == '__main__':
    import pylab as pl
    t1 = time.perf_counter()
    p, q = 0.001, 0.05
    diffu = Diffuse(p=p, q=q, sigma=0.1)
    diffu_cont = diffu.repete_diffuse(repetes=20)
    print(f"参数设置: p--{p}, q--{q} sigma--{diffu.sigma}")
    print(f"用时{time.perf_counter() - t1:.2f}秒")
    fig = pl.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    for line in diffu_cont:
        ax.plot(line, 'k-', lw=0.5, alpha=0.5)
    ax.plot(np.mean(diffu_cont, axis=0), 'r-', lw=2)
    pl.show()

