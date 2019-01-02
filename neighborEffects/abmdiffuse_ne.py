# coding=utf-8
import numpy as np
import networkx as nx
import time
import random
from abmdiffuse import Diffuse

if __name__ == '__main__':
    t1 = time.perf_counter()
    p, q, alpha = 0.01, 0.5, 1
    diffu = Diffuse(p, q, alpha=alpha)
    diffu_cont = diffu.repete_diffuse(repetes=10)
    print(f"参数设置: p--{p}, q--{q}, alpha--{alpha} network--{diffu.g.number_of_nodes()}")
    print(f"用时{time.perf_counter() - t1:.2f}秒")
