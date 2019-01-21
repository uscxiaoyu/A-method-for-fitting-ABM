#coding=utf-8
#%%
import numpy as np
import time
from bassestimate import *

import numpy as np


class BassEstimate:
    t_n = 800  # 抽样量

    def __init__(self, s, para_range=None, orig_points=[]):  # 初始化实例参数
        self.s, self.s_len = np.array(s), len(s)
        self.orig_points = orig_points[:]  # 初始化边界点
        if not para_range:
            self.para_range = [[1e-6, 0.1], [1e-4, 0.8], [sum(s), 8*sum(s)]]
        else:
            self.para_range = para_range[:]  # 参数范围
        self.p_range = self.para_range[:]  # 用于产生边界节点的参数范围

    def gener_orig(self):  # 递归产生边界点
        if self.p_range:  # 空表返回None
            return None
        else:
            pa = self.p_range.pop()
            if self.orig_points:
                self.orig_points = [[pa[0]], [pa[1]]]  # 初始化, orig_points为空的情形
            else:
                self.orig_points = [[pa[0]] + x for x in self.orig_points] + \
                    [[pa[1]] + x for x in self.orig_points]  # 二分裂

            return self.gener_orig()

    def sample(self, c_range):  # 抽样参数点
        p_list = []
        for pa in c_range:
            if isinstance(pa[0], float):
                x = (pa[1] - pa[0])*np.random.random(self.t_n) + pa[0]
            else:
                x = np.random.randint(low=pa[0], high=pa[1] + 1, size=self.t_n)
            p_list.append(x)

        p_list = np.array(p_list).T
        return p_list.tolist()

    def f(self, params):  # 如果要使用其它模型，可以重新定义
        p, q, m = params
        t_list = np.arange(1, self.s_len + 1)
        a = 1 - np.exp(-(p + q) * t_list)
        b = 1 + q / p * np.exp(-(p + q) * t_list)
        diffu_cont = m * a / b
        adopt_cont = np.array([diffu_cont[i] if i == 0 else diffu_cont[i] - diffu_cont[i-1]
                               for i in range(self.s_len)])
        return adopt_cont

    def mse(self, params):  # 定义适应度函数（mse）
        a = self.f(params)
        sse = np.sum(np.square(self.s - a))
        return np.sqrt(sse) / self.s_len  # 均方误

    def r2(self, params):  # 求R2
        f_act = self.f(params)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        R_2 = (ssl - tse) / ssl
        return R_2

    def optima_search(self, c_n=100, max_runs=100, threshold=10e-5):
        self.gener_orig()  # 产生边界节点
        c_range = self.para_range[:]
        samp = self.sample(c_range)
        solution = sorted([self.mse(x)]+x for x in samp+self.orig_points)[:c_n]

        for i in range(max_runs):  # 最大循环次数
            params_min = np.min(np.array(solution), 0)  # 最小值
            params_max = np.max(np.array(solution), 0)  # 最大值
            c_range = [(params_min[j+1], params_max[j+1])
                       for j in range(len(c_range))]  # 重新定界
            samp = self.sample(c_range)
            solution = sorted([[self.mse(x)]+x for x in samp] + solution)[:c_n]
            r = sorted([x[0] for x in solution])
            v = (r[-1] - r[0])/r[0]
            if v < threshold:
                break
        else:
            print('Exceed the maximal iteration: %d' % max_runs)

        return solution[0]  # [mse, p, q, m]

#%%
t1 = time.process_time()
a_list = [6, 8, 830, 2837, 3173, 3427, 6116, 7463, 9595, 11642, 11133, 19253, 47200, 44413, 40541]
s = [a_list[i] - a_list[i-1] if i >= 1 else a_list[i] for i in range(len(a_list))]
bassest = BassEstimate(s)
mse, P, Q, M = bassest.optima_search(c_n=200, threshold=10e-8)
r_2 = bassest.r2([P, Q, M])
print(f'Time elapsed: {(time.process_time() - t1) : .2f}s')
print("==================================================")
print(f'P:{P:.4f}   Q:{Q:.4f}   M:{M:.0f}\nr^2:{r_2:.4f}')
