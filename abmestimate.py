# coding=utf-8
from abmdiffuse import Diffuse
from bassestimate import BassEstimate
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import networkx as nx
import time


class estimateABM:
    num_conds = 2  # 构建网格的节点个数

    def __init__(self, s, intv_p=0.0005, intv_q=0.005, G=nx.gnm_random_graph(10000, 30000), m_p=True):
        self.s = s
        self.s_len = len(s)
        self.intv_p = intv_p
        self.intv_q = intv_q
        self.G = G
        self.k = nx.number_of_edges(self.G) / nx.number_of_nodes(self.G)
        self.m_p = m_p  # 传给Diffuse，确定是否多进程

    def __repr__(self):
        return "<G: {0.G!r} s_len={0.s_len} d_p={0.intv_p} d_q={0.intv_q}>".format(self)

    def r2(self, f_act):
        f_act = np.array(f_act)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        return (ssl - tse)/ssl

    def get_M(self, p, q):  # 获取对应扩散率曲线的最优潜在市场容量
        if p <= 0:
            raise Exception(f"p={p}小于0!")
        elif q <= 0:
            raise Exception(f"q={q}小于0!")
        else:
            diffu = Diffuse(p, q, g=self.G, num_runs=self.s_len, multi_proc=self.m_p)
            s_estim = diffu.repete_diffuse()
            x = np.mean(s_estim, axis=0)
            a = np.sum(np.square(x)) / np.sum(self.s)  # 除以np.sum(self.s)是为减少a的大小
            b = -2*np.sum(x*self.s) / np.sum(self.s)
            c = np.sum(np.square(self.s)) / np.sum(self.s)
            mse = np.sqrt(sum(self.s) * (4*a*c - b**2) / (4*a*self.s_len))
            sigma = -b/(2*a)
            m = sigma*self.G.number_of_nodes()
            return [mse, p, q, m], list(x*sigma)

    def gener_grid(self, p, q):
        p, q = round(p, 5), round(q, 5)
        temp = {(round(p - self.intv_p, 5) if p > self.intv_p else round(p/2, 5), 
                    round(q - self.intv_q, 5) if q > self.intv_q else round(q/2, 5)),
                (round(p, 5),
                    round(q - self.intv_q, 5) if q > self.intv_q else round(q/2, 5)),
                (round(p + self.intv_p, 5),
                    round(q - self.intv_q, 5) if q > self.intv_q else round(q/2, 5)),
                (round(p - self.intv_p, 5) if p > self.intv_p else round(p/2, 5), 
                    round(q, 5)),
                (round(p, 5), round(q, 5)),
                (round(p + self.intv_p, 5), round(q, 5)),
                (round(p - self.intv_p, 5) if p > self.intv_p else round(p/2, 5), 
                    round(q + self.intv_q, 5)),
                (round(p, 5), round(q + self.intv_q, 5)),
                (round(p + self.intv_p, 5), round(q + self.intv_q, 5))
                }
        return temp

    def gener_init_pq(self):  # 生成初始搜索点(p0,q0)
        rgs = BassEstimate(self.s)
        P0, Q0 = rgs.optima_search()[1 : 3]  # SABM最优点（P0,Q0）
        p_range = np.linspace(0.4*P0, P0, num=3)
        q_range = np.linspace(0.2*Q0/self.k, 0.6*Q0/self.k, num=3)
        to_fit = {}
        params_cont = []
        for p in p_range:  # 取9个点用于确定参数与估计值之间的联系
            for q in q_range:
                diffu = Diffuse(p, q, g=self.G, num_runs=self.s_len, multi_proc=self.m_p)
                s_estim = diffu.repete_diffuse()
                s_estim_avr = np.mean(s_estim, axis=0)
                rgs_1 = BassEstimate(s_estim_avr)
                P, Q = rgs_1.optima_search()[1: 3]
                params_cont.append([p, q, P, Q])

        to_fit = pd.DataFrame(params_cont, columns=['p', 'q', 'P', 'Q'])
        result_p = smf.ols('p~P+Q-1', data=to_fit).fit()
        result_q = smf.ols('q~P+Q-1', data=to_fit).fit()

        p0 = result_p.params['P']*P0 + result_p.params['Q']*Q0
        q0 = result_q.params['P']*P0 + result_q.params['Q']*Q0
        return round(p0, 5), round(q0, 5)   # 保留5位小数位，防止出错

    def solution_search(self, p0, q0):
        solution_list = []
        pq_set = self.gener_grid(p0, q0)
        pq_trace = [pq_set.copy()]  # 初始化(p, q)的搜索轨迹，set().copy()得到新的对象
        for p, q in pq_set:
            try:
                solution = self.get_M(p, q)
                solution_list.append(solution)  # ([mse, p, q, s_M], [扩散曲线])
            except Exception:
                print(f"p:{p}, q:{q}, 不搜索！")

        best_solution = sorted(solution_list)[:self.num_conds]  # 选取num_conds个候选点
        condidate_points = [(z[0][1], z[0][2]) for z in best_solution]
        his_cond = [condidate_points, ]
        i = 0
        while True:
            i += 1
            pq_set2 = set()
            for z in condidate_points:  # 可以取多个作为候选最优解
                temp = self.gener_grid(z[0], z[1])
                pq_set2.update(temp)

            new_points = pq_set2 - pq_set  # 集合减, 未包含在pd_set中的新(p, q)
            print(f"第{i}轮, 新增点个数:{len(new_points)}")
            if len(new_points) == 0:
                break
            else:
                pq_trace.append(new_points)  # 将新增加的点添加到pq_trace中
                for y in new_points:
                    try:
                        solution = self.get_M(y[0], y[1])
                        solution_list.append(solution)
                    except Exception:
                        print(f"p:{p}, q:{q}, 不搜索！")

                best_solution = sorted(solution_list, key=lambda x: x[0][0])[: self.num_conds]
                condidate_points = [(z[0][1], z[0][2]) for z in best_solution]
                his_cond.append(condidate_points)
                opt_solution = best_solution[0]  # [mse, p, q, m], [扩散数据]
                opt_curve = opt_solution[1]  # [扩散数据]
                pq_set.update(new_points)

        R2 = self.r2(opt_curve)
        search_steps = len(pq_set)  # 搜索点的数量
        result = {'params': opt_solution[0][1:],  # 估计值 [p, q, m]
                  'fitness': R2,
                  'best_curve': opt_curve,  # 最优拟合曲线
                  'num_nodes': search_steps,    # 搜索点的数量
                  'path': pq_trace,  # [{(p, q),}, ]的搜索轨迹
                  'his_cond': his_cond,  # 候选点历史
                  'his_data': solution_list}
        return result


if __name__ == '__main__':
    import pylab as pl
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    data_set = {'room air conditioners':(np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586,
                                                               1673, 1800, 1580, 1500]),
            'color televisions':(np.arange(1963,1971),[747,1480,2646,5118,5777,5982,5962,4631]),
            'clothers dryers':(np.arange(1949,1962),[106,319,492,635,737,890,1397,1523,1294,1240,1425,1260,1236]),
            'ultrasound':(np.arange(1965,1979),[5,3,2,5,7,12,6,16,16,28,28,21,13,6]),
            'mammography':(np.arange(1965,1979),[2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15,6,5,1]),
            'foreign language':(np.arange(1952,1964),[1.25,0.77,0.86,0.48,1.34,3.56,3.36,6.24,5.95,6.24,4.89,0.25]),
            'accelerated program':(np.arange(1952,1964),[0.67,0.48,2.11,0.29,2.59,2.21,16.80,11.04,14.40,
                                                         6.43,6.15,1.15])}

    china_set = {'color televisions': (np.arange(1997,2013),[2.6,1.2,2.11,3.79,3.6,7.33,7.18,5.29,8.42,5.68,6.57,5.49,
                                                            6.48,5.42,10.72,5.15]),
             'mobile phones':(np.arange(1997,2013),[1.7,1.6,3.84,12.36,14.5,28.89,27.18,21.33,25.6,15.88,12.3,6.84,
                                                    9.02,7.82,16.39,7.39]),
             'computers':(np.arange(1997,2013),[2.6,1.2,2.11,3.79,3.6,7.33,7.18,5.29,8.42,5.68,6.57,5.49,6.48,5.42,
                                                10.72,5.15]),
             'conditioners':(np.arange(1992,2013),[1.19,1.14,2.67,3.09,3.52,4.68,3.71,4.48,6.32,5.0,15.3,10.69,8.01,
                                                   10.87,7.12,7.29,5.2,6.56,5.23,9.93,4.81]),
             'water heaters':(np.arange(1988,2013),[28.07,8.4,5.86,6.37,3.9,4.08,5.42,4.12,3.45,3.31,3.12,1.64,2.36,
                                                    1.8,5.48,1.35,1.47,0.52,1.03,3.28,-1.4,1.72,1.26,0.62,1.25])
             }

    txt = "mobile phones"
    t1 = time.perf_counter()
    s = china_set[txt][1]
    year = china_set[txt][0]
    est_abm = estimateABM(s,)
    p0, q0 = est_abm.gener_init_pq()
    t2 = time.perf_counter()
    print(f"=========={txt}===========")
    print(f"第一阶段: {t2 - t1:.2f}秒")
    print(f'    p0:{p0:.5f}, q0:{q0:.5f}')
    
    result = est_abm.solution_search(p0, q0)
    t3 = time.perf_counter()
    print(f'第二阶段:: {t3 - t2:.2f}秒')
    print(f'一共用时: {t3 - t1:.2f}秒')
    print(f"R2:{result['fitness']:.4f}    num_nodes:{result['num_nodes']}")

    pl.style.use('ggplot')
    fig = pl.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Year', fontsize=20)
    ax.set_ylabel('Number of Adopters', fontsize=20)
    ax.plot(year, result['best_curve'])
    ax.scatter(year, s, c='grey', s=30, alpha=0.5)
    ax.grid(False)
    pl.show()
