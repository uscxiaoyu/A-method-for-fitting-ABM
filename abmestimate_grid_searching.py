from abmdiffuse import Diffuse
from bassestimate import BassEstimate
from bassestimate import BassEstimate
from pymongo import MongoClient
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import networkx as nx
import time

def gener_init_params(S, g=nx.gnm_random_graph(10000, 30000)):  # 生成初始搜索点(p0,q0)
    rgs = BassEstimate(S)
    P0, Q0 = rgs.optima_search()[1 : 3]  # BASS最优点（P0,Q0）
    p_range = np.linspace(0.3*P0, P0, num=3)
    q_range = np.linspace(0.06*Q0, 0.3*Q0, num=3)
    to_fit = {}
    params_cont = []
    for p in p_range:  # 取9个点用于确定参数与估计值之间的联系
        for q in q_range:
            diffu = Diffuse(p, q, g=g, num_runs=len(S), multi_proc=True)
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

def r2_mse_abm(params, S):
    p, q = params
    T = len(S)
    diffuse = Diffuse(p, q, num_runs=T, multi_proc=True)
    diffuse_cont = diffuse.repete_diffuse(repetes=10)
    X = np.mean(diffuse_cont, axis=0)
    a = np.sum(np.square(X)) / np.sum(S)  # 除以np.sum(S)是为减少a的大小
    b = -2*np.sum(X*S) / np.sum(S)
    c = np.sum(np.square(S)) / np.sum(S)
    sigma = -b/(2*a)  # 倍数
    mse = np.sqrt(np.sum(S) * (4*a*c - b**2) / (4*a*T))
    hat_S = list(X*sigma)
    tse = np.sum(np.square(S - hat_S))
    mean_y = np.mean(S)
    ssl = np.sum(np.square(S - mean_y))
    R_2 = (ssl - tse) / ssl
    return R_2, mse, sigma

if __name__ == "__main__":
    client = MongoClient('106.14.27.147')
    db = client.abmDiffusion
    proj = db.compaAlgorithms

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

    txt = "clothers dryers"
    t1 = time.perf_counter()
    S = np.array(data_set[txt][1])

    delta_p = 0.001
    delta_q = 0.005

    # study 1: grid searching without the hint from DEM
    # params_bound = [(1e-7, 0.1), (1e-4, 0.5)]

    # p_bound = params_bound[0]
    # q_bound = params_bound[1]

    # p_cont = np.arange(p_bound[0], p_bound[1], delta_p)
    # q_cont = np.arange(q_bound[0], q_bound[1], delta_q)

    # # proj.insert_one({'algorithm': 'GS-0', 'details': []})
    # params_cont = []
    # res_cont = []
    # i = 0
    # t1 = time.perf_counter()
    # for p in p_cont:
    #     for q in q_cont:
    #         i += 1
    #         r_2, mse_, sigma = r2_mse_abm([p, q], S)
    #         m = 10000 * sigma
    #         proj.update_one({'algorithm': 'GS-0'}, {'$addToSet': {'details': [r_2, mse_, p, q, m]}})
    #         res_cont.append([r_2, mse_, p, q, m])
    #         if i % 500 == 0:
    #             best_sol = sorted(res_cont, key=lambda x: x[1])[0]
    #             print("====================================================")
    #             print(f"{i}, time elasped: {time.perf_counter()-t1:.4f}")
    #             print(f"Best solution: r^2={best_sol[0]:.4f}, p={best_sol[2]:.4f}, q={best_sol[3]:.4f}, m={best_sol[4]:.2f}")
                
    # best_sol = sorted(res_cont, key=lambda x: x[1])[0]
    # proj.update_one({'algorithm': 'GS-0'}, {'$set': {'res': best_sol}}, upsert=True)
    
    # study 2: grid searching with the hint from DEM
    for iter in range(10):
        p0, q0 = gener_init_params(S)
        params_bound = [[max(1e-7, p0 - 10*delta_p), min(0.1, p0 + 10*delta_p)], 
                        [max(1e-4, q0 - 10*delta_q), min(0.5, q0 + 10*delta_q)]]
        p_bound = params_bound[0]
        q_bound = params_bound[1]

        p_cont = np.arange(p_bound[0], p_bound[1], delta_p)
        q_cont = np.arange(q_bound[0], q_bound[1], delta_q)

        params_cont = []
        res_cont = []
        i = 0
        t1 = time.perf_counter()
        for p in p_cont:
            for q in q_cont:
                i += 1
                r_2, mse_, sigma = r2_mse_abm([p, q], S)
                m = 10000 * sigma
                res_cont.append([r_2, mse_, p, q, m])
                    
        best_sol = sorted(res_cont, key=lambda x: x[1])[0]
        print("====================================================")
        print(f"Iteration {iter}, time elasped: {time.perf_counter()-t1:.4f}")
        print(f"Best solution: r^2={best_sol[0]:.4f}, p={best_sol[2]:.4f}, q={best_sol[3]:.4f}, m={best_sol[4]:.2f}")
        proj.insert_one({'algorithm': 'GS-1', 'res': best_sol, 'grids': i})