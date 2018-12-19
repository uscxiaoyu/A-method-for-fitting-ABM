# -*- coding: utf-8 -*-
import numpy as np
import time


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
                x = (pa[1] - pa[0]) * np.random.random(self.t_n) + pa[0]
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
            c_range = [(params_min[j+1], params_max[j+1]) for j in range(len(c_range))]  # 重新定界
            samp = self.sample(c_range)
            solution = sorted([[self.mse(x)]+x for x in samp] + solution)[:c_n]
            r = sorted([x[0] for x in solution])
            v = (r[-1] - r[0])/r[0]
            if v < threshold:
                break
        else:
            print('Exceed the maximal iteration: %d' % max_runs)

        return solution[0]  # [mse, p, q, m]


class BassForecast:
    def __init__(self, s, n, b_idx, e_idx):
        self.s, self.n = s, n
        self.s_len = len(s)
        self.b_idx = b_idx  # 开始拟合的索引
        self.e_idx = min(e_idx, self.s_len - 1)  # 结束拟合的索引

    def f(self, params):  # 如果要使用其它模型，可以重新定义
        p, q, m = params
        t_list = np.arange(1, self.s_len + 1)
        a = 1 - np.exp(-(p + q)*t_list)
        b = 1 + q/p*np.exp(-(p + q)*t_list)
        diffu_cont = m*a/b
        adopt_cont = np.array([diffu_cont[i] if i == 0 else diffu_cont[i] - diffu_cont[i-1]
                               for i in range(self.s_len)])
        return adopt_cont

    def predict(self):  # 返回b_idx到e_idx索引的扩散数据
        pred_cont = []
        for i in range(self.e_idx - self.b_idx):  # 拟合次数
            idx = self.b_idx + 1 + i
            x = self.s[:idx]
            para_range = [[1e-5, 0.1], [1e-5, 0.8], [sum(x), 5*sum(self.s)]]
            bass_est = BassEstimate(x, para_range)
            est = bass_est.optima_search()
            params = est[1:]  # est: [mse, p, q, m]
            pred_s = list(self.f(params))
            pred_cont.append(pred_s[idx:])
        self.pred_res = pred_cont

    def one_step_ahead(self):
        pred_cont = np.array([x[0] for x in self.pred_res])
        mad = np.mean(np.abs(pred_cont - self.s[self.b_idx + 1 : self.e_idx + 1]))
        mape = np.mean(np.abs(pred_cont - self.s[self.b_idx + 1 : self.e_idx + 1])/
                    self.s[self.b_idx + 1 : self.e_idx + 1])
        mse = np.mean(np.sqrt(np.sum(np.square(pred_cont - 
                    self.s[self.b_idx + 1: self.e_idx + 1]))))
        return [mad, mape, mse]

    def n_step_ahead(self):
        pred_cont = np.array([x[:self.n] for x in self.pred_res if self.n <= len(x)])
        act_cont = np.array([self.s[self.b_idx + i + 1 : self.b_idx + i + 1 + self.n] 
                                for i in range(len(pred_cont))])
        mad = np.mean(np.abs(pred_cont - act_cont))
        mape = np.mean(np.abs(pred_cont - act_cont) / act_cont)
        mse = np.mean(np.sqrt(np.sum(np.square(pred_cont - act_cont))))
        return [mad, mape, mse]

    def run(self):
        self.predict()
        one_cont = self.one_step_ahead()
        n_cont = self.n_step_ahead()
        return [one_cont, n_cont]


if __name__=='__main__':
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673,
                                                                  1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425,
                                                            1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24,
                                                             4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40,
                                                                6.43, 6.15, 1.15])}
    china_set = {'color tv': (np.arange(1997, 2013),[2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57,
                                                     5.49, 6.48, 5.42, 10.72, 5.15]),
                 'mobile phone': (np.arange(1997, 2013),[1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88,
                                                         12.3, 6.84, 9.02,
                                   7.82, 16.39, 7.39])}
    S = data_set['clothers dryers'][1]

    """
    m_idx = np.argmax(S)
    s = S[ : m_idx + 2]
    t1 = time.process_time()
    para_range = [[1e-5, 0.1], [1e-5, 0.8], [sum(s), 10*sum(s)]]
    bassest = BassEstimate(s, para_range)
    mse, P, Q, M = bassest.optima_search(c_n=100, threshold=10e-5)
    r_2 = bassest.r2([P, Q, M])
    print(f'Time elapsed: {(time.process_time() - t1) : .2f}s')
    print("==================================================")
    print(f'P:{P:.4f}   Q:{Q:.4f}   M:{M:.0f}\nr^2:{r_2:.4f}')
    """

    bass_fore = BassForecast(S, n=3, b_idx=6, e_idx=10)
    res = bass_fore.run()

    print('1步向前预测:', end=' ')
    print('MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[0])
    print('3步向前预测:', end=' ')
    print('MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[1])
    
