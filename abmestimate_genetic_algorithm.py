from abmdiffuse import Diffuse
from bassestimate import BassEstimate
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import networkx as nx

class GeneticAlgorithm:
    TOUR_SIZE = 2  # the number of individuals selected for tournament competitions 
    def __init__(self, fitne_func, indiv_bound, muta_rate=0.25, cros_rate=0.8, num_popul=200):
        self.muta_rate = muta_rate
        self.cros_rate = cros_rate
        self.num_popul = num_popul
        self.indiv_bound = indiv_bound
        self.indiv_dim = len(self.indiv_bound)
        self.fitne_func = fitne_func
        self.fitne_values = []
        self.last_iter = {"population": [], "fitness": []}
        self.population = []
        self.best_solu = []
        for _ in range(self.num_popul):
            self.population.append([(x[1] - x[0])*np.random.rand() + x[0] for x in self.indiv_bound])
    
    def eval_fitness(self):
        self.fitne_values = []
        for indiv in self.population:
            if indiv in self.last_iter["population"]:
                idx = self.last_iter["population"].index(indiv)
                self.fitne_values.append(self.last_iter["fitness"][idx])
            else:
                self.fitne_values.append(self.fitne_func(indiv))
    
    def selection(self):
        '''
        Implementation of binary tournament selection
        '''
        candi_father = np.random.randint(low=0, high=self.num_popul, size=self.TOUR_SIZE)
        candi_mother = np.random.randint(low=0, high=self.num_popul, size=self.TOUR_SIZE)
        return (np.argmin([self.fitne_values[i] for i in candi_father]),
                np.argmin([self.fitne_values[i] for i in candi_mother]))
    
    def crossover(self, i, j):
        '''
        Interchange gene of two selected individuals according to the probability of crossover
        '''
        father, mother = self.population[i], self.population[j]
        if np.random.rand() < self.cros_rate:
            div_dim = np.random.randint(low=0, high=self.indiv_dim-1)  # choose the gene
            indiv_1 = father[:div_dim+1] + mother[div_dim+1:]
            indiv_2 = mother[:div_dim+1] + father[div_dim+1:]
            return indiv_1, indiv_2
        else:
            return father, mother
        
    
    def mutation(self, indiv):
        '''
        Change the gene of an individual according the probability of mutation
        '''
        if np.random.rand() < self.muta_rate:
            dim = np.random.randint(low=0, high=self.indiv_dim)  # choose one gene to mutate
            indiv[dim] = (self.indiv_bound[dim][1] - self.indiv_bound[dim][0])*np.random.rand() + self.indiv_bound[dim][0]
        
        return indiv
    
        
    def reproduction(self):
        '''
        Generate new generation: including selection, crossover and mutation
        '''
        new_generation = []
        new_fitne_values = []
        while len(new_generation) < self.num_popul:
            i, j = self.selection()
            indiv_1, indiv_2 = self.crossover(i, j)
            self.mutation(indiv_1)
            self.mutation(indiv_2)
            
            if indiv_1 not in new_generation:
                new_generation.append(indiv_1)
            if indiv_2 not in new_generation:
                new_generation.append(indiv_1)
        
        self.population = new_generation
        self.fitne_values = new_fitne_values
    
    def evolution(self, num_generation=10, is_print=False):
        v = 0
        while True:
            self.eval_fitness()
            best_solu_idx = np.argmin(self.fitne_values)
            self.best_solu.append([self.fitne_values[best_solu_idx], self.population[best_solu_idx]])
            if is_print:
                print(f"Generation {v}, the best fitness value is {self.fitne_values[best_solu_idx]}, the best individual is{self.population[best_solu_idx]}")
            v += 1
            if v == num_generation:
                break
            
            self.last_iter = {'population': self.population[:], 'fitness': self.fitne_values[:]}  # record the last iteration infos
            self.reproduction()


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


if __name__ == "__main__":
    import time
    from pymongo import MongoClient
    client = MongoClient()
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
    # the bass diffusion model
    # def f(params, T):
    #     p, q, m = params
    #     t_list = np.arange(1, T + 1)
    #     a = 1 - np.exp(-(p + q) * t_list)
    #     b = 1 + q / p * np.exp(-(p + q) * t_list)
    #     diffu_cont = m * a / b
    #     adopt_cont = np.array(
    #         [diffu_cont[i] if i == 0 else diffu_cont[i] - diffu_cont[i - 1] for i in range(T)]
    #     )
    #     return adopt_cont
    # # mse for the bass model
    # def mse(params, S=S):
    #     a = f(params, len(S))
    #     sse = np.sum(np.square(S - a))
    #     return np.sqrt(sse) / len(S)  # 均方误

    # def r2(func, params, S=S):  # 
    #     f_act = func(params, len(S))
    #     tse = np.sum(np.square(S - f_act))
    #     mean_y = np.mean(S)
    #     ssl = np.sum(np.square(S - mean_y))
    #     R_2 = (ssl - tse) / ssl
    #     return R_2
    
    # agent-based diffusion model
    def fitness(individual, S=S):
        p, q = individual
        T = len(S)
        if p <= 0:
            raise Exception(f"p={p}小于0!")
        elif q <= 0:
            raise Exception(f"q={q}小于0!")
        else:
            diffuse = Diffuse(p, q, num_runs=T, multi_proc=True)
            diffuse_cont = diffuse.repete_diffuse(repetes=10)
            X = np.mean(diffuse_cont, axis=0)
            a = np.sum(np.square(X)) / np.sum(S)  # 除以np.sum(S)是为减少a的大小
            b = -2*np.sum(X*S) / np.sum(S)
            c = np.sum(np.square(S)) / np.sum(S)
            mse = np.sqrt(np.sum(S) * (4*a*c - b**2) / (4*a*T))
            # sigma = -b/(2*a)  # 倍数
        return mse
        # return {'mse': mse, 'estimates': [p, q, m], 'curve': list(X*sigma)}
    
    def r2_mse_abm(params, S=S):
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
    
    # t1 = time.perf_counter()
    # ga = GeneticAlgorithm(fitne_func=mse, indiv_bound=([(1e-7, 0.2), (1e-4, 1), (np.sum(S), 5*np.sum(S))]),
    #                       muta_rate=0.25, cros_rate=0.8, num_popul=100)
    # ga.evolution(num_generation=100, is_print=False)
    # best_sol = sorted(ga.best_solu, key=lambda x: x[0])[0]
    # r_2 = r2(f, best_sol[1], S)
    # print(f"Time elapsed {time.perf_counter() - t1:.4f}s")
    # print(f"The best individual, mse: {best_sol[0]:.4f}")
    # print(f"p: {best_sol[1][0]: .5f}, q: {best_sol[1][1]: .5f}, m: {best_sol[1][2]: .2f}")
    # print(f"r^2:{r_2:.4f}")
    
    # ga = GeneticAlgorithm(fitne_func=fitness, indiv_bound=([(1e-7, 0.2), (1e-4, 1)]),
    #                       muta_rate=0.25, cros_rate=0.8, num_popul=100)
    # ga.evolution(num_generation=2, is_print=False)
    # best_sol = sorted(ga.best_solu, key=lambda x: x[0])[0]
    # r_2, mse_ = r2_mse_abm(best_sol[1], S=np.array(S))
    
    # print(f"Time elapsed {time.perf_counter() - t1:.4f}s")
    # print(f"Best individual, mse: {best_sol[0]:.4f}, p: {best_sol[1][0]: .5f}, q: {best_sol[1][1]: .5f}")
    # print(f"r^2:{r_2:.4f}, mse: {mse_:.4f}")
    
    best_sol_cont = []
    for i in range(10):
        t1 = time.perf_counter()
        ga = GeneticAlgorithm(fitne_func=fitness, indiv_bound=([(1e-7, 0.2), (1e-4, 1)]),
                          muta_rate=0.25, cros_rate=0.8, num_popul=100)
        ga.evolution(num_generation=100, is_print=False)
        best_sol = sorted(ga.best_solu, key=lambda x: x[0])[0]
        r_2, mse_, sigma = r2_mse_abm(best_sol[1], S=S)
        p, q, m = best_sol[1][0], best_sol[1][1], 10000*sigma
        proj.insert_one({'algorithm': 'GA-0', 'res': [r_2, mse_, p, q, m]})
        print("======================================================")
        print(f"Repetition {i+1}")
        print(f"Time elapsed {time.perf_counter() - t1:.4f}s")
        print(f"Best individual, mse: {best_sol[0]:.4f}")
        print(f"p: {best_sol[1][0]: .5f}, q: {best_sol[1][1]: .5f}")
        print(f"r^2:{r_2:.4f}, mse: {mse_:.4f}\n")
        best_sol_cont.append([r_2, mse_, best_sol[1][0], best_sol[1][1], 10000*sigma])
    
    # study 2: 
    p0, q0 = gener_init_params(S)
    delta_p = 0.001
    delta_q = 0.005
    params_bound = [[max(1e-7, p0 - 20*delta_p), min(0.2, p0 + 20*delta_p)], 
                    [max(1e-4, q0 - 20*delta_q), min(1, q0 + 20*delta_q)]]
