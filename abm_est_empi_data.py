from abmestimate import estimateABM
from pymongo import MongoClient
import networkx as nx
import numpy as np
import time


def generate_random_graph(degre_sequance):
    G = nx.configuration_model(degre_sequance, create_using=None, seed=None)
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G


def insert_data(g, g_name, key, data_set):
    s = data_set[key][1]
    res = prj.find_one({"_id": g_name}, projection={key: 1})
    if res:  # 查看有没有插入g_name，如果插入了，则查看是否存在res[key]
        len_res = len(res.get(key, []))
    else:
        len_res = 0
    while len_res < 10:
        print(f"第{len_res + 1}轮:")
        t1 = time.perf_counter()
        est_abm = estimateABM(s, G=g, m_p=True)
        p0, q0 = est_abm.gener_init_pq()
        result = est_abm.solution_search(p0, q0)
        t3 = time.perf_counter()
        print(f' 用时: {t3 - t1:.2f}秒')
        print(f'    p0:{p0:.5f}, q0:{q0:.5f}')
        print(f" R2:{result['fitness']:.4f}    num_nodes:{result['num_nodes']}")

        if len_res == 0:
            prj.insert_one({"_id": g_name, key: [str(result)]})  # result的字符串形式，为了和MongoDB兼容
        else:
            prj.update_one({"_id": g_name}, {"$push": {key: str(result)}})

        res = prj.find_one({"_id": g_name}, projection={key: 1})
        len_res = len(res.get(key, []))
    else:
        print(f"  任务已完成！")


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.abmEstimate
    data_set = {'room air conditioners': (np.arange(1949, 1962), 
                        [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971),
                        [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962),
                        [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236])}   
    np.random.seed(999)
    expon_seq = np.load('dataSources/exponential_sequance.npy')
    gauss_seq = np.load('dataSources/gaussian_sequance.npy')
    logno_seq = np.load('dataSources/lognormal_sequance.npy')
    facebook_graph = nx.read_gpickle('dataSources/facebook.gpickle')
    epinions_graph = nx.read_gpickle('dataSources/epinions.gpickle')  
    g_name_cont = ['barabasi_albert_graph(10000,3)', 'gnm_random_graph(10000,30000)', 
                'watts_strogatz_graph(10000,6,0)', 'watts_strogatz_graph(10000,6,0.1)',
                'watts_strogatz_graph(10000,6,0.3)', 'watts_strogatz_graph(10000,6,0.5)',
                'watts_strogatz_graph(10000,6,0.7)', 'watts_strogatz_graph(10000,6,0.9)',
                'watts_strogatz_graph(10000,6,1.0)',
                'exponential_graph(10000,3)', 'gaussian_graph(10000,3)', 
                'lognormal_graph(10000,3)', 'facebook_graph', 'epinions_graph']

    for key in ["color televisions", "room air conditioners", "clothers dryers"]:
        for i, g_name in enumerate(g_name_cont):
            print(i, g_name)
            if g_name in ['exponential_graph(10000,3)', 'gaussian_graph(10000,3)', 'lognormal_graph(10000,3)']:
                graph = g_name[:5] + "_seq"
                g = eval(graph)
            elif g_name in ['facebook_graph', 'epinions_graph']:
                g = eval(g_name)
            else:
                g = eval("nx." + g_name)
            insert_data(g, g_name_cont[i], key, data_set)
