# for python3
import estimate_bass as eb
import numpy as np
import time
import os
import multiprocessing
import networkx as nx

file_list = []


def vst_dir(path, exclude='.pkl', include='.npy'):
    for x in os.listdir(path):
        sub_path = os.path.join(path, x)
        if os.path.isdir(sub_path):
            vst_dir(sub_path)
        else:
            if include in sub_path.lower() and exclude not in sub_path.lower():
                file_list.append(sub_path)


def func(x, para_range, n=1):
    p, q = x[:2]
    s_full = x[2:]
    max_ix = np.argmax(s_full)
    s = s_full[:max_ix + 1 + n]
    bassest = eb.Bass_Estimate(s, para_range)
    bassest.t_n = 1000
    params = bassest.optima_search(c_n=200, threshold=10e-8)
    return [p, q] + list(params)


if __name__ == '__main__':
    '''
    diff_data = np.load(path + '/complete_graph(10000).npy')
    pool = multiprocessing.Pool(processes=6)
    para_range = [[1e-6, 0.1], [1e-4, 1], [0, 50000]]
    result = []
    t1 = time.clock()
    for x in diff_data:
        result.append(pool.apply_async(func, (x, para_range)))

    pool.close()
    pool.join()
    to_save = []
    for res in result:
        to_save.append(res.get())

    print(': Time elapsed: %.2fs' % (time.clock() - t1))
    np.save(path + '/estimate complete_graph(10000)', to_save)

    path = 'auto_data/'
    vst_dir(path)
    file_list = ['auto_data/watts_strogatz_graph(10000,6,0).npy',
                 'auto_data/watts_strogatz_graph(10000,6,0.1).npy',
                 'auto_data/watts_strogatz_graph(10000,6,0.3).npy',
                 'auto_data/watts_strogatz_graph(10000,6,0.5).npy']


    file_list = ['auto_data/watts_strogatz_graph(10000,6,0).npy', 'auto_data/watts_strogatz_graph(10000,6,0.1).npy',
                 'auto_data/gnm_random_graph(10000,30000),0.5.npy', 'auto_data/gnm_random_graph(10000,30000),0.7.npy',
                 'auto_data/gnm_random_graph(10000,30000),0.9.npy', 'auto_data/gnm_random_graph(10000,30000),1.0.npy']


    diff_data = np.load('auto_data/gnm_random_graph(10000,30000).npy')
    for n in [-1, 0, 1, 2, 3, 4, 5]:
        pool = multiprocessing.Pool(processes=6)
        para_range = [[0.00002, 0.09], [0.005, 0.9], [0, 30000]]
        result = []
        t1 = time.clock()
        for x in diff_data:
            result.append(pool.apply_async(func, (x, n, para_range)))

        pool.close()
        pool.join()
        to_save = []
        for res in result:
            to_save.append(res.get())

        print n, ': Time elapsed: %.2fs' % (time.clock() - t1)
        np.save('auto_data/' + 'estimate_' + 'gnm_random_graph(10000,30000)_Peak%s' % n, to_save)


    text_list = ['gnm_random_graph(10000,40000)', 'gnm_random_graph(10000,50000)', 'gnm_random_graph(10000,60000)',
                 'gnm_random_graph(10000,70000)', 'gnm_random_graph(10000,80000)', 'gnm_random_graph(10000,90000)',
                 'gnm_random_graph(10000,100000)']
'''
    g_1 = nx.read_gpickle('/dataSources/facebook.gpickle')
    g_2 = nx.read_gpickle('/dataSources/epinions.gpickle')
    num_nodes = {"facebook_network": nx.number_of_nodes(g_1),
                 "epinions_network": nx.number_of_nodes(g_2)}

    text_list = ["facebook_network", "epinions_network"]
    for text in text_list:
        diff_data = np.load('auto_data/' + text + '.npy')
        pool = multiprocessing.Pool(processes=5)
        para_range = [[0.00002, 0.08], [0.005, 0.8], [0.2 * num_nodes[text], 2 * num_nodes[text]]]
        result = []
        t1 = time.process_time()
        for x in diff_data:
            result.append(pool.apply_async(func, (x, para_range)))

        pool.close()
        pool.join()
        to_save = []
        for res in result:
            to_save.append(res.get())

        print(f"{text}: Time elapsed: {(time.process_time() - t1):.2f}s")
        np.save(f"generatedData/estimate_{text}", to_save)

