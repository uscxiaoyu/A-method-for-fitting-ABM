# coding=utf-8
#%%
get_ipython().run_line_magic('pylab', 'inline')
from pymongo import MongoClient
from abmestimate import estimateABM
import networkx as nx
import numpy as np
import time
#%%
def insert_data(g, g_name, key, data_set):
    s = data_set[key][1]
    res = prj.find_one({"_id": g_name}, projection={key: 1})
    len_key = len(res.get(key, [])) if res else 0  # 第一次查找key，res为None
    while len_key < 10:
        print(f"第{len_key + 1}轮:")
        t1 = time.perf_counter()
        est_abm = estimateABM(s, G=g, m_p=True)
        p0, q0 = est_abm.gener_init_pq()
        result = est_abm.solution_search(p0, q0)
        t3 = time.perf_counter()
        print(f' 用时: {t3 - t1:.2f}秒')
        print(f'    p0:{p0:.5f}, q0:{q0:.5f}')
        print(f" R2:{result['fitness']:.4f}    num_nodes:{result['num_nodes']}")

        if not res:
            prj.insert_one({"_id": g_name, key: [str(result)]})  # result的字符串形式，为了和MongoDB兼容
        elif len_key == 0:
            prj.update_one({"_id": g_name}, {"$set": {key: [str(result)]}})
        else:
            prj.update_one({"_id": g_name}, {"$push": {key: str(result)}})

        res = prj.find_one({"_id": g_name}, projection={key: 1})
        len_key = len(res.get(key, []))
    else:
        print(f"  任务已完成！")
#%%
client = MongoClient('localhost', 27017)
for i, d in enumerate(client.list_databases()):
    print(i, f"{d['name']:<20}\t{d['sizeOnDisk']/(1024**2):.2f}M")
#%%
db = client.abmDiffusion
db.list_collection_names()
#%%
prj = db.abmEstimate
for x in prj.find({}, projection={"_id":1}):
    print(x["_id"])

#%%
ep = prj.find_one({"_id": "epinions_graph"})
#%%
g = nx.read_gpickle('dataSources/epinions.gpickle')
s = [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]
#%%
new_dict = {}
for key in ep.keys():
    valid_items = []
    if key != "_id":
        print(key)
        print("========================================")
        for i, str_item in enumerate(ep[key]):
            item = eval(str_item)
            if not np.isnan(item["fitness"]):
                valid_items.append(str_item)
            else:  # 如果无效，则重新实验
                print(key, i, item["params"])
                try:
                    t1 = time.perf_counter()
                    est_abm = estimateABM(s, G=g, m_p=True)
                    p0, q0 = est_abm.gener_init_pq()
                    result = est_abm.solution_search(p0, q0)
                    valid_items.append(str(result))
                    t2 = time.perf_counter()
                    print(f' 用时: {t2 - t1:.2f}秒')
                    print(f'    p0:{p0:.5f}, q0:{q0:.5f}')
                    print(f" R2:{result['fitness']:.4f}    num_nodes:{result['num_nodes']}")
                except Exception as e:
                    print(e)
        print("========================================")
        new_dict[key] = valid_items
#%%
for key in new_dict:
    print(len(new_dict[key]))
#%%
new_dict["clothers dryers"][0]
#%%
len(ep["clothers dryers"])
#%%
prj.update_one({"_id": "epinions_graph"}, {"$set": {"clothers dryers": new_dict["clothers dryers"]}})
#%%
try:
    t1 = time.perf_counter()
    est_abm = estimateABM(s, G=g, m_p=True)
    p0, q0 = est_abm.gener_init_pq()
    result = est_abm.solution_search(p0, q0)
    valid_items.append(str(result))
    t2 = time.perf_counter()
    print(f' 用时: {t2 - t1:.2f}秒')
    print(f'    p0:{p0:.5f}, q0:{q0:.5f}')
    print(f" R2:{result['fitness']:.4f}    num_nodes:{result['num_nodes']}")
except Exception as e:
    print(e)
#%%
prj.update_one({"_id": "epinions_graph"}, {"$push": {"clothers dryers": str(result)}})
