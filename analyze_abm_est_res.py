# coding=utf-8
#%%
get_ipython().run_line_magic('pylab', 'inline')
from pymongo import MongoClient
import pandas as pd

#%%
client = MongoClient('localhost', 27017)
db = client.abmDiffusion
db.list_collection_names()
#%%
prj = db.abmEstimate
item = prj.find_one({})
item.keys()
#%%
x = eval(item["color televisions"][0])
x["params"], x["fitness"], x["num_nodes"]
#%%
# 将各数据集的10次估计转换出来
def single_item(x_list):
    x_dict = {"p":[], "q":[], "m":[], "r2":[], "num_nodes":[]}
    for x in x_list:
        a = eval(x)
        x_dict["p"].append(a["params"][0])
        x_dict["q"].append(a["params"][1])
        x_dict["m"].append(a["params"][2])
        x_dict["r2"].append(a["fitness"])
        x_dict["num_nodes"].append(a["num_nodes"])
    res = pd.DataFrame(x_dict)
    mean_res = res.mean().values
    std_res = res.std().values
    f_res = []  # 将p, q, m, r2, num_nodes均值和方差依次展开
    for d in zip(mean_res, std_res):
        f_res.extend(d)

    return f_res # [mean_p, std_p, mean_q, ...., mean_num, std_num]
#%%
# 将各ABM的数据整理出来
def trans_data(item_list, cate):
    columns = ["mean_p", "std_p", "mean_q", "std_q", "mean_m", "std_m",
               "mean_r2", "std_r2", "mean_num", "std_num"]  # 列名
    key_list = []  # 索引
    d_list = []  # 数据
    for item in item_list:
        new_key = item["_id"] + "$" + cate
        key_list.append(new_key)

        f_res = single_item(item[cate])
        d_list.append(f_res)

    return pd.DataFrame(d_list, index=key_list, columns=columns)
#%%
# color televisions
cate = "color televisions"
items = prj.find({}, projection={cate: 1})
res = trans_data(items, cate)
res = res.sort_index()
res.head()
#%%
res.to_excel(excel_writer = cate + '.xls')


