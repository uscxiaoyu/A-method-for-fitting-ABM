#coding=utf-8
from pymongo import MongoClient
import numpy as np
import pylab as pl
import pandas as pd


class abmDBfit:
    
    def __init__(self, db):
        self.curves = db["diffuse_curves"]
        if db["_id"] == 'facebook_graph':
            self.num_nodes = 4039
        elif db["_id"] == 'epinions_graph':
            self.num_nodes = 75879
        else:
            self.num_nodes = 10000
    
    def squr_r(self, s, f_act):
        f_act = np.array(f_act)
        tse = np.sum(np.square(s - f_act))
        mean_y = np.mean(s)
        ssl = np.sum(np.square(s - mean_y))
        return (ssl - tse)/ssl

    def get_M(self, s, d):  # 获取对应扩散率曲线的最优潜在市场容量
        '''
        s: 待拟合数据
        d: 数据库中的曲线
        '''
        s, d = np.array(s), np.array(d)
        s_len = len(s)
        p, q = d[:2]
        if s_len > len(d[2:]):
            raise Exception("待拟合数据超过长度!")
        elif sum(d[2:]) == 0:
            raise Exception("拟合数据异常！")
        else:
            x = d[2 : 2+s_len]

        a = np.sum(np.square(x)) / np.sum(s)  # 为减少a的大小, 除以np.sum(self.s)
        b = -2*np.sum(x*s) / np.sum(s)
        c = np.sum(np.square(s)) / np.sum(s)
        mse = np.sqrt(sum(s) * (4*a*c - b**2) / (4*a*s_len))
        sigma = -b/(2*a)
        m = sigma*self.num_nodes
        return [mse, p, q, m], list(d[2:]*sigma)

    def fit(self, s=None):
        res_list = []
        for key in self.curves:
            try:
                d = self.curves[key]
                res = self.get_M(s, d)
                res_list.append(res)
            except Exception:
                pass
        
        res_list.sort(key=lambda x:x[0][0])
        return res_list[0]

    def predict(self, s, b_idx=6):
        one_step = []
        for i in range(b_idx, len(s)):
            x = s[:i]
            res = self.fit(x)
            delta = abs(res[1][i] - s[i])
            one_step.append([delta/s[i], delta, delta**2])  # map, mad, mse
        return one_step


if __name__ == "__main__":
    s_dict = {'room air conditioners':(np.arange(1949, 1962), 
                [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
        'color televisions':(np.arange(1963,1971),
                [747,1480,2646,5118,5777,5982,5962,4631]),
        'clothers dryers':(np.arange(1949,1962),
                [106,319,492,635,737,890,1397,1523,1294,1240,1425,1260,1236])}
    client = MongoClient()
    database = client.abmDiffusion
    prj = database.abmDatabase
    
    s = s_dict["color televisions"][1]
    db = prj.find_one({"_id": 'barabasi_albert_graph(10000,3)'})
    abm = abmDBfit(db)
    res = abm.fit(s)
    _, p, q, m = res[0]
    s_fit = res[1][:len(s)]
    r2 = abm.squr_r(s, s_fit)
    pred = abm.predict(s)
    print(f"p:{p}, q:{q}, m:{m:.0f}, r2:{r2:.4f}")
    print(np.mean(pred, axis=0))

    '''
    id_cont = []
    for x in prj.find({}, projection={"_id":1}):
        id_cont.append(x["_id"])

    for i, _id in enumerate(id_cont):
        db = prj.find_one({"_id": _id})
        abm = abmDBfit(db)
        res = abm.fit(s)
        _, p, q, m = res[0]
        s_fit = res[1][:len(s)]
        r2 = abm.squr_r(s, s_fit)
        print(f"{i, _id}\n\t p:{p}, q:{q}, m:{m:.0f}, r2:{r2:.4f}")
    '''
    
