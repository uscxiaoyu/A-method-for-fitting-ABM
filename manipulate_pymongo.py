# coding=utf-8
#%%
get_ipython().run_line_magic('pylab', 'inline')
from pymongo import MongoClient
from bassestimate import BassEstimate
import numpy as np
import datetime
import time

#%%
client = MongoClient('localhost', 27017)
db = client.abmDiffusion
db.list_collection_names()

#%%

#%%
prj = db.indivHeter
prj.find_one({}, projection={"_id":1, "forecasts":1})

#%%
# 重新加载数据库
get_ipython().system('mongorestore -d abmDiffusion --dir "./abmDiffusion"')

#%%
# 删除数据库
get_ipython().system('mongodump -d abmDiffusion -o "./"')