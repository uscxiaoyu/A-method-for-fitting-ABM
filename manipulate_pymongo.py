# coding=utf-8
#%%
get_ipython().run_line_magic('pylab', 'inline')
from pymongo import MongoClient

#%%
client = MongoClient('localhost', 27017)

#%%
client.list_database_names()  # 列出所有的数据库

#%%
for i, d in enumerate(client.list_databases()):
    print(i, f"{d['name']:<20}\t{d['sizeOnDisk']/(1024**2):.2f}M")

#%%
db = client.abmDiffusion
db.list_collection_names()

#%%
# 删除数据库
client.drop_database("abmDiffusion")

#%%
# 导入数据库
get_ipython().system('mongorestore -d abmDiffusion --dir "./abmDiffusion"')

#%%
# 备份数据库
get_ipython().system('mongodump -d abmDiffusion -o "./"')
