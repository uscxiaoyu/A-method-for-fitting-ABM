#%%
f = open("mongoatlas.txt", 'r')
connect_string = f.read()
print(connect_string)

#%%
from pymongo import MongoClient
client = MongoClient(
    "mongodb+srv://yxmongo:xiaoyu%401986@cluster0.mongodb.net/admin")
#%%
db = client.abmDiffusion

#%%
prj = db.test
prj.insert_one({"x":1, "y":2})

