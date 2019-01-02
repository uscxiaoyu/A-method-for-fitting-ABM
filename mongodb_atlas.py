#%%
f = open("mongoatlas.txt", 'r')
connect_string = f.read()
print(connect_string)

#%%
from pymongo import MongoClient
get_ipython().run_line_magic('pylab', 'inline')
client = MongoClient("mongodb+srv: // abmdiffusion-kk6qc.azure.mongodb.net/test --username yxmongo")

#%%
db = client.abmDiffusion

#%%
prj = db.test
prj.insert_one({"x":1, "y":2})

