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
get_ipython().system('mongorestore -d abmDiffusion --dir "./abmDiffusion"')

#%%
get_ipython().system('mongodump -d abmDiffusion -o "./"')
