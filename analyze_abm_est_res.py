# coding=utf-8
#%%
get_ipython().run_line_magic('pylab', 'inline')
from pymongo import MongoClient
from bassestimate import BassEstimate
import numpy as np
import pandas as pd
import datetime
import time

#%%
client = MongoClient('localhost', 27017)

#%%
client.list_database_names()

#%%
db = client.abmDiffusion
db.list_collection_names()

#%%
db.networks.find_one().keys()

#%%
prj = db.abmEstimate
prj.find_one({}, projection={"_id": 1, "forecasts": 1})
