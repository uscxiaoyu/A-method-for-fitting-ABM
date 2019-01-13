import os
import time
raw_dated = os.popen("sudo pip3.6 list --outdated").read()
outdated = [x.split(' ')[0] for x in raw_dated.split('\n')[2:]]
for dist in outdated:
    t1 = time.process_time()
    os.system("sudo pip3.6 install --upgrade " + dist)
    print(f'{dist} 耗时{time.process_time()-t1 : .2f}秒')
