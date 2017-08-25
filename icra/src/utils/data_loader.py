import scipy.io as sio
import time


sphero_data = sio.loadmat("../data/sphero.mat")

for k, v in sphero_data.items():
    for k in sphero_data['exp']:
        print(k, v)
        time.sleep(2)
# expt_I = sphero_data['expt']
# print(expt_I)
