import time
from collections import OrderedDict
import gtimer as gt
from tqdm import trange


for e in gt.timed_for(trange(1,100),save_itrs=True):
    time.sleep(1)
    gt.stamp('aa')
    time.sleep(2)
    gt.stamp('bb')

    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        tt = times_itrs[key][-1]
        epoch_time += tt
        times['time/{} (s)'.format(key)] = tt
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total

    print(times)