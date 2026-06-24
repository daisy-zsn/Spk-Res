import os.path
import threading
import time
import numpy as np
import spikeinterface.full as si
# from .. import res_utils
from metrics_utils import calc_hitrate

def main():
    intp_way = "whl" # prt / whl
    # factor_list = [2,4,8,16]
    factor_list = [2]
    # method_list = ["remove","krig","intp"]
    method_list = ["restormer"]

    base_path = "{}/results_hitrate_1".format(intp_way) # 存储路径
    origin_file = "/root/autodl-tmp/sub-MEAREC-250neuron-Neuropixels_ecephys.nwb"
    origin_rec, sorting = si.read_nwb(origin_file, load_recording=True, load_sorting=True)
    channel_ids = origin_rec.channel_ids

    # ground truth spike_train
    spike_train_list = []
    unit_ids = sorting.get_unit_ids()
    for i in np.arange(len(unit_ids)):
        unit = unit_ids[i]
        spikes_train = sorting.get_unit_spike_train(unit)
        spike_train_list.extend(spikes_train)
    spike_train_list = np.sort(np.asarray(spike_train_list)).astype(float)

    threads = []
    t1 = time.time()
    for factor in factor_list:
        bad_chans = channel_ids[channel_ids % factor == 1] # 坏通道
        good_chans = channel_ids[~np.isin(channel_ids, bad_chans)] # 好通道
        save_path = os.path.join(base_path, "factor_{}".format(factor)) # 结果存储路径
        for method in method_list:
            if intp_way == "prt": # 部分植入
                file = "/root/autodl-tmp/{}_resfile.nwb".format(method)
                rec = si.read_nwb(file, load_recording=True, load_sorting=False)
            else:
                if method=='krig': # 克里金插值，此前不可以删除通道
                    rec = si.interpolate_bad_channels(origin_rec, bad_chans)
                elif method=='remove': # 将损坏通道插值为0
                    rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
                    rec = si.interpolate_bad_channels(origin_rec, bad_chans, weights=rmv_weights)
                else:
                    file = "/root/autodl-tmp/{}_{}.nwb".format(method,factor)
                    rec = si.read_nwb(file, load_recording=True, load_sorting=False)

            t = threading.Thread(target=calc_hitrate, name='factor_{},method_{}'.format(factor,method), args=(save_path,spike_train_list,rec,method))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    print("一共耗时%ds" % (time.time() - t1))
    print("All threads have finished execution.")


if __name__ == '__main__':
    main()

