import os.path
import threading
import time
import numpy as np
import spikeinterface.full as si
from ... import res_utils


def main():
    intp_way = "whl" # prt / whl
    factor_list = [2]
    method_list = ["remove","krig","intp"]

    base_path = "{}/results_hitrate".format(intp_way) # 存储路径
    origin_file = "sub-mouse412804_ecephys.nwb"
    origin_rec = si.read_nwb(origin_file, load_recording=True, load_sorting=False)
    channel_ids = origin_rec.channel_ids

    # calc spike_train
    spike_train_list = res_utils.metrics_utils.calc_spike_train(base_path,origin_rec)

    threads = []
    t1 = time.time()
    # 创建线程
    for factor in factor_list:
        bad_chans = channel_ids[channel_ids % factor == 1] # 坏通道
        good_chans = channel_ids[~np.isin(channel_ids, bad_chans)] # 好通道
        save_path = os.path.join(base_path, "factor_{}".format(factor)) # 结果存储路径
        for method in method_list:
            rec = None
            if intp_way == "prt":
                file = "/root/autodl-tmp/factor{}/{}_resfile.nwb".format(factor,method)
                rec = si.read_nwb(file, load_recording=True, load_sorting=False)
            else:
                if method == 'krig':
                    rec = si.interpolate_bad_channels(origin_rec, bad_chans)
                elif method == 'remove':
                    rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
                    rec = si.interpolate_bad_channels(origin_rec, bad_chans, weights=rmv_weights)
                else:
                    file = "/root/autodl-tmp/factor{}_{}_resfile.nwb".format(factor, method)
                    rec = si.read_nwb(file, load_recording=True, load_sorting=False)

            t = threading.Thread(target=res_utils.metrics_utils.calc_hitrate, name='factor_{},method_{}'.format(factor,method), args=(save_path,spike_train_list,rec,method))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    print("All threads have finished execution.")
    print("total time %ds" % (time.time() - t1))


if __name__ == '__main__':
    main()


