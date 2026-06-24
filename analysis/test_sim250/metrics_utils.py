import os
import numpy as np
from tqdm import tqdm
import spikeinterface.full as si
import time
import threading
from spikeinterface import get_noise_levels
from spikeinterface.sortingcomponents.peak_detection import detect_peaks


# 计算NRMSE（归一化均方根误差）
def calc_NRMSE(y_gt, y_pred):  # 数据格式：spike个数×波形采样点数量，需要将阵列数据展开成一维的。
    mse = np.mean((y_gt - y_pred) ** 2, axis=1)
    rmse = np.sqrt(mse)  # 开方，并求均值
    diff = np.abs(np.max(y_gt, axis=1) - np.min(y_gt, axis=1))
    nrmse = np.mean(rmse / diff)
    return nrmse


# precision / hit rate: TP/(TP+FN)
# TP: 若预测时间与ground truth时间差值 < 0.5ms，视作正确的spikes信号，则TP+1
# FN: 若预测时间与ground truth时间差值 > 0.5ms，视作错误的spikes信号，则FN+1
# 插值前后的spike sorting的结果对比
# （2）直接根据sorter检测出的spikes，放入模型中，进行后续处理。详见test_model_sorter.py
# 插值前后的spike localization的结果对比

def calc_hitrate(save_path, spike_train_list, rec, method):
    start = time.time()
    my_thread_name = threading.current_thread().name  # 获取当前线程名称
    print('%s开始运行...' % my_thread_name)
    my_thread_id = threading.current_thread().ident  # 获取当前线程id
    print('当前线程为：{}，线程id为：{}，所在进程为：{}'.format(my_thread_name, my_thread_id, os.getpid()))

    method_path = os.path.join(save_path, method)
    if not os.path.exists(method_path):
        os.makedirs(method_path)

    sample_index_path = os.path.join(method_path, "sample_index.npy")
    # 阈值法检测spikes
    if not os.path.exists(sample_index_path):
        peaks = detect_peaks(rec, method="locally_exclusive",
                             noise_levels=get_noise_levels(rec, return_scaled=False),
                             pipeline_nodes=None, chunk_duration='2s', n_jobs=8)
        sample_index = peaks["sample_index"]
        np.save(sample_index_path, sample_index)
    else:
        sample_index = np.load(sample_index_path)

    hit_rate_path = os.path.join(method_path, "hit_rate.npy")
    if not os.path.exists(hit_rate_path):
        # 计算hit rate
        res_num = 0
        interval = 9
        spikes_list = np.copy(spike_train_list)
        for i in tqdm(sample_index):
            slist = np.abs(spikes_list - i)
            si = np.argmin(slist)
            if slist[si] <= interval:
                spikes_list[si] = np.inf
                res_num += 1

        hit_rate = res_num / len(spikes_list)
        print("hit_rate:" + str(hit_rate))
        np.save(hit_rate_path, hit_rate)
    else:
        hit_rate = np.load(hit_rate_path)
        print("hit_rate:" + str(hit_rate))
    print('%s线程运行结束，耗时%ds...' % (my_thread_name, time.time() - start))


def calc_spike_train(save_path, rec):
    if not os.path.exists(os.path.join(save_path, 'spike_train_list.npy')):
        peaks = detect_peaks(rec, method="locally_exclusive",
                             noise_levels=get_noise_levels(rec, return_scaled=False),
                             pipeline_nodes=None, chunk_duration='2s', n_jobs=20)
        sample_index = peaks["sample_index"]
        np.save(save_path, sample_index)
    else:
        sample_index = np.load(os.path.join(save_path, 'spike_train_list.npy'))
    return sample_index