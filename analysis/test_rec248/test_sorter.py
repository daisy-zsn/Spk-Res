import pickle
import numpy as np
import spikeinterface.full as si
import os
import pandas as pd
import spikeextractors as se
from spikeinterface.sorters import read_sorter_folder
import spikecomparison as sc

intp_list = ['origin','whl'] # prt / whl / origin
sorter_list = ['kilosort4', 'herdingspikes']

folder = 'D:\\Work\\dataset\\origin_dataset\\spikeInterface-sub-mouse412804'
file = os.path.join(folder,'sub-mouse412804_ecephys.nwb')
rec = si.read_nwb(file,load_recording=True)

fileC1 = os.path.join(folder,'sub-mouse412804_ses-20200824T155542.nwb')
fileC2 = os.path.join(folder,'sub-mouse412804_ses-20200824T155543.nwb')
sampling_frequency = rec.sampling_frequency
curated1 = se.NwbSortingExtractor(fileC1, sampling_frequency=sampling_frequency)  # 人工sorter1
curated2 = se.NwbSortingExtractor(fileC2, sampling_frequency=sampling_frequency)  # 人工sorter2

for intp_way in intp_list:
    if intp_way=='origin':
        factor_list = [0]
        method_list = ['origin']
    else:
        factor_list = [2,4,8,16]
        method_list = ['intp','krig','remove']

    for factor in factor_list:
        # 初始化dataframe
        df = pd.DataFrame(columns=['{0:03b}'.format(v) for v in range(1, 2 ** 3)])
        for method in method_list:
            if intp_way == "prt": # partial 部分插值，保留没有损坏的时刻
                file = "/root/autodl-tmp/{}_resfile.nwb".format(method)
                rec = si.read_nwb(file, load_recording=True, load_sorting=False)
            elif intp_way == 'whl': # whole 全部插值，不保留没有损坏的时刻
                channel_ids = rec.channel_ids
                bad_chans = channel_ids[channel_ids % factor == 1]
                good_chans = channel_ids[~np.isin(channel_ids, bad_chans)]
                if method == 'intp':
                    file = "/root/autodl-tmp/{}_resfile.nwb".format(method)
                elif method == 'krig': # 克里金插值，此前不可以删除通道
                    rec = si.interpolate_bad_channels(rec, bad_chans)
                elif method == 'remove': # 将损坏通道插值为0
                    rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
                    rec = si.interpolate_bad_channels(rec, bad_chans, weights=rmv_weights)

            # 运行代码
            for sorter_name in sorter_list:
                output_folder = "{}/real_sorter_factor{}/{}/{}_output".format(intp_way,factor,method,sorter_name)
                if not os.path.exists(output_folder):
                    if sorter_name == 'kilosort4':
                        sorter = si.run_sorter(sorter_name, recording=rec, output_folder=output_folder,
                                               remove_existing_folder=False, do_correction=True)
                    else:
                        sorter = si.run_sorter(sorter_name, recording=rec, output_folder=output_folder,
                                               remove_existing_folder=False)
                else:
                    sorter = read_sorter_folder(output_folder=output_folder)

                ######################### 1. 与人工sorter对比 #########################
                if not os.path.exists(os.path.join(output_folder,'cmp_data.pickle')):
                    if sorter_name == 'kilosort4':
                        cluster_group = pd.read_csv(os.path.join(output_folder, 'sorter_output\\cluster_group.tsv'), sep='\t')
                        bad_units_ids = cluster_group.loc[cluster_group['KSLabel'] != 'good', 'cluster_id'].tolist()
                        sorter = sorter.remove_units(remove_unit_ids=bad_units_ids)
                        print('remove units:{}'.format(sorter.count_total_num_spikes()))

                    sorter_name = ['curated1', 'curated2', sorter_name]
                    sorters = [curated1, curated2, sorter]

                    comparison_curated = sc.compare_multiple_sorters(sorting_list=sorters, name_list=sorter_name)
                    i = {}
                    for k in ['{0:03b}'.format(v) for v in range(1, 2 ** 3)]:
                        i[k] = 0
                    i['111'] = len(comparison_curated.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids())  # 3个sorter
                    s = comparison_curated.get_agreement_sorting(minimum_agreement_count=2,minimum_agreement_count_only=True)
                    units = [s.get_unit_property(u, 'sorter_unit_ids').keys() for u in s.get_unit_ids()]
                    for u in units:
                        if sorter_name[0] in u and sorter_name[1] in u:  # c1 and c2
                            i['110'] += 1
                        if sorter_name[0] in u and sorter_name[2] in u:  # c1 and ks4
                            i['101'] += 1
                        if sorter_name[1] in u and sorter_name[2] in u:  # c2 and ks4
                            i['011'] += 1
                    s = comparison_curated.get_agreement_sorting(minimum_agreement_count=1,minimum_agreement_count_only=True)
                    units = [s.get_unit_property(u, 'sorter_unit_ids').keys() for u in s.get_unit_ids()]
                    for u in units:
                        if sorter_name[0] in u:  # c1
                            i['100'] += 1
                        if sorter_name[1] in u:  # c2
                            i['010'] += 1
                        if sorter_name[2] in u:  # ks4
                            i['001'] += 1

                    # 存储dict类型的数据
                    with open(os.path.join(output_folder,'cmp_data.pickle'), 'wb') as f:
                        pickle.dump(i, f)
                else:
                    with open(os.path.join(output_folder,'cmp_data.pickle'), 'rb') as f:
                        i = pickle.load(f)

                index_name = '{}_{}'.format(method,sorter_name)
                df.loc[index_name] = i

        # 存储df
        df.to_csv('{}/real_sorter_factor{}/cmp_results.csv'.format(intp_way,factor))
        print(df)


print("recon..")