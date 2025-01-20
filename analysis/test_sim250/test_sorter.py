import numpy as np
from spikeinterface.comparison import GroundTruthStudy, compare_sorter_to_ground_truth
import spikeinterface.full as si
import os
import pandas as pd

def main():
    intp_way = "whl" # pattern: prt (Partial) / whl (Whole)
    factor_list = [2]
    # method_list = ['intp','krig','remove']
    method_list = ['restormer']
    sorter_list = ['kilosort4','herdingspikes']
    # sorter_list = ['kilosort4']

    for factor in factor_list:
        study_folder = '{}_sorter_factor{}_1'.format(intp_way,factor)
        datasets,cases = {},{}
        for method in method_list:
            print(method)
            if intp_way == "prt": # partial 部分插值，保留没有损坏的时刻
                file = "/root/autodl-tmp/{}_resfile.nwb".format(method)
                rec, sorting = si.read_nwb(file, load_recording=True, load_sorting=True)
            else: # whole 全部插值，不保留没有损坏的时刻
                file = "/root/autodl-tmp/sub-MEAREC-250neuron-Neuropixels_ecephys.nwb" # 原始信号
                rec, sorting = si.read_nwb(file, load_recording=True, load_sorting=True)
                channel_ids = rec.channel_ids
                bad_chans = channel_ids[channel_ids % factor == 1]
                good_chans = channel_ids[~np.isin(channel_ids, bad_chans)]
                if method == 'krig': # 克里金插值，此前不可以删除通道
                    rec = si.interpolate_bad_channels(rec, bad_chans)
                elif method == 'remove': # 将损坏通道插值为0
                    rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
                    rec = si.interpolate_bad_channels(rec, bad_chans, weights=rmv_weights)
                else:
                    file = "/root/autodl-tmp/{}_{}.nwb".format(method,factor)
                    rec, sorting = si.read_nwb(file, load_recording=True, load_sorting=True)
                print(file)
                
            data = {
                method:(rec,sorting)
            }
            datasets.update(data)

        for method in datasets:
            for sorter_name in sorter_list:
                case_i = {
                    (sorter_name, method): {
                        "label": "{} on {}".format(sorter_name,method),
                        "dataset": method,
                        "run_sorter_params": {
                            "sorter_name": sorter_name,
                        },
                    }
                }
                cases.update(case_i)

        if os.path.exists(study_folder):
            study = GroundTruthStudy(study_folder)
        else:
            study = GroundTruthStudy.create(study_folder, datasets=datasets, cases=cases,
                                            levels=["sorter_name", "dataset"])
            study.run_sorters()
            # Remove units in Kilosort4 that violate the refractory period
            for key in cases:
                if key[0] == 'kilosort4':
                    sorter_folder = study.folder / "sorters" / study.key_to_str(key)
                    cluster_group = pd.read_csv(os.path.join(sorter_folder, 'sorter_output/cluster_group.tsv'), sep='\t')
                    bad_units_ids = cluster_group.loc[cluster_group['KSLabel'] != 'good', 'cluster_id'].tolist()
                    # kilosort
                    sorter = study.sortings[key]
                    rmv_sorter = sorter.remove_units(remove_unit_ids=bad_units_ids)
                    study.sortings[key] = rmv_sorter  # 重新修改
                    print('remove units:{}'.format(rmv_sorter.count_total_num_spikes()))

            study.run_comparisons(exhaustive_gt=True) # run all comparisons and loop over the results

        # # this is a dataframe
        # perfs = study.get_performance_by_unit()
        # print(perfs)

        # this is a dataframe
        unit_counts = study.get_count_units()
        unit_counts.to_csv(os.path.join(study_folder, 'unit_counts.csv'))
        print(unit_counts)

        # we can also access run times
        run_times = study.get_run_times()
        print(run_times)

        print(study.cases.keys())

if __name__ == '__main__':
    main()
