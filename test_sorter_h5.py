import os
from spikeinterface import get_channel_distances
from spikeinterface.sortingcomponents.peak_localization import localize_peaks, LocalizeCenterOfMass
import numpy as np
import spikeinterface.full as si
import MEArec as mr
import pandas as pd
from spikeinterface.comparison import GroundTruthStudy

def run_sim_sorter(origin_file=None, opt_path = None, sorter_list = None, method_list = None, study_folder = 'test_sorter', bad_chans = None, factor=16):
    datasets,cases = {},{}
    # rec,sorting = si.read_mearec(opt_path)

    for method in method_list:
        
        rec, sorting = si.read_mearec(origin_file)
        channel_ids = rec.channel_ids.astype(int)
        if bad_chans is None:
            bad_chans = rec.channel_ids[channel_ids % factor == 1]
        good_chans = rec.channel_ids[~np.isin(rec.channel_ids, bad_chans)]
        if method == 'krig': 
            rec = si.interpolate_bad_channels(rec, bad_chans)
        elif method == 'remove': 
            rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
            rec = si.interpolate_bad_channels(rec, bad_chans, weights=rmv_weights)
        else:
            rec,sorting = si.read_mearec(opt_path)
            
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
                study.sortings[key] = rmv_sorter 
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

     # recordings_60cells_Neuropixels-128_30.0_10.0uV.h5 / recordings_50cells_SqMEA-10-15_30.0_10.0uV.h5
    origin_file = '/root/autodl-tmp/recordings_60cells_Neuropixels-128_30.0_10.0uV.h5'
    opt_path = '../np128_res.h5'  # sqmea_16.h5/ np128_16.h5
    # opt_path = '../res_rec/mix/sqmea_16.h5'

    bad_chans = np.load('../generate_traces_for_train/np/generate_krig_traces_factor_2/bad_chls.npy')

    study_folder = 'test_sorter_h5'
    method_list = ['krig', 'remove', 'edsr']
    sorter_list = ['mountainsort5','herdingspikes']
    run_sim_sorter(origin_file, opt_path, sorter_list, method_list, study_folder, bad_chans, factor=2)

    print('test..')