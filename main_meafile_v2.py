"""MEArec h5 文件完整流程（重构版）。

基于 ``main_meafile.py`` 的最新修改，使用重构后的 ``generate_utils_v2.Generate``。

流程：
1. 生成训练数据（单文件 / mixture 可选）
2. 训练模型
3. 重建 h5 录音（含 z-score 归一化 / 反归一化）
4. spike sorter 评估（krig / remove / spkres × mountainsort5 / herdingspikes）
"""
from res_utils.generate_utils_v2 import Generate
from test_sorter_h5 import run_sim_sorter


def main(opt_path=None, electrode=None, sorter_fold=None, mode='few_shot', model=None):
    generate = Generate(opt_path)

    # 1. generate training data
    generate.generate_train_list()  # single file
    # generate.generate_train_list_mix()  # mixture file

    # 2. train model
    generate.generate_train_weights()

    # 3. restored recordings (and calculate MSE/NRMSE)
    restored_file = '../{}_res.h5'.format(electrode)
    generate.generate_recon_h5file(restored_file, mode=mode)

    # 4. evaluate: spike sorter / hit rate
    method_list = ['krig', 'remove', 'spkres']
    sorter_list = ['mountainsort5', 'herdingspikes']
    study_folder = '../sorter/{}/factor_{}/{}_{}'.format(model, generate.factor, sorter_fold, mode)
    run_sim_sorter(generate.rec_path, restored_file, sorter_list, method_list,
                   study_folder, generate.bad_chans, factor=generate.factor)

    print('test..')


if __name__ == '__main__':

    # opt_path = 'options_linux/sim250/Reconstrcution_Restormer.yml'

    model = 'spkres'
    mode = 'few_shot'  # few_shot / zero_shot / zero_shot_gcl
    factor_list = [16, 8, 4, 2]

    # sorter_fold_list = ['fs_sq_rd_nm_p1_5_kd', 'fs_sq_rd_nm_p1_5_ewc', 'fs_sq_rd_nm_p1_5_llr']
    # sorter_fold_list = ['rec_np_rd_nm_p1_5_kd', 'rec_np_rd_nm_p1_5_ewc', 'rec_np_rd_nm_p1_5_llr']

    # sorter_fold_list = ['single_np_rd_nm_p1_5_vae']
    # sorter_fold_list = ['single_sq_rd_nm_p1_5_vae']

    # single_np_rd / single_np_rd_nm / single_sq_rd / single_sq_rd_nm
    # zs_sq_rd / zs_sq_rd_nm
    # fs_sq_rd_nm_p1_5 / fs_sq_rd_nm_p1_5_ewc / fs_sq_rd_nm_p1_5_kd / fs_sq_rd_nm_p1_5_l2 / fs_sq_rd_nm_p1_5_llr
    # rec_np_rd_nm_p1_5 / rec_np_rd_nm_p1_5_ewc / rec_np_rd_nm_p1_5_kd / rec_np_rd_nm_p1_5_l2 / rec_np_rd_nm_p1_5_llr
    # sorter_fold_list = ['fs_sq_rd_nm_p1_5', 'fs_sq_rd_nm_p1_5_ewc', 'fs_sq_rd_nm_p1_5_kd', 'fs_sq_rd_nm_p1_5_l2', 'fs_sq_rd_nm_p1_5_llr']
    # sorter_fold_list = ['rec_np_rd_nm_p1_5','rec_np_rd_nm_p1_5_ewc', 'rec_np_rd_nm_p1_5_l2', 'rec_np_rd_nm_p1_5_llr']

    sorter_fold_list = ['fs_sq_rd_nm_p1_5_kd']
    electrode = 'sqmea'  # sqmea: 100 / np128: 128

    for factor in factor_list:
        for sorter_fold in sorter_fold_list:
            opt_path = 'options_linux_list/{}/{}/factor{}.yml'.format(model, sorter_fold, factor)
            main(opt_path=opt_path, electrode=electrode, sorter_fold=sorter_fold,
                 mode=mode, model=model)

    print('end..')