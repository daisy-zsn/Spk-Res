"""NWB 文件流程（重构版）。

基于 ``main_nwbfile.py``，使用重构后的 ``generate_utils_v2.Generate``。
恢复了被注释的 nwb 重建调用，接入统一的 ``generate_recon_nwbfile``。
"""
from res_utils.generate_utils_v2 import Generate


def main(opt_path=None):
    generate = Generate(opt_path)

    # generate.generate_train_list()

    # 2. train model
    generate.generate_train_weights()

    # 3. restored recordings (and calculate MSE/NRMSE)
    restored_file = '../{}_{}.nwb'.format(generate.method, generate.factor)
    generate.generate_recon_nwbfile(restored_file)

    # 4. analysis dict: spike sorter / hit rate

    print('test..')


if __name__ == '__main__':
    # opt_path = 'options_linux/sim250/Reconstrcution_EDSR.yml'
    opt_path = 'options_linux/test_nwb_demo.yml'
    main(opt_path=opt_path)