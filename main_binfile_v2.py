from res_utils.generate_utils_v2 import Generate


def main(opt_path=None):
    generate = Generate(opt_path)

    # 1. generate training data
    generate.generate_train_list()

    # 2. train model
    generate.generate_train_weights()

    # 3. restored recordings (and calculate MSE/NRMSE)
    restored_file = '../p2_intp'  # bin 文件前缀（scan_files 会找到对应的 .bin/.meta）
    generate.generate_recon_binfile(restored_file)

    # 4. analysis: spike sorter / hit rate / test_drifting

    print('test..')


if __name__ == '__main__':
    opt_path = 'options_linux/test_bin_demo.yml'
    main(opt_path=opt_path)