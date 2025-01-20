from utils.generate_utils import Generate


def main(opt_path=None):

   generate = Generate(opt_path)

   # 1. generate training data
   generate.generate_train_list(is_save_path=False)

   # 2. train model
   generate.generate_train_weights()

   # 3. restored recordings (and calculate MSE/NRMSE)
   restored_file = 'p2_intp'
   weights_path = 'weights_random'
   generate.generate_recon_binfile(restored_file,weights_path)

   # 4.evaluate: spike sorter / hit rate


   print('test..')

if __name__ == '__main__':
   opt_path = 'D:/Work/code/test_SpkRes/test_drifting/Reconstrcution_SpkRes.yml'
   main(opt_path=opt_path)