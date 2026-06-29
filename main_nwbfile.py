from res_utils import Generate


def main(opt_path=None):

   generate = Generate(opt_path)

   # 1. generate training data
<<<<<<< HEAD
   # generate.generate_train_list()

   # 2. train model
   generate.generate_train_weights()
=======
   generate.generate_train_list()

   # 2. train model
   # generate.generate_train_weights()
>>>>>>> master

   # 3. restored recordings (and calculate MSE/NRMSE)
   # restored_file = '../{}_{}.nwb'.format(generate.method,generate.factor)
   # weights_path = 'weights/sim250/net_g_factor{}.pth'.format(generate.factor)
   # generate.generate_recon_nwbfile(restored_file,weights_path)

   # 4.evaluate: spike sorter / hit rate

   print('test..')

if __name__ == '__main__':
   opt_path = 'options_linux/sim250/Reconstrcution_Restormer.yml'
<<<<<<< HEAD
=======
   # opt_path = 'options_linux/test_EDSR.yml'
>>>>>>> master
   main(opt_path=opt_path)