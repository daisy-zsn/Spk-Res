import os
import warnings
import h5py
import numpy as np
import spikeinterface.full as si
import spikeinterface.extractors as se
from runpy import run_path
import yaml
from pynwb import NWBHDF5IO
from tqdm import tqdm
import torch
from skimage.metrics import mean_squared_error
from neo.rawio.spikeglxrawio import scan_files


from .metrics_utils import calc_NRMSE
from .train_utils import train_model


class Generate:
    def __init__(self,opt_path):
        self.opt_path = opt_path
        self.read_yml()
        self.init_settings()

    def read_yml(self):
        # read YAML file
        with open(self.opt_path, 'r') as file:
            self.opt = yaml.safe_load(file)
        return self.opt

    def write_yml(self):
        with open(self.opt_path, 'w') as file:
            yaml.dump(self.opt, file)

    def init_settings(self):
        opt = self.opt

        # 1. train settings
        self.is_intp = opt['settings']['is_intp']
        self.is_random = opt['settings']['is_random']
        self.is_krig = opt['settings']['is_krig']
        self.slide_window = opt['settings']['slide_window']
        self.factor = opt['settings']['factor']  # proportion of abnormal channels
        self.rec_path = opt['settings']['rec_path']  # origin recording path
        self.base_path = opt['settings']['save_path']  # save path
        self.train_prop = opt['settings']['train_prop'] # partial frames used for training
        self.test_prop = opt['settings']['test_prop']

        # 2. restored settings
        self.method = opt['settings']['recon_method']  # reconstruction method
        self.is_whole = opt['settings']['is_whole']  # whole / partial pattern
        self.base_loss_path = opt['settings']['base_loss_path']  # base loss path

        # 3. model type
        self.model_type = opt['name']

        # 4. recording
        file_type = self.rec_path.split('.')
        if 'nwb' in file_type:  # nwb file
            self.recording = si.read_nwb(self.rec_path, load_recording=True, load_sorting=False)
            self.channel_ids = self.recording.channel_ids
        elif 'bin' in file_type:   # bin file
            self.recording = se.read_spikeglx(self.rec_path)
            self.channel_ids = np.arange(len(self.recording.channel_ids))
        elif 'h5' in file_type:  # h5 file
            self.recording = se.MEArecRecordingExtractor(self.rec_path)
            self.channel_ids = self.recording.channel_ids.astype(int)

        # 5. window
        self.window = None
        channel_num = self.opt.get('settings', {}).get('channel_num') 
        if channel_num is not None:
            self.window = channel_num
        else:
            self.window = len(self.channel_ids)
        
        # 5.1 pad_width
        self.pad_width = 0
        if self.window > len(self.channel_ids):
            self.pad_width = int(self.window-len(self.channel_ids))
        
        # 6. bad_chans / good_chans
        chl_path = self.opt.get('settings', {}).get('chl_path')
        if chl_path is not None:
            if not os.path.exists(os.path.join(chl_path,'factor_{}.npy'.format(self.factor))):
                chl_num = np.sum(self.channel_ids % self.factor == 1)
                bad_chans_index = np.random.choice(self.channel_ids, size=chl_num, replace=False)
                bad_chans = self.recording.channel_ids[bad_chans_index]
                good_chans = self.recording.channel_ids[~np.isin(self.recording.channel_ids, bad_chans)]
                np.save(os.path.join(chl_path,'factor_{}.npy'.format(self.factor)), bad_chans)
            else:
                bad_chans = np.load(os.path.join(chl_path,'factor_{}.npy'.format(self.factor)))
                good_chans = self.recording.channel_ids[~np.isin(self.recording.channel_ids, bad_chans)]
        else:
            bad_chans = self.recording.channel_ids[self.channel_ids % self.factor == 1]
            good_chans = self.recording.channel_ids[~np.isin(self.recording.channel_ids, bad_chans)]
        
        self.bad_chans = bad_chans
        self.good_chans = good_chans

        # 7. train_list / test_list
        self.total_sample = self.recording.get_total_samples()
        self.sliding_window = self.window - self.slide_window
        self.window_num = len(range(0, self.total_sample, self.sliding_window))

        start_frame_list = range(0, self.total_sample, self.sliding_window)
        train_num = int(len(start_frame_list) // self.train_prop)
        test_num = int(len(start_frame_list) // self.test_prop)

        if self.is_random:  # select random frames for training
            self.train_list = np.random.choice(start_frame_list, size=train_num, replace=False)  # train
            self.test_list = np.random.choice(np.setdiff1d(start_frame_list, self.train_list), size=test_num, replace=False)  # test
        else:
            self.train_list = start_frame_list[:train_num]
            self.test_list = start_frame_list[train_num:train_num + test_num]

            self.norm_path = self.opt.get('settings', {}).get('norm_path') 
            if self.norm_path is not None:
                gt_input = self.recording.get_traces(start_frame=0, end_frame=train_num) # 训练集
                self.mean_chl = np.mean(gt_input, axis=0)
                self.std_chl = np.std(gt_input, axis=0)
                np.save(os.path.join(self.norm_path,'factor_{}.npy'.format(self.factor)),[self.mean_chl, self.std_chl])

                # lq_input = self.recording.get_traces(start_frame=0, 
                #                                      end_frame=self.recording.get_total_samples(),
                #                                      channel_ids=self.good_chans)  # good chans 
                
                lq_input = self.recording.get_traces(start_frame=0, end_frame=self.recording.get_total_samples())
                self.lq_mean_chl = np.mean(lq_input, axis=0)
                self.lq_std_chl = np.std(lq_input, axis=0)
                np.save(os.path.join(self.norm_path,'lq_factor_{}.npy'.format(self.factor)),[self.lq_mean_chl, self.lq_mean_chl])
 
        # 8. input rec / save path
        if self.is_intp:
            if self.is_krig:  # kriging interpolation
                krig_weights = None
                self.input_rec = si.interpolate_bad_channels(self.recording, bad_chans, weights=krig_weights)
                self.traces_path = 'generate_krig_traces_factor_{}'.format(self.factor)
            else:  # set all data on the abnormal channel to 0.
                rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
                self.input_rec = si.interpolate_bad_channels(self.recording, bad_chans, weights=rmv_weights)
                self.traces_path = 'generate_zero_traces_factor_{}'.format(self.factor)
        else: # generate lr traces
            self.input_rec = self.recording.remove_channels(remove_channel_ids=bad_chans)
            self.traces_path = 'generate_lr_traces_factor_{}'.format(self.factor)


    def generate_train_list_mix(self):

        train_index_i, test_index_i = 0, 0
        mixture = self.opt['settings']['mixture'] # mixture files
        channel_num = self.opt['settings']['channel_num'] # channel num

        for mx_i in mixture:

            mx_path = mixture[mx_i]
            file_type = mx_path.split('.') # h5

            if 'nwb' in file_type:  # nwb file
                recording = si.read_nwb(mx_path, load_recording=True, load_sorting=False)
                channel_ids = recording.channel_ids
            elif 'bin' in file_type:   # bin file
                recording = se.read_spikeglx(mx_path)
                channel_ids = np.arange(len(recording.channel_ids))
            elif 'h5' in file_type:  # h5 file
                recording = se.MEArecRecordingExtractor(mx_path)
                channel_ids = recording.channel_ids.astype(int)
            else:
                print('process ending..')
                break
            
            bad_chans = recording.channel_ids[channel_ids % self.factor == 1]
            good_chans = recording.channel_ids[~np.isin(recording.channel_ids, bad_chans)]

            window = channel_num
            total_sample = recording.get_total_samples()
            sliding_window = window - self.slide_window
            start_frame_list = range(0, total_sample, sliding_window)
            train_num = int(len(start_frame_list) // self.train_prop)
            test_num = int(len(start_frame_list) // self.test_prop)

            if self.is_random:  # select random frames for training
                train_list = np.random.choice(start_frame_list, size=train_num, replace=False)  # train
                test_list = np.random.choice(np.setdiff1d(start_frame_list, train_list), size=test_num, replace=False)  # test
            else:
                train_list = start_frame_list[:train_num]
                test_list = start_frame_list[train_num:train_num + test_num]

            if self.is_intp:
                if self.is_krig:  # kriging interpolation
                    krig_weights = None
                    rec = si.interpolate_bad_channels(recording, bad_chans, weights=krig_weights)
                    traces_path = 'generate_krig_traces_factor_{}'.format(self.factor)
                else:  # set all data on the abnormal channel to 0.
                    rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
                    rec = si.interpolate_bad_channels(recording, bad_chans, weights=rmv_weights)
                    traces_path = 'generate_zero_traces_factor_{}'.format(self.factor)
            else: # generate lr traces
                rec = recording.remove_channels(remove_channel_ids=bad_chans)
                traces_path = 'generate_lr_traces_factor_{}'.format(self.factor)
                # window = len(rec.channel_ids) # sacle:(2,2)

            save_path = os.path.join(self.base_path, traces_path)
            train_path = os.path.join(save_path, 'train')  # training path
            test_path = os.path.join(save_path, 'test')  # testing path

            # save train dataset
            for j in tqdm(train_list):
                start_frame = j
                end_frame = j + window

                if end_frame > total_sample:
                    start_frame = total_sample - window
                    end_frame = total_sample

                lq_krig = rec.get_traces(start_frame=start_frame, end_frame=end_frame)  # kriging interpolated inputs
                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)  # ground truth frames
                # print('mse:{}'.format(mean_squared_error(lq_krig,gt)))

                pad_width = channel_num - lq_krig.shape[1]
                lq_krig = np.pad(lq_krig, pad_width=((0,0),(0,pad_width)), mode='constant', constant_values=0)
                gt = np.pad(gt, pad_width=((0,0),(0,pad_width)), mode='constant', constant_values=0)
                # cropped = lq_krig[:, :-pad_width] 

                train_index_path = os.path.join(train_path, str(train_index_i))
                if not os.path.exists(train_index_path):
                    os.makedirs(train_index_path)
                    np.save(os.path.join(train_index_path, 'lq.npy'), lq_krig)
                    np.save(os.path.join(train_index_path, 'gt.npy'), gt)

                train_index_i += 1
            print("{}: train datasets generate done!!".format(mx_path))

            # 存储测试训练集
            for j in tqdm(test_list):
                start_frame = j
                end_frame = j + window

                if end_frame > total_sample:
                    start_frame = total_sample - window
                    end_frame = total_sample

                lq_krig = rec.get_traces(start_frame=start_frame, end_frame=end_frame)  # kriging interpolated inputs
                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)  # ground truth frames
                # print('mse:{}'.format(mean_squared_error(lq_krig, gt)))

                pad_width = channel_num - lq_krig.shape[1]
                lq_krig = np.pad(lq_krig, pad_width=((0,0),(0,pad_width)), mode='constant', constant_values=0)
                gt = np.pad(gt, pad_width=((0,0),(0,pad_width)), mode='constant', constant_values=0)

                test_index_path = os.path.join(test_path, str(test_index_i))
                if not os.path.exists(test_index_path):
                    os.makedirs(test_index_path)
                    np.save(os.path.join(test_index_path, 'lq.npy'), lq_krig)
                    np.save(os.path.join(test_index_path, 'gt.npy'), gt)
                
                test_index_i += 1
            print("{}: test datasets generate done!!".format(mx_path))
        

    def generate_train_list(self):
        
        save_path = os.path.join(self.base_path, self.traces_path)
        train_path = os.path.join(save_path, 'train')  # training path
        test_path = os.path.join(save_path, 'test')  # testing path

        # save train dataset
        for j in tqdm(self.train_list):
            start_frame = j
            end_frame = j + self.window

            if end_frame > self.total_sample:
                start_frame = self.total_sample - self.window
                end_frame = self.total_sample

            lq_krig = self.input_rec.get_traces(start_frame=start_frame, end_frame=end_frame)  # kriging interpolated inputs
            gt = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame)  # ground truth frames
            # print('mse:{}'.format(mean_squared_error(lq_krig,gt))

            if self.norm_path is not None:
                lq_krig = (lq_krig - self.mean_chl) / self.std_chl
                gt = (gt - self.mean_chl) / self.std_chl

            lq_krig = np.pad(lq_krig,((0, 0),(0, self.pad_width)), constant_values=0)
            gt = np.pad(gt,((0, 0),(0, self.pad_width)), constant_values=0)

            train_index_path = os.path.join(train_path, str(start_frame))
            if not os.path.exists(train_index_path):
                os.makedirs(train_index_path)
            np.save(os.path.join(train_index_path, 'lq.npy'), lq_krig)
            np.save(os.path.join(train_index_path, 'gt.npy'), gt)

        print("train datasets generate done!!")

        # 存储测试训练集
        for j in tqdm(self.test_list):
            start_frame = j
            end_frame = j + self.window

            if end_frame > self.total_sample:
                start_frame = self.total_sample - self.window
                end_frame = self.total_sample

            lq_krig = self.input_rec.get_traces(start_frame=start_frame, end_frame=end_frame)  # kriging interpolated inputs
            gt = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame)  # ground truth frames
            # print('mse:{}'.format(mean_squared_error(lq_krig, gt)))

            if self.norm_path is not None: 
                lq_krig = (lq_krig - self.mean_chl) / self.std_chl
                gt = (gt - self.mean_chl) / self.std_chl

            lq_krig = np.pad(lq_krig,((0, 0),(0, self.pad_width)), constant_values=0)
            gt = np.pad(gt,((0, 0),(0, self.pad_width)), constant_values=0)

            test_index_path = os.path.join(test_path, str(start_frame))
            if not os.path.exists(test_index_path):
                os.makedirs(test_index_path)
            np.save(os.path.join(test_index_path, 'lq.npy'), lq_krig)
            np.save(os.path.join(test_index_path, 'gt.npy'), gt)
        print("recon datasets generate done!!")

    def generate_model(self):
        opt = self.opt
        parameters,load_arch = None,None
        model_type = self.model_type
        # spkres / restormer
        if self.model_type == 'SpkRes' or self.model_type == 'Restormer':
            parameters = {'inp_channels': opt['network_g']['inp_channels'],
                          'out_channels':  opt['network_g']['out_channels'],
                          'dim': opt['network_g']['dim'],
                          'num_blocks': opt['network_g']['num_blocks'],
                          'num_refinement_blocks': opt['network_g']['num_refinement_blocks'],
                          'heads': opt['network_g']['heads'],
                          'ffn_expansion_factor': opt['network_g']['ffn_expansion_factor'],
                          'bias': opt['network_g']['bias'],
                          'LayerNorm_type': opt['network_g']['LayerNorm_type'],
                          'dual_pixel_task': opt['network_g']['dual_pixel_task']}
            model_type = 'Restormer'
            load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))

        elif self.model_type == 'SpkRecon' or self.model_type == 'SwinIR':
            parameters = {'in_chans': opt['network_g']['in_chans'],
                          'upscale': opt['network_g']['upscale'],
                          'img_size': opt['network_g']['img_size'],
                          'embed_dim': opt['network_g']['embed_dim'],
                          'window_size': opt['network_g']['window_size'],
                          'img_range': opt['network_g']['img_range'],
                          'num_heads': opt['network_g']['num_heads'],
                          'depths': opt['network_g']['depths'],
                          'upsampler': opt['network_g']['upsampler'],
                          'resi_connection': opt['network_g']['resi_connection'],
                          'mlp_ratio': opt['network_g']['mlp_ratio']}
            model_type = 'SpkRecon'
            load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'spkrecon_arch.py'))

        elif self.model_type == 'EDSR':
            parameters = {'num_channels': opt['network_g']['num_channels'],
                          'factor': opt['network_g']['factor'],
                          'width': opt['network_g']['width'],
                          'depth': opt['network_g']['depth'],
                          'kernel_size': opt['network_g']['kernel_size']}
            load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'edsr_arch.py'))

        elif self.model_type == 'VAESR':
            parameters = {'num_channels': opt['network_g']['num_channels'],
                          'width': opt['network_g']['width'],
                          'depth': opt['network_g']['depth'],
                          'kernel_size': opt['network_g']['kernel_size'],
                          'latent_dim': opt['network_g'].get('latent_dim', 32),
                          'beta': opt['network_g'].get('beta', 1.0),
                          'kl_anneal_start': opt['network_g'].get('kl_anneal_start', 0.1)}
            load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'vae_arch.py'))

        self.model = load_arch[model_type](**parameters)
        return self.model

    def generate_train_weights(self):
        train_model(self.opt_path)
       
    def generate_recon_binfile(self,res_file_name=None,weights_path='weights'): # bin file

        ###################### parameters ######################
        method = self.method  # reconstruction method
        factor = self.factor  # proportion of abnormal channels
        is_whole = self.is_whole  # whole / partial
        slide_window = self.slide_window  # sliding window
        rec_path = self.rec_path  # origin recording path
        train_prop = self.train_prop  # partial mode

        base_loss_path = self.base_loss_path  # base loss path

        ###################### model ######################
        model = self.generate_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        ###################### weight #######################
        weights = weights_path  # 网络模型权重
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['params'])
        model.eval()

        ###################### origin recording #######################
        recording = se.read_spikeglx(rec_path)
        channel_ids = np.arange(len(recording.channel_ids))
        bad_chans = recording.channel_ids[channel_ids % factor == 1]
        good_chans = recording.channel_ids[~np.isin(recording.channel_ids, bad_chans)]
        window = len(channel_ids)
        total_sample = recording.get_total_samples()
        sliding_window = window - slide_window
        window_num = len(range(0, total_sample, sliding_window))

        # (1) kriging
        krig_weights = None
        krig_rec = si.interpolate_bad_channels(recording, bad_chans, weights=krig_weights)

        # (2) remove: interpolate all damaged channels to 0
        rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
        rmv_rec = si.interpolate_bad_channels(recording, bad_chans, weights=rmv_weights)

        ###################### intp recording #######################
        # pattern
        if is_whole:
            partial_sample = 0
            pattern = 'whole'
        else:
            partial_sample = total_sample // train_prop
            pattern = 'partial'

        # loss save path
        loss_path = os.path.join(base_loss_path, '{}_factor{}'.format(pattern, factor))
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)

        # MSE loss
        signals_info_dict = {}
        signals_info_list = scan_files(os.path.join(res_file_name))
        for info in signals_info_list:
            # key is (seg_index, stream_name)
            key = (info["seg_index"], info["stream_name"])
            assert key not in signals_info_dict
            signals_info_dict[key] = info

            num_chan = info["num_chan"]
            sample_length = info["sample_length"]

            # create memmap
            data = np.memmap(info["bin_file"], dtype="int16", mode="r+", offset=0, order="C",
                             shape=(sample_length, num_chan))

            start_sample = 0
            step = 0
            mse_loss_dict = np.zeros([window_num])
            nrmse_loss_dict = np.zeros([window_num])
            for j in tqdm(range(start_sample, total_sample, sliding_window)):
                start_frame = j
                end_frame = j + window

                if end_frame > total_sample:
                    start_frame = total_sample - window
                    end_frame = total_sample

                lq_krig = krig_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                lq_rmv = rmv_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                gt = np.asarray(recording.get_traces(start_frame=start_frame, end_frame=end_frame)) # ground truth
                # intp_gt = np.asarray(intp_rec.get_traces(start_frame=start_frame, end_frame=end_frame))
                # print('test_intp:{}'.format(mean_squared_error(gt, intp_gt)))

                intp_trace = None
                if end_frame <= partial_sample:
                    intp_trace = gt
                else:
                    if method == 'intp':
                        lq_krig = np.expand_dims(lq_krig, axis=2)
                        lq_krig = torch.from_numpy(lq_krig).float().permute(2, 0, 1).unsqueeze(0).to(device)
                        intp_trace = model(lq_krig)
                        intp_trace = intp_trace.squeeze().cpu().detach().numpy().astype(np.int16)

                        data[start_frame:end_frame, :-1] = intp_trace
                        data.flush()

                    elif method == 'krig':
                        intp_trace = lq_krig
                    elif method == 'remove':
                        intp_trace = rmv_rec.get_traces(start_frame=start_frame, end_frame=end_frame)

                    # calc loss
                mse_loss = mean_squared_error(gt, intp_trace)
                mse_loss_dict[step] = mse_loss
                print("{}_mse_loss:{}".format(method, mse_loss))

                # calc NRMSE
                nrmse_loss = calc_NRMSE(y_pred=intp_trace.astype(float), y_gt=gt.astype(float))
                nrmse_loss_dict[step] = nrmse_loss
                print("{}_nrmse_loss:{}".format(method, nrmse_loss))

                step = step + 1

            np.save(os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(method)), mse_loss_dict)
            np.save(os.path.join(loss_path, '{}_nrmse_loss_dict.npy'.format(method)), nrmse_loss_dict)

    def generate_recon_h5file(self, res_file_name='res_file.h5',mode='zero_shot_gcl'): # restored h5 file

        if self.norm_path is not None:
            # zero-shot
            if mode == 'zero_shot_gcl': 
                mean_chl = self.lq_mean_chl
                std_chl = self.lq_std_chl
            else:
                mean_chl = self.mean_chl
                std_chl = self.std_chl

        ###################### model ######################
        model = self.generate_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        ###################### weight #######################
        weights_path = self.opt.get('settings', {}).get('weights_path')
        if weights_path is not None:
            checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['params'])
        model.eval()

        ###################### intp recording #######################
        # pattern
        if self.is_whole:
            partial_sample = 0
            pattern = 'whole'
        else:
            partial_sample = self.total_sample // self.train_prop
            pattern = 'partial'

        # loss save path
        loss_path = os.path.join(self.base_loss_path, '{}_factor{}'.format(pattern, self.factor))
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)
        else:
            file_name = os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(self.method))
            if os.path.exists(file_name):
                warnings.warn(f"The file '{file_name}' exists in the folder.", Warning)

        # MSE loss
        mse_loss_dict = np.zeros([self.window_num])
        nrmse_loss_dict = np.zeros([self.window_num])

        # restored path
        restored_path = res_file_name
        with h5py.File(restored_path, 'r+') as io: # h5

            # restored_nwbfile = io.read()
            restored_nwbfile = io
            keys_list = list(io.keys())
            print(keys_list)

            # h5 file
            res_rec = restored_nwbfile['recordings'] # time × channel
            print('size of rec: {}'.format(res_rec.shape))
            
            step = 0
            for j in tqdm(range(0, self.total_sample, self.sliding_window)):
                start_frame = j
                end_frame = j + self.window

                if end_frame > self.total_sample:
                    start_frame = self.total_sample - self.window
                    end_frame = self.total_sample

                print("start_frame:" + str(start_frame) + "," + "end_frame:" + str(end_frame))

                gt = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame) # ground truth
                if end_frame <= partial_sample:
                    intp_trace = gt
                else:
                    lq = self.input_rec.get_traces(start_frame=start_frame, end_frame=end_frame)

                    # z-score
                    if self.norm_path is not None:
                        lq = (lq - mean_chl) / std_chl
                    
                    # pad_width = self.window - lq.shape[1]
                    lq = np.pad(lq, pad_width=((0,0),(0,self.pad_width)), mode='constant', constant_values=0)

                    lq = np.expand_dims(lq, axis=2)
                    lq = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0).to(device)

                    # gt = np.pad(gt, pad_width=((0,0),(0,pad_width)), mode='constant', constant_values=0)

                    intp_trace = model(lq)
                    intp_trace = intp_trace.squeeze().cpu().detach().numpy() # output

                    if self.pad_width > 0:
                        intp_trace = intp_trace[:, :-self.pad_width]

                    if self.norm_path is not None:
                        intp_trace = (intp_trace*std_chl) + mean_chl                 

                # calc mse
                mse_loss = mean_squared_error(gt, intp_trace)
                mse_loss_dict[step] = mse_loss
                print("{}_mse_loss:{}".format(self.method, mse_loss))

                # calc NRMSE
                nrmse_loss = calc_NRMSE(y_pred=intp_trace.astype(float), y_gt=gt.astype(float))
                nrmse_loss_dict[step] = nrmse_loss
                print("{}_nrmse_loss:{}".format(self.method, nrmse_loss))

                # restored data
                res_rec[start_frame:end_frame] = intp_trace
                step = step + 1

            np.save(os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(self.method)), mse_loss_dict)
            np.save(os.path.join(loss_path, '{}_nrmse_loss_dict.npy'.format(self.method)), nrmse_loss_dict)

    
    def generate_recon_nwbfile(self,res_file_name='res_file.nwb',weights_path='weights'): # restored nwb file

        ###################### parameters ######################
        method = self.method  # reconstruction method
        factor = self.factor  # proportion of abnormal channels
        is_whole = self.is_whole  # whole / partial
        slide_window = self.slide_window  # sliding window
        rec_path = self.rec_path  # origin recording path
        train_prop = self.train_prop # partial mode

        base_loss_path = self.base_loss_path  # base loss path

        ###################### model ######################
        model = self.generate_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        ###################### weight #######################
        weights = weights_path 
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['params'])
        model.eval()

        ###################### origin recording #######################
        file_type = self.rec_path.split('.')
        if 'h5' in file_type:  
            recording = se.MEArecRecordingExtractor(self.rec_path)
        else:
            recording = si.read_nwb(rec_path, load_recording=True, load_sorting=False)

        # channel_ids = recording.channel_ids
        channel_ids = recording.channel_ids.astype(int)
        bad_chans = recording.channel_ids[channel_ids % factor == 1]
        good_chans = recording.channel_ids[~np.isin(channel_ids, bad_chans)]
        window = len(channel_ids)

        # (1) kriging
        krig_weights = None
        krig_rec = si.interpolate_bad_channels(recording, bad_chans, weights=krig_weights)

        # (2) remove: interpolate all damaged channels to 0
        rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
        rmv_rec = si.interpolate_bad_channels(recording, bad_chans, weights=rmv_weights)

        # (3) umsample
        lr_rec = recording.remove_channels(remove_channel_ids=bad_chans)

        ###################### intp recording #######################
        total_sample = recording.get_total_samples()
        sliding_window = window - slide_window
        window_num = len(range(0, total_sample, sliding_window))

        # pattern
        if is_whole:
            partial_sample = 0
            pattern = 'whole'
        else:
            partial_sample = total_sample // train_prop
            pattern = 'partial'

        # loss save path
        loss_path = os.path.join(base_loss_path, '{}_factor{}'.format(pattern, factor))
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)
        else:
            file_name = os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(method))
            if os.path.exists(file_name):
                warnings.warn(f"The file '{file_name}' exists in the folder.", Warning)

        # MSE loss
        mse_loss_dict = np.zeros([window_num])
        nrmse_loss_dict = np.zeros([window_num])

        # restored path
        restored_path = res_file_name
        with NWBHDF5IO(restored_path, "r+") as io: # h5 / nwb
            restored_nwbfile = io.read()
            step = 0
            for j in tqdm(range(0, total_sample, sliding_window)):
                start_frame = j
                end_frame = j + window

                if end_frame > total_sample:
                    start_frame = total_sample - window
                    end_frame = total_sample

                print("start_frame:" + str(start_frame) + "," + "end_frame:" + str(end_frame))

                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame) # ground truth
                if end_frame <= partial_sample:
                    intp_trace = gt
                else:
                    if method == 'krig':
                        intp_trace = krig_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                    elif method == 'remove':
                        intp_trace = rmv_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                    else:
                        if self.is_intp:
                            if self.is_krig: # krig input
                                lq_rec = krig_rec
                            else: # rmv input
                                lq_rec = rmv_rec
                        else: # unsample input
                            lq_rec = lr_rec

                        lq = lq_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                        lq = np.expand_dims(lq, axis=2)
                        lq = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0).to(device)
                        intp_trace = model(lq)
                        intp_trace = intp_trace.squeeze().cpu().detach().numpy()

                # calc mse
                mse_loss = mean_squared_error(gt, intp_trace)
                mse_loss_dict[step] = mse_loss
                print("{}_mse_loss:{}".format(method, mse_loss))

                # calc NRMSE
                nrmse_loss = calc_NRMSE(y_pred=intp_trace.astype(float), y_gt=gt.astype(float))
                nrmse_loss_dict[step] = nrmse_loss
                print("{}_nrmse_loss:{}".format(method, nrmse_loss))

                # restored data
                restored_nwbfile.acquisition["ElectricalSeries"].data[start_frame:end_frame] = intp_trace

                step = step + 1

            np.save(os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(method)), mse_loss_dict)
            np.save(os.path.join(loss_path, '{}_nrmse_loss_dict.npy'.format(method)), nrmse_loss_dict)
