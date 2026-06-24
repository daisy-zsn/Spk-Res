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
    def __init__(self, opt_path):
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

    # ==================== Helper Methods ====================

    @staticmethod
    def _load_recording(file_path):
        """Load recording from various file formats."""
        file_type = file_path.split('.')
        if 'nwb' in file_type:
            recording = si.read_nwb(file_path, load_recording=True, load_sorting=False)
            channel_ids = recording.channel_ids
        elif 'bin' in file_type:
            recording = se.read_spikeglx(file_path)
            channel_ids = np.arange(len(recording.channel_ids))
        elif 'h5' in file_type:
            recording = se.MEArecRecordingExtractor(file_path)
            channel_ids = recording.channel_ids.astype(int)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return recording, channel_ids

    @staticmethod
    def _compute_window_bounds(j, window, total_sample):
        """Compute start_frame and end_frame with boundary handling."""
        start_frame = j
        end_frame = j + window
        if end_frame > total_sample:
            start_frame = total_sample - window
            end_frame = total_sample
        return start_frame, end_frame

    @staticmethod
    def _save_trace_pair(lq, gt, save_path, index):
        """Save lq and gt traces to disk."""
        index_path = os.path.join(save_path, str(index))
        if not os.path.exists(index_path):
            os.makedirs(index_path)
            np.save(os.path.join(index_path, 'lq.npy'), lq)
            np.save(os.path.join(index_path, 'gt.npy'), gt)

    @staticmethod
    def _calc_and_save_losses(gt, intp_trace, method, step, mse_loss_dict, nrmse_loss_dict):
        """Calculate MSE and NRMSE losses, print and store them."""
        mse_loss = mean_squared_error(gt, intp_trace)
        mse_loss_dict[step] = mse_loss
        print("{}_mse_loss:{}".format(method, mse_loss))

        nrmse_loss = calc_NRMSE(y_pred=intp_trace.astype(float), y_gt=gt.astype(float))
        nrmse_loss_dict[step] = nrmse_loss
        print("{}_nrmse_loss:{}".format(method, nrmse_loss))

    def _prepare_model(self):
        """Load model, move to device, and set to eval mode."""
        model = self.generate_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model, device

    def _prepare_loss_path(self, pattern, factor):
        """Create and return loss save path."""
        loss_path = os.path.join(self.base_loss_path, '{}_{}'.format(pattern, factor))
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)
        else:
            file_name = os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(self.method))
            if os.path.exists(file_name):
                warnings.warn(f"The file '{file_name}' exists in the folder.", Warning)
        return loss_path

    def _get_bad_good_chans(self, recording, channel_ids):
        """Get bad and good channels based on factor."""
        bad_chans = recording.channel_ids[channel_ids % self.factor == 1]
        good_chans = recording.channel_ids[~np.isin(recording.channel_ids, bad_chans)]
        return bad_chans, good_chans

    def _get_intp_recs(self, recording, bad_chans, good_chans):
        """Create kriging and remove interpolation recordings."""
        krig_weights = None
        krig_rec = si.interpolate_bad_channels(recording, bad_chans, weights=krig_weights)
        rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
        rmv_rec = si.interpolate_bad_channels(recording, bad_chans, weights=rmv_weights)
        return krig_rec, rmv_rec

    def _process_window_loop(self, total_sample, window, sliding_window, window_num,
                              recording, input_rec, model, device, partial_sample,
                              loss_path, method, res_file, write_func,
                              norm_data=None):
        """Core sliding window processing loop for reconstruction methods."""
        mse_loss_dict = np.zeros([window_num])
        nrmse_loss_dict = np.zeros([window_num])

        step = 0
        for j in tqdm(range(0, total_sample, sliding_window)):
            start_frame, end_frame = self._compute_window_bounds(j, window, total_sample)
            print("start_frame:" + str(start_frame) + "," + "end_frame:" + str(end_frame))

            gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)

            if end_frame <= partial_sample:
                intp_trace = gt
            else:
                lq = input_rec.get_traces(start_frame=start_frame, end_frame=end_frame)

                # z-score normalization
                if norm_data is not None:
                    lq = (lq - norm_data[0]) / norm_data[1]

                pad_width = window - lq.shape[1]
                lq = np.pad(lq, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)

                lq = np.expand_dims(lq, axis=2)
                lq = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0).to(device)

                intp_trace = model(lq)
                intp_trace = intp_trace.squeeze().cpu().detach().numpy()

                # denormalize
                if norm_data is not None:
                    intp_trace = (intp_trace * norm_data[1]) + norm_data[0]

                if pad_width > 0:
                    intp_trace = intp_trace[:, :-pad_width]

            self._calc_and_save_losses(gt, intp_trace, method, loss_path, step,
                                       mse_loss_dict, nrmse_loss_dict)

            # write restored data
            write_func(res_file, start_frame, end_frame, intp_trace)

            step = step + 1

        np.save(os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(method)), mse_loss_dict)
        np.save(os.path.join(loss_path, '{}_nrmse_loss_dict.npy'.format(method)), nrmse_loss_dict)

    # ==================== Initialization ====================

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
        self.train_prop = opt['settings']['train_prop']  # partial frames used for training
        self.test_prop = opt['settings']['test_prop']

        # 2. restored settings
        self.method = opt['settings']['recon_method']  # reconstruction method
        self.is_whole = opt['settings']['is_whole']  # whole / partial pattern
        self.base_loss_path = opt['settings']['base_loss_path']  # base loss path

        # 3. model type
        self.model_type = opt['name']

        # 4. recording
        self.recording, self.channel_ids = self._load_recording(self.rec_path)

        # 5. window
        self.window = None
        channel_num = self.opt.get('settings', {}).get('channel_num')
        if channel_num is not None:
            self.window = channel_num
        else:
            self.window = len(self.channel_ids)

        # 6. bad_chans / good_chans
        chl_path = self.opt.get('settings', {}).get('chl_path')
        if chl_path is not None:
            if not os.path.exists(os.path.join(chl_path, 'factor_{}.npy'.format(self.factor))):
                chl_num = np.sum(self.channel_ids % self.factor == 1)
                bad_chans_index = np.random.choice(self.channel_ids, size=chl_num, replace=False)
                bad_chans = self.recording.channel_ids[bad_chans_index]
                good_chans = self.recording.channel_ids[~np.isin(self.recording.channel_ids, bad_chans)]
                np.save(os.path.join(chl_path, 'factor_{}.npy'.format(self.factor)), bad_chans)
            else:
                bad_chans = np.load(os.path.join(chl_path, 'factor_{}.npy'.format(self.factor)))
                good_chans = self.recording.channel_ids[~np.isin(self.recording.channel_ids, bad_chans)]
        else:
            bad_chans, good_chans = self._get_bad_good_chans(self.recording, self.channel_ids)

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
                if not os.path.exists(os.path.join(self.norm_path, 'factor_{}.npy'.format(self.factor))):
                    gt_input = self.recording.get_traces(start_frame=0, end_frame=train_num)
                    self.mean_chl = np.mean(gt_input, axis=0)
                    self.std_chl = np.std(gt_input, axis=0)
                    np.save(os.path.join(self.norm_path, 'factor_{}.npy'.format(self.factor)), [self.mean_chl, self.std_chl])
                else:
                    zscore_dt = np.load(os.path.join(self.norm_path, 'factor_{}.npy'.format(self.factor)))
                    self.mean_chl, self.std_chl = zscore_dt[0], zscore_dt[1]

        # 8. input rec / save path
        if self.is_intp:
            if self.is_krig:  # kriging interpolation
                krig_weights = None
                self.input_rec = si.interpolate_bad_channels(self.recording, self.bad_chans, weights=krig_weights)
                self.traces_path = 'generate_krig_traces_factor_{}'.format(self.factor)
            else:  # set all data on the abnormal channel to 0.
                rmv_weights = np.zeros([len(self.good_chans), len(self.bad_chans)])
                self.input_rec = si.interpolate_bad_channels(self.recording, self.bad_chans, weights=rmv_weights)
                self.traces_path = 'generate_zero_traces_factor_{}'.format(self.factor)
        else:  # generate lr traces
            self.input_rec = self.recording.remove_channels(remove_channel_ids=self.bad_chans)
            self.traces_path = 'generate_lr_traces_factor_{}'.format(self.factor)

    # ==================== Train List Generation ====================

    def _save_traces_from_list(self, trace_list, rec, recording, save_path, is_train=True,
                                norm_data=None, window=None, total_sample=None, channel_num=None):
        """Save traces for a list of start frames (used by both train and test)."""
        if window is None:
            window = self.window
        if total_sample is None:
            total_sample = self.total_sample

        for idx, j in enumerate(tqdm(trace_list)):
            start_frame, end_frame = self._compute_window_bounds(j, window, total_sample)

            lq = rec.get_traces(start_frame=start_frame, end_frame=end_frame)
            gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)

            if norm_data is not None:
                lq = (lq - norm_data[0]) / norm_data[1]
                gt = (gt - norm_data[0]) / norm_data[1]

            if channel_num is not None:
                pad_width = channel_num - lq.shape[1]
                lq = np.pad(lq, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                gt = np.pad(gt, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)

            self._save_trace_pair(lq, gt, save_path, start_frame)

    def generate_train_list(self):
        save_path = os.path.join(self.base_path, self.traces_path)
        train_path = os.path.join(save_path, 'train')  # training path
        test_path = os.path.join(save_path, 'test')  # testing path

        norm_data = (self.mean_chl, self.std_chl) if self.norm_path is not None else None

        # save train dataset
        self._save_traces_from_list(self.train_list, self.input_rec, self.recording,
                                     train_path, is_train=True, norm_data=norm_data)
        print("train datasets generate done!!")

        # save test dataset
        self._save_traces_from_list(self.test_list, self.input_rec, self.recording,
                                     test_path, is_train=False, norm_data=norm_data)
        print("recon datasets generate done!!")

    def generate_train_list_mix(self):
        train_index_i, test_index_i = 0, 0
        mixture = self.opt['settings']['mixture']  # mixture files
        channel_num = self.opt['settings']['channel_num']  # channel num

        for mx_i in mixture:
            mx_path = mixture[mx_i]
            recording, channel_ids = self._load_recording(mx_path)

            bad_chans, good_chans = self._get_bad_good_chans(recording, channel_ids)

            window = channel_num
            total_sample = recording.get_total_samples()
            sliding_window = window - self.slide_window
            start_frame_list = range(0, total_sample, sliding_window)
            train_num = int(len(start_frame_list) // self.train_prop)
            test_num = int(len(start_frame_list) // self.test_prop)

            if self.is_random:  # select random frames for training
                train_list = np.random.choice(start_frame_list, size=train_num, replace=False)
                test_list = np.random.choice(np.setdiff1d(start_frame_list, train_list), size=test_num, replace=False)
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
            else:  # generate lr traces
                rec = recording.remove_channels(remove_channel_ids=bad_chans)
                traces_path = 'generate_lr_traces_factor_{}'.format(self.factor)

            save_path = os.path.join(self.base_path, traces_path)
            train_path = os.path.join(save_path, 'train')
            test_path = os.path.join(save_path, 'test')

            # save train dataset
            for j in tqdm(train_list):
                start_frame, end_frame = self._compute_window_bounds(j, window, total_sample)

                lq = rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)

                pad_width = channel_num - lq.shape[1]
                lq = np.pad(lq, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                gt = np.pad(gt, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)

                self._save_trace_pair(lq, gt, train_path, train_index_i)
                train_index_i += 1
            print("{}: train datasets generate done!!".format(mx_path))

            # save test dataset
            for j in tqdm(test_list):
                start_frame, end_frame = self._compute_window_bounds(j, window, total_sample)

                lq = rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)

                pad_width = channel_num - lq.shape[1]
                lq = np.pad(lq, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                gt = np.pad(gt, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)

                self._save_trace_pair(lq, gt, test_path, test_index_i)
                test_index_i += 1
            print("{}: test datasets generate done!!".format(mx_path))

    # ==================== Model Generation ====================

    def generate_model(self):
        opt = self.opt
        parameters, load_arch = None, None

        if self.model_type in ('SpkRes', 'Restormer'):
            params_keys = ['inp_channels', 'out_channels', 'dim', 'num_blocks',
                           'num_refinement_blocks', 'heads', 'ffn_expansion_factor',
                           'bias', 'LayerNorm_type', 'dual_pixel_task']
            parameters = {k: opt['network_g'][k] for k in params_keys}
            load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))

        elif self.model_type in ('SpkRecon', 'SwinIR'):
            params_keys = ['in_chans', 'upscale', 'img_size', 'embed_dim',
                           'window_size', 'img_range', 'num_heads', 'depths',
                           'upsampler', 'resi_connection', 'mlp_ratio']
            parameters = {k: opt['network_g'][k] for k in params_keys}
            load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'spkrecon_arch.py'))

        elif self.model_type == 'EDSR':
            params_keys = ['num_channels', 'factor', 'width', 'depth', 'kernel_size']
            parameters = {k: opt['network_g'][k] for k in params_keys}
            load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'edsr_arch.py'))

        self.model = load_arch[self.model_type](**parameters)
        return self.model

    def generate_train_weights(self):
        train_model(self.opt_path)

    # ==================== Reconstruction Methods ====================

    def generate_recon_binfile(self, res_file_name=None, weights_path='weights'):
        ###################### parameters ######################
        method = self.method
        factor = self.factor
        is_whole = self.is_whole
        slide_window = self.slide_window
        rec_path = self.rec_path
        train_prop = self.train_prop
        base_loss_path = self.base_loss_path

        ###################### model ######################
        model, device = self._prepare_model()

        ###################### weight #######################
        checkpoint = torch.load(weights_path)
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
        krig_rec, rmv_rec = self._get_intp_recs(recording, bad_chans, good_chans)

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
        mse_loss_dict = np.zeros([window_num])
        nrmse_loss_dict = np.zeros([window_num])

        signals_info_dict = {}
        signals_info_list = scan_files(os.path.join(res_file_name))
        for info in signals_info_list:
            key = (info["seg_index"], info["stream_name"])
            assert key not in signals_info_dict
            signals_info_dict[key] = info

            num_chan = info["num_chan"]
            sample_length = info["sample_length"]

            data = np.memmap(info["bin_file"], dtype="int16", mode="r+", offset=0, order="C",
                             shape=(sample_length, num_chan))

            start_sample = 0
            step = 0
            for j in tqdm(range(start_sample, total_sample, sliding_window)):
                start_frame, end_frame = self._compute_window_bounds(j, window, total_sample)

                lq_krig = krig_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                lq_rmv = rmv_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                gt = np.asarray(recording.get_traces(start_frame=start_frame, end_frame=end_frame))

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

                self._calc_and_save_losses(gt, intp_trace, method, loss_path, step,
                                           mse_loss_dict, nrmse_loss_dict)
                step = step + 1

            np.save(os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(method)), mse_loss_dict)
            np.save(os.path.join(loss_path, '{}_nrmse_loss_dict.npy'.format(method)), nrmse_loss_dict)

    def generate_recon_h5file(self, res_file_name='res_file.h5'):
        ###################### model ######################
        model, device = self._prepare_model()

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
        loss_path = self._prepare_loss_path(pattern, self.factor)

        norm_data = (self.mean_chl, self.std_chl) if self.norm_path is not None else None

        # Define write function for h5 file
        def write_h5(res_file, start_frame, end_frame, intp_trace):
            res_file[start_frame:end_frame] = intp_trace

        # restored path
        with h5py.File(res_file_name, 'r+') as io:
            res_rec = io['recordings']
            print('size of rec: {}'.format(res_rec.shape))

            self._process_window_loop(
                total_sample=self.total_sample,
                window=self.window,
                sliding_window=self.sliding_window,
                window_num=self.window_num,
                recording=self.recording,
                input_rec=self.input_rec,
                model=model,
                device=device,
                partial_sample=partial_sample,
                loss_path=loss_path,
                method=self.method,
                res_file=res_rec,
                write_func=write_h5,
                norm_data=norm_data
            )

    def generate_recon_nwbfile(self, res_file_name='res_file.nwb', weights_path='weights'):
        ###################### parameters ######################
        method = self.method
        factor = self.factor
        is_whole = self.is_whole
        slide_window = self.slide_window
        rec_path = self.rec_path
        train_prop = self.train_prop
        base_loss_path = self.base_loss_path

        ###################### model ######################
        model, device = self._prepare_model()

        ###################### weight #######################
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['params'])
        model.eval()

        ###################### origin recording #######################
        recording, channel_ids = self._load_recording(rec_path)
        bad_chans, good_chans = self._get_bad_good_chans(recording, channel_ids)
        window = len(channel_ids)

        # (1) kriging & remove
        krig_rec, rmv_rec = self._get_intp_recs(recording, bad_chans, good_chans)

        # (2) unsample
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
        with NWBHDF5IO(res_file_name, "r+") as io:
            restored_nwbfile = io.read()
            step = 0
            for j in tqdm(range(0, total_sample, sliding_window)):
                start_frame, end_frame = self._compute_window_bounds(j, window, total_sample)
                print("start_frame:" + str(start_frame) + "," + "end_frame:" + str(end_frame))

                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)

                if end_frame <= partial_sample:
                    intp_trace = gt
                else:
                    if method == 'krig':
                        intp_trace = krig_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                    elif method == 'remove':
                        intp_trace = rmv_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                    else:
                        if self.is_intp:
                            lq_rec = krig_rec if self.is_krig else rmv_rec
                        else:
                            lq_rec = lr_rec

                        lq = lq_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                        lq = np.expand_dims(lq, axis=2)
                        lq = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0).to(device)
                        intp_trace = model(lq)
                        intp_trace = intp_trace.squeeze().cpu().detach().numpy()

                self._calc_and_save_losses(gt, intp_trace, method, loss_path, step,
                                           mse_loss_dict, nrmse_loss_dict)

                # restored data
                restored_nwbfile.acquisition["ElectricalSeries"].data[start_frame:end_frame] = intp_trace
                step = step + 1

            np.save(os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(method)), mse_loss_dict)
            np.save(os.path.join(loss_path, '{}_nrmse_loss_dict.npy'.format(method)), nrmse_loss_dict)
