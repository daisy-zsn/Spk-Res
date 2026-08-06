"""Generate v2 -- refactored data generation / model reconstruction utility class.

Key improvements over the original ``generate_utils.py``:

1. **Eliminated duplication**: common logic such as recording loading, window
   boundary calculation, MSE/NRMSE computation, loss saving, and interpolated
   recording construction is unified into helper methods, reused across the
   bin / h5 / nwb formats.
2. **Bug fixes**:
   - Fixed the indentation error of the ``norm_path`` assignment (the original
     code mistakenly placed it inside the ``else`` branch, causing
     ``self.norm_path`` to be undefined when ``is_random=True``);
   - Fixed the lq normalization statistics saving bug where ``lq_mean_chl``
     was incorrectly written as ``lq_std_chl``;
   - In ``generate_recon_h5file``, when ``weights_path`` is ``None`` the
     ``checkpoint`` was uninitialized and raised a NameError; now it raises an
     explicit error instead.
3. **Compatibility with new yml fields**: ``channel_num`` / ``chl_path`` /
   ``norm_path`` / ``weights_path`` are all read via ``.get()``, so old yml
   files (without these fields) still work.
4. **Unified sliding-window reconstruction loop**: the bin / h5 / nwb output
   formats share the same ``_run_recon_loop``, differentiated only by the
   ``get_intp_trace`` / ``write_func`` callbacks.
"""
import os
import warnings

import h5py
import numpy as np
import spikeinterface.full as si
import spikeinterface.extractors as se
import torch
import yaml
from neo.rawio.spikeglxrawio import scan_files
from pynwb import NWBHDF5IO
from runpy import run_path
from skimage.metrics import mean_squared_error
from tqdm import tqdm

from .metrics_utils import calc_NRMSE
from .train_utils import train_model


#: Mapping of the ``name`` field in yml -> (actual network class name, network definition file)
_ARCH_MAP = {
    'SpkRes': ('Restormer', 'restormer_arch.py'),
    'Restormer': ('Restormer', 'restormer_arch.py'),
    'SpkRecon': ('SpkRecon', 'spkrecon_arch.py'),
    'SwinIR': ('SpkRecon', 'spkrecon_arch.py'),
    'EDSR': ('EDSR', 'edsr_arch.py'),
    'VAESR': ('VAESR', 'vae_arch.py'),
}

#: ``network_g`` parameter keys corresponding to each network class
_NET_G_PARAM_KEYS = {
    'Restormer': [
        'inp_channels', 'out_channels', 'dim', 'num_blocks',
        'num_refinement_blocks', 'heads', 'ffn_expansion_factor',
        'bias', 'LayerNorm_type', 'dual_pixel_task',
    ],
    'SpkRecon': [
        'in_chans', 'upscale', 'img_size', 'embed_dim', 'window_size',
        'img_range', 'num_heads', 'depths', 'upsampler',
        'resi_connection', 'mlp_ratio',
    ],
    'EDSR': ['num_channels', 'factor', 'width', 'depth', 'kernel_size'],
    'VAESR': [
        'num_channels', 'width', 'depth', 'kernel_size',
        'latent_dim', 'beta', 'kl_anneal_start',
    ],
}

#: Default values for optional VAESR parameters
_VAE_OPTIONAL_PARAMS = {'latent_dim': 32, 'beta': 1.0, 'kl_anneal_start': 0.1}

#: Reconstruction methods (used for branching during bin / nwb reconstruction)
_RECON_METHOD_KW = ('intp', 'krig', 'remove')


class Generate:
    """Data generator responsible for training data generation, model reconstruction and evaluation.

    Parameters
    ----------
    opt_path : str
        Path to the configuration yml file.
    """

    def __init__(self, opt_path):
        self.opt_path = opt_path
        self.read_yml()
        self.init_settings()

    # ==================== YAML I/O ====================

    def read_yml(self):
        """Read the yml config and store it in ``self.opt``."""
        with open(self.opt_path, 'r') as file:
            self.opt = yaml.safe_load(file)
        return self.opt

    def write_yml(self):
        """Write ``self.opt`` back to the yml file."""
        with open(self.opt_path, 'w') as file:
            yaml.dump(self.opt, file)

    # ==================== Recording-related helpers ====================

    @staticmethod
    def _load_recording(file_path):
        """Load a recording based on the file extension, and return (recording, channel_ids)."""
        file_type = file_path.split('.')
        if 'nwb' in file_type:
            recording = si.read_nwb(file_path, load_recording=True, load_sorting=False)
            channel_ids = recording.channel_ids
        elif 'bin' in file_type:
            directory = os.path.dirname(file_path)
            recording = se.read_spikeglx(directory)
            channel_ids = np.arange(len(recording.channel_ids))
        elif 'h5' in file_type:
            recording = se.MEArecRecordingExtractor(file_path)
            channel_ids = recording.channel_ids.astype(int)
        else:
            raise ValueError(f'Unsupported file type: {file_path}')
        return recording, channel_ids

    @staticmethod
    def _compute_window_bounds(j, window, total_sample):
        """Compute the sliding window start_frame / end_frame, handling out-of-bounds."""
        start_frame = j
        end_frame = j + window
        if end_frame > total_sample:
            start_frame = total_sample - window
            end_frame = total_sample
        return start_frame, end_frame

    @staticmethod
    def _get_bad_good_chans(recording, channel_ids, factor):
        """Compute bad channels / good channels according to the factor pattern."""
        bad_chans = recording.channel_ids[channel_ids % factor == 1]
        good_chans = recording.channel_ids[~np.isin(recording.channel_ids, bad_chans)]
        return bad_chans, good_chans

    @staticmethod
    def _build_interp_recs(recording, bad_chans, good_chans):
        """Build the kriging-interpolated recording and the zero-filled (remove) interpolated recording."""
        krig_rec = si.interpolate_bad_channels(recording, bad_chans, weights=None)
        rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
        rmv_rec = si.interpolate_bad_channels(recording, bad_chans, weights=rmv_weights)
        return krig_rec, rmv_rec

    @staticmethod
    def _pad_to_channel_num(lq, gt, channel_num):
        """Pad lq / gt with zeros on the right along the channel dimension up to channel_num."""
        pad_width = channel_num - lq.shape[1]
        lq = np.pad(lq, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        gt = np.pad(gt, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        return lq, gt

    @staticmethod
    def _save_trace_pair(lq, gt, save_path, index):
        """Save the (lq, gt) trace pair to the ``save_path/<index>/`` directory."""
        index_path = os.path.join(save_path, str(index))
        os.makedirs(index_path, exist_ok=True)
        np.save(os.path.join(index_path, 'lq.npy'), lq)
        np.save(os.path.join(index_path, 'gt.npy'), gt)

    def _build_input_rec(self, recording, bad_chans, good_chans):
        """Build the model input recording according to ``is_intp`` / ``is_krig``."""
        if self.is_intp:
            if self.is_krig:
                return si.interpolate_bad_channels(recording, bad_chans, weights=None)
            rmv_weights = np.zeros([len(good_chans), len(bad_chans)])
            return si.interpolate_bad_channels(recording, bad_chans, weights=rmv_weights)
        return recording.remove_channels(remove_channel_ids=bad_chans)

    def _make_traces_path(self):
        """Generate the directory name for saving train / test traces."""
        if self.is_intp:
            prefix = ('generate_krig_traces_factor_{}' if self.is_krig
                      else 'generate_zero_traces_factor_{}')
        else:
            prefix = 'generate_lr_traces_factor_{}'
        return prefix.format(self.factor)

    # ==================== Initialization ====================

    def init_settings(self):
        """Parse the yml config and initialize the various attributes."""
        opt = self.opt
        s = opt['settings']

        # 1. train settings
        self.is_intp = s['is_intp']
        self.is_random = s['is_random']
        self.is_krig = s['is_krig']
        self.slide_window = s['slide_window']
        self.factor = s['factor']  # proportion of abnormal channels
        self.rec_path = s['rec_path']  # origin recording path
        self.base_path = s['save_path']  # save path
        self.train_prop = s['train_prop']  # partial frames used for training
        self.test_prop = s['test_prop']

        # 2. restored settings
        self.method = s['recon_method']  # reconstruction method
        self.is_whole = s['is_whole']  # whole / partial pattern
        self.base_loss_path = s['base_loss_path']  # base loss path

        # 3. model type
        self.model_type = opt['name']

        # 4. recording
        self.recording, self.channel_ids = self._load_recording(self.rec_path)

        # 5. window / pad_width
        channel_num = opt.get('settings', {}).get('channel_num')
        self.window = channel_num if channel_num is not None else len(self.channel_ids)
        self.pad_width = self.window - len(self.channel_ids)

        # 6. bad_chans / good_chans
        self._init_bad_good_chans()

        # 7. train_list / test_list + z-score statistics
        self._init_train_test_and_norm()

        # 8. input rec / save path
        self.input_rec = self._build_input_rec(self.recording, self.bad_chans, self.good_chans)
        self.traces_path = self._make_traces_path()

    def _init_bad_good_chans(self):
        """Initialize bad / good channels: prefer loading/saving via ``chl_path``, otherwise compute by factor."""
        chl_path = self.opt.get('settings', {}).get('chl_path')
        if chl_path is not None:
            chl_file = os.path.join(chl_path, 'factor_{}.npy'.format(self.factor))
            if not os.path.exists(chl_file):
                chl_num = np.sum(self.channel_ids % self.factor == 1)
                bad_idx = np.random.choice(self.channel_ids, size=chl_num, replace=False)
                bad_chans = self.recording.channel_ids[bad_idx]
                np.save(chl_file, bad_chans)
            else:
                bad_chans = np.load(chl_file)
            good_chans = self.recording.channel_ids[~np.isin(self.recording.channel_ids, bad_chans)]
        else:
            bad_chans, good_chans = self._get_bad_good_chans(
                self.recording, self.channel_ids, self.factor)

        self.bad_chans = bad_chans
        self.good_chans = good_chans

    def _init_train_test_and_norm(self):
        """Generate the train / test window lists, and compute (or load) z-score normalization statistics."""
        self.total_sample = self.recording.get_total_samples()
        self.sliding_window = self.window - self.slide_window
        self.window_num = len(range(0, self.total_sample, self.sliding_window))

        start_frame_list = range(0, self.total_sample, self.sliding_window)
        train_num = int(len(start_frame_list) // self.train_prop)
        test_num = int(len(start_frame_list) // self.test_prop)

        if self.is_random:  # select random frames for training
            self.train_list = np.random.choice(start_frame_list, size=train_num, replace=False)
            self.test_list = np.random.choice(
                np.setdiff1d(start_frame_list, self.train_list), size=test_num, replace=False)
        else:
            self.train_list = start_frame_list[:train_num]
            self.test_list = start_frame_list[train_num:train_num + test_num]

        # z-score statistics (new field: norm_path)
        self.norm_path = self.opt.get('settings', {}).get('norm_path')
        if self.norm_path is not None:
            self._compute_norm_stats(train_num)

    def _compute_norm_stats(self, train_num):
        """Compute and cache the mean / std of gt (full / training segment) and lq (full)."""
        norm_file = os.path.join(self.norm_path, 'factor_{}.npy'.format(self.factor))
        lq_norm_file = os.path.join(self.norm_path, 'lq_factor_{}.npy'.format(self.factor))

        if not os.path.exists(norm_file):
            gt_input = self.recording.get_traces(start_frame=0, end_frame=train_num)  # training segment
            self.mean_chl = np.mean(gt_input, axis=0)
            self.std_chl = np.std(gt_input, axis=0)
            np.save(norm_file, [self.mean_chl, self.std_chl])
        else:
            self.mean_chl, self.std_chl = np.load(norm_file)

        if not os.path.exists(lq_norm_file):
            lq_input = self.recording.get_traces(
                start_frame=0, end_frame=self.recording.get_total_samples())
            self.lq_mean_chl = np.mean(lq_input, axis=0)
            self.lq_std_chl = np.std(lq_input, axis=0)
            np.save(lq_norm_file, [self.lq_mean_chl, self.lq_std_chl])
        else:
            self.lq_mean_chl, self.lq_std_chl = np.load(lq_norm_file)

    # ==================== Training data generation ====================

    def _save_trace_list(self, trace_list, lq_rec, save_path, norm_data=None):
        """Save the (lq, gt) traces of a list of windows to ``save_path``."""
        for j in tqdm(trace_list):
            start_frame, end_frame = self._compute_window_bounds(j, self.window, self.total_sample)

            lq = lq_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
            gt = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame)

            if norm_data is not None:  # z-score
                mean, std = norm_data
                lq = (lq - mean) / std
                gt = (gt - mean) / std

            lq = np.pad(lq, ((0, 0), (0, self.pad_width)), constant_values=0)
            gt = np.pad(gt, ((0, 0), (0, self.pad_width)), constant_values=0)

            self._save_trace_pair(lq, gt, save_path, start_frame)

    def generate_train_list(self):
        """Generate the training / test datasets for a single file and save them to disk."""
        save_path = os.path.join(self.base_path, self.traces_path)
        train_path = os.path.join(save_path, 'train')
        test_path = os.path.join(save_path, 'test')

        norm_data = (self.mean_chl, self.std_chl) if self.norm_path is not None else None

        # save train dataset
        self._save_trace_list(self.train_list, self.input_rec, train_path, norm_data)
        print("train datasets generate done!!")

        # save test dataset
        self._save_trace_list(self.test_list, self.input_rec, test_path, norm_data)
        print("recon datasets generate done!!")

    def generate_train_list_mix(self):
        """Generate training / test datasets from multiple (mixture) files and save them to disk."""
        train_index_i, test_index_i = 0, 0
        mixture = self.opt['settings']['mixture']  # mixture files
        channel_num = self.opt['settings']['channel_num']

        for mx_i in mixture:
            mx_path = mixture[mx_i]
            recording, channel_ids = self._load_recording(mx_path)

            bad_chans, good_chans = self._get_bad_good_chans(recording, channel_ids, self.factor)
            rec = self._build_input_rec(recording, bad_chans, good_chans)

            total_sample = recording.get_total_samples()
            sliding_window = channel_num - self.slide_window
            start_frame_list = range(0, total_sample, sliding_window)
            train_num = int(len(start_frame_list) // self.train_prop)
            test_num = int(len(start_frame_list) // self.test_prop)

            if self.is_random:
                train_list = np.random.choice(start_frame_list, size=train_num, replace=False)
                test_list = np.random.choice(
                    np.setdiff1d(start_frame_list, train_list), size=test_num, replace=False)
            else:
                train_list = start_frame_list[:train_num]
                test_list = start_frame_list[train_num:train_num + test_num]

            save_path = os.path.join(self.base_path, self._make_traces_path())
            train_path = os.path.join(save_path, 'train')
            test_path = os.path.join(save_path, 'test')

            # save train dataset
            for j in tqdm(train_list):
                start_frame, end_frame = self._compute_window_bounds(j, channel_num, total_sample)
                lq = rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
                lq, gt = self._pad_to_channel_num(lq, gt, channel_num)
                self._save_trace_pair(lq, gt, train_path, train_index_i)
                train_index_i += 1
            print(f"{mx_path}: train datasets generate done!!")

            # save test dataset
            for j in tqdm(test_list):
                start_frame, end_frame = self._compute_window_bounds(j, channel_num, total_sample)
                lq = rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
                lq, gt = self._pad_to_channel_num(lq, gt, channel_num)
                self._save_trace_pair(lq, gt, test_path, test_index_i)
                test_index_i += 1
            print(f"{mx_path}: test datasets generate done!!")

    # ==================== Model ====================

    def generate_model(self):
        """Instantiate the network model according to the ``network_g`` config."""
        if self.model_type not in _ARCH_MAP:
            raise ValueError(f'Unsupported model type: {self.model_type}')

        arch_class_name, arch_file = _ARCH_MAP[self.model_type]
        arch_module = run_path(os.path.join('basicsr', 'models', 'archs', arch_file))

        keys = _NET_G_PARAM_KEYS[arch_class_name]
        net_g = self.opt['network_g']
        if arch_class_name == 'VAESR':  # optional parameters with defaults
            parameters = {k: net_g.get(k, _VAE_OPTIONAL_PARAMS[k]) for k in keys}
        else:
            parameters = {k: net_g[k] for k in keys}

        return arch_module[arch_class_name](**parameters)

    def prepare_model(self, weights_path=None):
        """Load the model, move it to the device, load weights and set to eval mode."""
        model = self.generate_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        if weights_path is not None:
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint['params'])
        model.eval()
        return model, device

    def generate_train_weights(self):
        """Start model training (calls ``train_model``)."""
        train_model(self.opt_path)

    @staticmethod
    def _to_model_input(lq, device):
        """Convert a (T, C) trace into a model input tensor of shape (1, 1, T, C)."""
        lq = np.expand_dims(lq, axis=2)
        lq = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0).to(device)
        return lq

    def _model_predict(self, lq, model, device, norm_data=None, pad_width=None):
        """Run model inference on an lq trace and return the reconstructed trace.

        Parameters
        ----------
        lq : np.ndarray
            Input trace of shape (T, C).
        norm_data : tuple or None
            (mean_chl, std_chl); if provided, z-score normalization is applied before
            inference and de-normalization after the output.
        pad_width : int or None
            Zero-padding width along the channel dimension; defaults to ``self.pad_width``.
        """
        if pad_width is None:
            pad_width = self.pad_width

        if norm_data is not None:  # z-score
            mean, std = norm_data
            lq = (lq - mean) / std

        lq = np.pad(lq, ((0, 0), (0, pad_width)), constant_values=0)

        intp_trace = model(self._to_model_input(lq, device))
        intp_trace = intp_trace.squeeze().cpu().detach().numpy()

        if pad_width > 0:
            intp_trace = intp_trace[:, :-pad_width]

        if norm_data is not None:  # de-normalize
            intp_trace = intp_trace * std + mean

        return intp_trace

    # ==================== Common reconstruction logic ====================

    def _compute_pattern(self, total_sample):
        """Compute the partial_sample and pattern name for whole / partial modes."""
        if self.is_whole:
            return 0, 'whole'
        return total_sample // self.train_prop, 'partial'

    def _prepare_loss_path(self, pattern, factor, warn_if_exists=True):
        """Create / return the loss saving directory, optionally checking and warning if files already exist."""
        loss_path = os.path.join(self.base_loss_path, '{}_factor{}'.format(pattern, factor))
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)
        elif warn_if_exists:
            file_name = os.path.join(loss_path, '{}_mse_loss_dict.npy'.format(self.method))
            if os.path.exists(file_name):
                warnings.warn(f"The file '{file_name}' exists in the folder.", Warning)
        return loss_path

    def _run_recon_loop(self, recording, total_sample, window, sliding_window, window_num,
                        partial_sample, loss_path, method, get_intp_trace, write_func):
        """Core sliding-window reconstruction loop: inference -> compute MSE/NRMSE -> write back to file."""
        mse_loss_dict = np.zeros([window_num])
        nrmse_loss_dict = np.zeros([window_num])

        step = 0
        for j in tqdm(range(0, total_sample, sliding_window)):
            start_frame, end_frame = self._compute_window_bounds(j, window, total_sample)
            print(f"start_frame:{start_frame}, end_frame:{end_frame}")

            gt = recording.get_traces(start_frame=start_frame, end_frame=end_frame)  # ground truth

            if end_frame <= partial_sample:
                intp_trace = gt
            else:
                intp_trace = get_intp_trace(start_frame, end_frame, gt)

            # calc MSE
            mse_loss = mean_squared_error(gt, intp_trace)
            mse_loss_dict[step] = mse_loss
            print(f"{method}_mse_loss:{mse_loss}")

            # calc NRMSE
            nrmse_loss = calc_NRMSE(y_pred=intp_trace.astype(float), y_gt=gt.astype(float))
            nrmse_loss_dict[step] = nrmse_loss
            print(f"{method}_nrmse_loss:{nrmse_loss}")

            # write restored data
            write_func(start_frame, end_frame, intp_trace)
            step += 1

        np.save(os.path.join(loss_path, f'{method}_mse_loss_dict.npy'), mse_loss_dict)
        np.save(os.path.join(loss_path, f'{method}_nrmse_loss_dict.npy'), nrmse_loss_dict)

    # ==================== Reconstruction by format ====================
    def generate_recon_binfile(self, res_file_name=None):
        """Reconstruct a SpikeGLX bin file (written back via memmap), and compute MSE/NRMSE."""
        method = self.method
        factor = self.factor
        slide_window = self.slide_window

        weights_path = self.opt.get('settings', {}).get('weights_path')
        model, device = self.prepare_model(weights_path)

        recording, channel_ids = self._load_recording(self.rec_path)
        window = len(channel_ids)
        total_sample = recording.get_total_samples()
        sliding_window = window - slide_window
        window_num = len(range(0, total_sample, sliding_window))

        bad_chans, good_chans = self._get_bad_good_chans(recording, channel_ids, factor)
        krig_rec, rmv_rec = self._build_interp_recs(recording, bad_chans, good_chans)

        partial_sample, pattern = self._compute_pattern(total_sample)
        loss_path = self._prepare_loss_path(pattern, factor, warn_if_exists=False)

        signals_info_dict = {}
        for info in scan_files(res_file_name):
            key = (info['seg_index'], info['stream_name'])
            assert key not in signals_info_dict
            signals_info_dict[key] = info

            num_chan = info['num_chan']
            sample_length = info['sample_length']

            # create memmap
            data = np.memmap(info['bin_file'], dtype='int16', mode='r+', offset=0,
                             order='C', shape=(sample_length, num_chan))

            def get_intp_trace(start_frame, end_frame, gt):
                lq_krig = krig_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                if method == 'krig':
                    return lq_krig
                elif method == 'remove':
                    return rmv_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
                else:
                    return self._model_predict(lq_krig, model, device, pad_width=0).astype(np.int16)

                # raise ValueError(f'Unsupported recon method: {method}')

            def write_func(start_frame, end_frame, intp_trace):
                data[start_frame:end_frame, :-1] = intp_trace
                data.flush()

            self._run_recon_loop(recording, total_sample, window, sliding_window, window_num,
                                 partial_sample, loss_path, method, get_intp_trace, write_func)

    def generate_recon_h5file(self, res_file_name='res_file.h5', mode='zero_shot_gcl'):
        """Reconstruct a MEArec h5 file, and compute MSE/NRMSE.

        Parameters
        ----------
        res_file_name : str
            Path of the h5 file to write back (must already exist and contain the
            ``recordings`` dataset).
        mode : str
            ``'zero_shot_gcl'`` uses lq statistics for de-normalization, other modes
            use gt statistics.
        """
        # z-score statistics
        norm_data = None
        if self.norm_path is not None:
            if mode == 'zero_shot_gcl':
                norm_data = (self.lq_mean_chl, self.lq_std_chl)
            else:
                norm_data = (self.mean_chl, self.std_chl)

        # model
        weights_path = self.opt.get('settings', {}).get('weights_path')
        if weights_path is None:
            raise ValueError("`settings.weights_path` must be set in yml for reconstruction.")
        model, device = self.prepare_model(weights_path)

        # pattern & loss path
        partial_sample, pattern = self._compute_pattern(self.total_sample)
        loss_path = self._prepare_loss_path(pattern, self.factor, warn_if_exists=True)

        def get_intp_trace(start_frame, end_frame, gt):
            lq = self.input_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
            return self._model_predict(lq, model, device, norm_data=norm_data)

        with h5py.File(res_file_name, 'r+') as io:  # h5
            res_rec = io['recordings']  # time × channel
            print('size of rec: {}'.format(res_rec.shape))

            def write_func(start_frame, end_frame, intp_trace):
                res_rec[start_frame:end_frame] = intp_trace

            self._run_recon_loop(self.recording, self.total_sample, self.window,
                                 self.sliding_window, self.window_num, partial_sample,
                                 loss_path, self.method, get_intp_trace, write_func)

    def generate_recon_nwbfile(self, res_file_name='res_file.nwb'):
        """Reconstruct an NWB file, and compute MSE/NRMSE."""
        method = self.method
        factor = self.factor
        slide_window = self.slide_window

        weights_path = self.opt.get('settings', {}).get('weights_path')
        if weights_path is None:
            raise ValueError("`settings.weights_path` must be set in yml for reconstruction.")
        model, device = self.prepare_model(weights_path)

        recording, channel_ids = self._load_recording(self.rec_path)
        window = len(channel_ids)
        total_sample = recording.get_total_samples()
        sliding_window = window - slide_window
        window_num = len(range(0, total_sample, sliding_window))

        bad_chans, good_chans = self._get_bad_good_chans(recording, channel_ids, factor)
        krig_rec, rmv_rec = self._build_interp_recs(recording, bad_chans, good_chans)
        lr_rec = recording.remove_channels(remove_channel_ids=bad_chans)

        partial_sample, pattern = self._compute_pattern(total_sample)
        loss_path = self._prepare_loss_path(pattern, factor, warn_if_exists=True)

        def get_intp_trace(start_frame, end_frame, gt):
            if method == 'krig':
                return krig_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
            elif method == 'remove':
                return rmv_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
            # model reconstruction: select input according to is_intp / is_krig
            lq_rec = self._build_input_rec(recording, bad_chans, good_chans)
            lq = lq_rec.get_traces(start_frame=start_frame, end_frame=end_frame)
            return self._model_predict(lq, model, device, pad_width=0)

        with NWBHDF5IO(res_file_name, 'r+') as io:  # h5 / nwb
            restored_nwbfile = io.read()

            def write_func(start_frame, end_frame, intp_trace):
                restored_nwbfile.acquisition["ElectricalSeries"].data[start_frame:end_frame] = intp_trace

            self._run_recon_loop(recording, total_sample, window, sliding_window, window_num,
                                 partial_sample, loss_path, method, get_intp_trace, write_func)